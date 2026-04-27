from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

import numpy as np

from src.aggregation.feature_mapper import FeatureMapper
from src.aggregation.score_normalizer import ScoreNormalizer
from src.aggregation.calibration import get_calibrator, PassThroughCalibrator
from src.aggregation.weight_manager import WeightManager
from src.aggregation.risk_assessment import assess_truthlens_risks, RiskConfig
from src.aggregation.score_explainer import ScoreExplainer
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator
from src.aggregation.aggregation_config import AggregationConfig
from src.aggregation.score_schema import (
    TruthLensAggregationOutputModel,
    TruthLensScoreModel,
    TruthLensRiskModel,
    ExplanationModel,
    TaskScore,
    RiskValue,
)
from src.aggregation.aggregation_validator import AggregationValidator


logger = logging.getLogger(__name__)
EPS = 1e-12


class AggregationPipeline:

    def __init__(
        self,
        *,
        config: Optional[AggregationConfig] = None,
    ) -> None:

        self.config = config or AggregationConfig()

        self.mapper = FeatureMapper(strict=self.config.strict_mode)

        self.normalizer = ScoreNormalizer(
            method=self.config.normalization.method,
            strict=self.config.strict_mode,
            clip=self.config.normalization.clip,
        )

        # Calibrator: if not yet fitted we fall back to passthrough automatically
        self.calibrator = get_calibrator(self.config.calibration.method)

        self.weight_manager = WeightManager(
            smoothing=self.config.weights.smoothing,
        )

        self.calculator = TruthLensScoreCalculator(
            uncertainty_penalty=self.config.risk.uncertainty_penalty,
        )

        self.explainer = ScoreExplainer(
            method=self.config.attribution.method
        )

        self.risk_config = RiskConfig()
        self.validator = AggregationValidator()

        logger.info("[AggregationPipeline] Initialized")

    # =====================================================
    # MAIN
    # =====================================================

    def run(
        self,
        model_outputs: Dict[str, Any],
        *,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:

        if not isinstance(model_outputs, dict):
            raise ValueError("model_outputs must be dict")

        # =========================
        # 1. FEATURE MAPPING
        # =========================
        profile = self.mapper.map_from_model_outputs(model_outputs)
        confidence = self.mapper.extract_confidence(model_outputs)

        # =========================
        # 2. UNCERTAINTY (vectorized per task)
        # =========================
        entropy = self._compute_entropy(model_outputs)

        # =========================
        # 3. NORMALIZATION (per-section, fit+transform in one pass)
        # =========================
        normalized = self._normalize_profile(profile)

        # =========================
        # 4. CALIBRATION (passthrough if unfitted)
        # =========================
        calibrated = self._calibrate_profile(normalized)

        # =========================
        # 5. EXPLANATION (optional)
        # =========================
        explanation_scores: Dict[str, float] = {}
        explanations_raw: Dict[str, Any] = {}

        if self.config.enable_explanations:
            explanations_raw = self.explainer.explain_profile(calibrated)
            explanation_scores = explanations_raw.get("section_scores", {})

        # =========================
        # 6. ADAPTIVE WEIGHTS
        # =========================
        adaptive_weights = self.weight_manager.get_adaptive_weights(
            confidence=confidence if self.config.weights.use_confidence else None,
            entropy=entropy if self.config.weights.use_entropy else None,
            explanation_scores=(
                explanation_scores if self.config.weights.use_explainability else None
            ),
        )

        # =========================
        # 7. SCORING — adaptive weights forwarded into calculator
        # =========================
        scores_raw = self.calculator.compute_scores(
            calibrated,
            confidence=confidence,
            entropy=entropy,
            explanation_scores=explanation_scores,
            weights=adaptive_weights,
        )

        # =========================
        # 8. RISK — use correct key names
        # =========================
        risks_dict: Dict[str, Any] = {}

        if self.config.enable_risk:
            def _safe_risk(v: Any) -> float:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return 0.0
                return float(np.clip(fv if np.isfinite(fv) else 0.0, 0.0, 1.0))

            risk_input = {
                "truthlens_manipulation_risk": _safe_risk(scores_raw.get("manipulation_risk", 0.0)),
                "truthlens_credibility_score": _safe_risk(scores_raw.get("credibility_score", 0.0)),
                "truthlens_final_score": _safe_risk(scores_raw.get("final_score", 0.0)),
            }
            risks_dict = assess_truthlens_risks(
                risk_input,
                probabilities=None,
                config=self.risk_config,
            )

        # =========================
        # 9. BUILD TYPED MODELS
        # =========================
        section_scores = scores_raw.get("section_scores", {})

        def _safe_score(v: Any) -> float:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return 0.0
            return float(np.clip(fv if np.isfinite(fv) else 0.0, 0.0, 1.0))

        scores_model = TruthLensScoreModel(
            tasks={
                section: TaskScore(score=_safe_score(val))
                for section, val in section_scores.items()
            },
            manipulation_risk=float(np.clip(scores_raw.get("manipulation_risk", 0.0), 0.0, 1.0)),
            credibility_score=float(np.clip(scores_raw.get("credibility_score", 0.0), 0.0, 1.0)),
            final_score=float(np.clip(scores_raw.get("final_score", 0.0), 0.0, 1.0)),
        )

        risks_model = self._build_risk_model(risks_dict)
        explanations_model = self._build_explanation_model(explanations_raw)

        # =========================
        # 10. ASSEMBLE RESULT
        # =========================
        result = {
            "schema_version": self.config.config_version,
            "model_version": "truthlens-v2",

            "scores": scores_model.model_dump(),
            "raw_scores": {
                k: float(v)
                for k, v in scores_raw.items()
                if isinstance(v, (int, float)) and np.isfinite(v)
            },

            "risks": risks_model.model_dump(),
            "explanations": explanations_model.model_dump(),

            "analysis_modules": {
                "weights": adaptive_weights,
                "entropy": entropy,
            },
        }

        # =========================
        # 11. GRAPH INTEGRATION
        # =========================
        graph_output = model_outputs.get("graph_output")

        if graph_output is not None:
            try:
                if hasattr(graph_output, "to_dict"):
                    result["analysis_modules"]["graph"] = graph_output.to_dict()
                else:
                    result["analysis_modules"]["graph"] = graph_output

                if hasattr(graph_output, "explanation"):
                    result["analysis_modules"]["graph_explanation"] = graph_output.explanation
                elif isinstance(graph_output, dict):
                    result["analysis_modules"]["graph_explanation"] = graph_output.get("explanation")
            except Exception as e:
                logger.warning("[AggregationPipeline] Graph injection failed: %s", e)

        # =========================
        # 12. VALIDATION
        # =========================
        flat_scores = {
            "credibility_score": scores_model.credibility_score,
            "manipulation_risk": scores_model.manipulation_risk,
            "final_score": scores_model.final_score,
        }
        validation = self.validator.validate({"scores": flat_scores})
        result["analysis_modules"]["validation"] = validation

        if not validation["valid"]:
            logger.warning(
                "[AggregationPipeline] Validation issues: %s",
                validation["issues"],
            )

        # =========================
        # 13. FINAL SCHEMA VALIDATION
        # =========================
        validated = TruthLensAggregationOutputModel(**result)
        return validated.model_dump()

    # =====================================================
    # NORMALIZATION — fit+transform per section (one pass, no recompute)
    # =====================================================

    def _normalize_profile(self, profile: Dict[str, Any]) -> Dict[str, Dict[str, float]]:

        out: Dict[str, Dict[str, float]] = {}

        for section, feats in profile.items():

            if not isinstance(feats, dict):
                continue

            values = list(feats.values())

            if not values:
                out[section] = feats
                continue

            try:
                # Single fit_transform call — no repeated computation
                norm = self.normalizer.fit_transform(values)
                out[section] = dict(zip(feats.keys(), norm.tolist() if hasattr(norm, "tolist") else norm))
            except Exception:
                out[section] = feats

        return out

    # =====================================================
    # CALIBRATION — passthrough for scalar profile values
    # (calibrators are designed for logit arrays, not scalars)
    # =====================================================

    def _calibrate_profile(self, profile: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:

        # Only apply calibration if the calibrator has been fitted on real data;
        # unfitted calibrators return passthrough (already clipped [0,1]).
        if not getattr(self.calibrator, "fitted", False):
            return profile

        # If fitted (e.g. after offline calibration training), apply per-feature.
        out: Dict[str, Dict[str, float]] = {}

        for section, feats in profile.items():

            new_feats: Dict[str, float] = {}

            keys = list(feats.keys())
            values = np.array(list(feats.values()), dtype=np.float64)

            try:
                calibrated = np.clip(
                    self.calibrator.transform(values.reshape(1, -1)).ravel(),
                    0.0, 1.0,
                )
                new_feats = dict(zip(keys, calibrated.tolist()))
            except Exception:
                new_feats = {k: float(v) for k, v in feats.items()}

            out[section] = new_feats

        return out

    # =====================================================
    # ENTROPY — vectorized per task, normalized to [0,1]
    # =====================================================

    def _compute_entropy(self, outputs: Dict[str, Any]) -> Dict[str, float]:

        entropy: Dict[str, float] = {}

        for task, out in outputs.items():

            if not isinstance(out, dict):
                continue

            probs = out.get("probabilities")

            if probs is None:
                continue

            probs_arr = np.nan_to_num(
                np.asarray(probs, dtype=np.float64),
                nan=0.0, posinf=1.0, neginf=0.0,
            )

            if probs_arr.ndim == 2:
                probs_arr = probs_arr[0]

            probs_arr = np.clip(probs_arr, EPS, 1.0)
            total = np.sum(probs_arr)
            if not np.isfinite(total) or total <= 0.0:
                entropy[task] = 0.0
                continue

            probs_arr = probs_arr / total
            raw_ent = float(-np.sum(probs_arr * np.log(probs_arr)))
            max_ent = float(np.log(max(probs_arr.size, 2)))
            ent_val = raw_ent / max_ent if max_ent > 0.0 else 0.0
            entropy[task] = float(np.clip(ent_val, 0.0, 1.0))

        return entropy

    # =====================================================
    # RISK MODEL BUILDER
    # =====================================================

    def _build_risk_model(self, risks_dict: Dict[str, Any]) -> TruthLensRiskModel:

        def _rv(data: Any) -> Optional[RiskValue]:
            if not isinstance(data, dict):
                return None
            level = data.get("level")
            score = data.get("score")
            if level not in ("LOW", "MEDIUM", "HIGH"):
                return None
            return RiskValue(level=level, score=score)

        return TruthLensRiskModel(
            manipulation_risk=_rv(risks_dict.get("manipulation_risk")),
            credibility_level=_rv(risks_dict.get("credibility_level")),
            overall_truthlens_rating=_rv(risks_dict.get("overall_truthlens_rating")),
        )

    # =====================================================
    # EXPLANATION MODEL BUILDER
    # =====================================================

    def _build_explanation_model(self, raw: Dict[str, Any]) -> ExplanationModel:

        if not raw:
            return ExplanationModel(sections={})

        section_scores = raw.get("section_scores", {})
        top_features = raw.get("top_features", [])

        method = self.config.attribution.method
        sections: Dict[str, Any] = {}

        for section, score in section_scores.items():

            section_feats = [
                (k, v)
                for s, k, v in top_features
                if s == section
            ]

            attributions = [
                {
                    "token": str(k),
                    "importance": float(abs(v)),
                    "contribution": float(v),
                    "direction": "positive" if v >= 0.0 else "negative",
                }
                for k, v in section_feats[: self.config.attribution.top_k]
            ]

            sections[section] = {
                "method": method,
                "top_features": [a["token"] for a in attributions],
                "attributions": attributions,
                "section_score": float(np.clip(score, 0.0, 1.0)),
            }

        return ExplanationModel(sections=sections)

    # =====================================================
    # BATCH — sequential (pipeline state is per-call safe)
    # =====================================================

    def run_batch(self, batch_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.run(x) for x in batch_outputs]
