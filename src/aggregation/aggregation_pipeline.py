from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

import numpy as np

from src.aggregation.feature_mapper import FeatureMapper
from src.aggregation.score_normalizer import ScoreNormalizer
from src.aggregation.calibration import get_calibrator
from src.aggregation.weight_manager import WeightManager
from src.aggregation.risk_assessment import assess_truthlens_risks, RiskConfig
from src.aggregation.score_explainer import ScoreExplainer
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator
from src.aggregation.aggregation_config import AggregationConfig
from src.aggregation.score_schema import TruthLensAggregationOutputModel

# 🔥 NEW
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

        # =========================
        # COMPONENTS (CONFIG-DRIVEN)
        # =========================
        self.mapper = FeatureMapper(strict=self.config.strict_mode)

        self.normalizer = ScoreNormalizer(
            method=self.config.normalization.method,
            strict=self.config.strict_mode,
            clip=self.config.normalization.clip,
        )

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

        # 🔥 NEW: VALIDATOR
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
        # 2. UNCERTAINTY
        # =========================
        entropy = self._compute_entropy(model_outputs)

        # =========================
        # 3. NORMALIZATION
        # =========================
        normalized = self._normalize_profile(profile)

        # =========================
        # 4. CALIBRATION
        # =========================
        calibrated = self._calibrate_profile(normalized)

        # =========================
        # 5. EXPLANATION (OPTIONAL)
        # =========================
        explanation_scores = {}
        explanations = {}

        if self.config.enable_explanations:
            explanations = self.explainer.explain_profile(calibrated)
            explanation_scores = explanations.get("section_scores", {})

        # =========================
        # 6. ADAPTIVE WEIGHTS
        # =========================
        weights = self.weight_manager.get_adaptive_weights(
            confidence=confidence if self.config.weights.use_confidence else None,
            entropy=entropy if self.config.weights.use_entropy else None,
            explanation_scores=(
                explanation_scores if self.config.weights.use_explainability else None
            ),
        )

        # =========================
        # 7. SCORING
        # =========================
        scores = self.calculator.compute_scores(
            calibrated,
            confidence=confidence,
            entropy=entropy,
            explanation_scores=explanation_scores,
        )

        # =========================
        # 8. RISK
        # =========================
        risks = {}

        if self.config.enable_risk:
            risks = assess_truthlens_risks(
                scores,
                probabilities=None,
                config=self.risk_config,
            )

        # =========================
        # 9. OUTPUT (PRE-VALIDATION)
        # =========================
        result = {
            "schema_version": self.config.config_version,
            "model_version": "truthlens-v2",

            "scores": scores,
            "raw_scores": scores,

            "risks": risks,
            "explanations": explanations,

            "analysis_modules": {
                "weights": weights,
                "entropy": entropy,
            },
        }
        
        # =====================================================
        #  GRAPH INTEGRATION (NEW)
        # =====================================================
        
        graph_output = model_outputs.get("graph_output")
        
        if graph_output is not None:
        
            try:
                # full structured graph output
                if hasattr(graph_output, "to_dict"):
                    result["analysis_modules"]["graph"] = graph_output.to_dict()
                else:
                    result["analysis_modules"]["graph"] = graph_output
        
                #  graph explanation
                if hasattr(graph_output, "explanation"):
                    result["analysis_modules"]["graph_explanation"] = graph_output.explanation
                elif isinstance(graph_output, dict):
                    result["analysis_modules"]["graph_explanation"] = graph_output.get("explanation")
        
            except Exception as e:
                logger.warning("[AggregationPipeline] Graph injection failed: %s", e)
        


        # =========================
        # 10. VALIDATION ( NEW)
        # =========================
        validation = self.validator.validate(result)

        result["analysis_modules"]["validation"] = validation

        #  OPTIONAL LOGGING
        if not validation["valid"]:
            logger.warning(
                "[AggregationPipeline] Validation issues: %s",
                validation["issues"]
            )

        # =========================
        # 11. FINAL SCHEMA VALIDATION
        # =========================
        validated = TruthLensAggregationOutputModel(**result)

        return validated.model_dump()

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize_profile(self, profile):

        out = {}

        for section, feats in profile.items():

            if not isinstance(feats, dict):
                continue

            values = list(feats.values())

            try:
                norm = self.normalizer.fit_transform(values)
                out[section] = dict(zip(feats.keys(), norm))
            except Exception:
                out[section] = feats

        return out

    # =====================================================
    # CALIBRATION
    # =====================================================

    def _calibrate_profile(self, profile):

        out = {}

        for section, feats in profile.items():

            new_feats = {}

            for k, v in feats.items():
                try:
                    val = self.calibrator.transform(np.array([[v]]))[0][0]
                except Exception:
                    val = v

                new_feats[k] = float(val)

            out[section] = new_feats

        return out

    # =====================================================
    # ENTROPY
    # =====================================================

    def _compute_entropy(self, outputs): 

        entropy = {}

        for task, out in outputs.items():

            probs = out.get("probabilities")

            if probs is None:
                continue

            probs = np.asarray(probs)

            if probs.ndim == 2:
                probs = probs[0]

            ent = -np.sum(probs * np.log(probs + EPS))

            entropy[task] = float(ent)

        return entropy

    # =====================================================
    # BATCH
    # =====================================================

    def run_batch(self, batch_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.run(x) for x in batch_outputs]