from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List

from src.aggregation.feature_mapper import FeatureMapper
from src.aggregation.score_normalizer import ScoreNormalizer
from src.aggregation.calibration import get_calibrator
from src.aggregation.weight_manager import WeightManager
from src.aggregation.risk_assessment import assess_truthlens_risks, RiskConfig
from src.aggregation.score_explainer import ScoreExplainer
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator
from src.aggregation.aggregation_config import AggregationConfig
from src.aggregation.score_schema import TruthLensAggregationOutputModel


logger = logging.getLogger(__name__)


class AggregationPipeline:

    def __init__(
        self,
        *,
        config: Optional[AggregationConfig] = None,
    ) -> None:

        self.config = config or AggregationConfig()

        # Core components
        self.mapper = FeatureMapper(strict=self.config.strict_mode)
        self.normalizer = ScoreNormalizer(
            method=self.config.normalization.method,
            strict=self.config.strict_mode,
        )

        self.calibrator = get_calibrator("sigmoid")
        self.weight_manager = WeightManager()
        self.calculator = TruthLensScoreCalculator(strict=self.config.strict_mode)
        self.explainer = ScoreExplainer(method=self.config.attribution.method)

        self.risk_config = RiskConfig()

        logger.info("[Pipeline] Initialized")

    # =========================================================
    # MAIN RUN
    # =========================================================
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
        # 2. NORMALIZATION
        # =========================
        normalized_profile = self._normalize_profile(profile)

        # =========================
        # 3. CALIBRATION (OPTIONAL)
        # =========================
        calibrated_profile = self._calibrate_profile(normalized_profile)

        # =========================
        # 4. AGGREGATION
        # =========================
        scores = self.calculator.compute_scores(
            calibrated_profile,
            confidence=confidence,
        )

        # =========================
        # 5. RISK
        # =========================
        risks = {}
        if self.config.enable_risk:
            risks = assess_truthlens_risks(
                scores,
                config=self.risk_config,
                calibrated=True,
            )

        # =========================
        # 6. EXPLANATION
        # =========================
        explanations = {}
        if self.config.enable_explanations:
            explanations = self.explainer.explain_profile(calibrated_profile)

        # =========================
        # 7. OUTPUT
        # =========================
        result = {
            "scores": scores,
            "raw_scores": scores,
            "risks": risks,
            "explanations": explanations,
            "analysis_modules": {},
        }

        validated = TruthLensAggregationOutputModel(**result)

        return validated.model_dump()

    # =========================================================
    # NORMALIZATION
    # =========================================================
    def _normalize_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}

        for section, features in profile.items():
            if not isinstance(features, dict):
                continue

            values = list(features.values())

            try:
                norm_vals = self.normalizer.fit_transform(values)
                normalized[section] = dict(zip(features.keys(), norm_vals))
            except Exception:
                normalized[section] = features

        return normalized

    # =========================================================
    # CALIBRATION
    # =========================================================
    def _calibrate_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        calibrated = {}

        for section, features in profile.items():
            calibrated_section = {}

            for k, v in features.items():
                try:
                    calibrated_val = self.calibrator.transform([v])[0]
                except Exception:
                    calibrated_val = v

                calibrated_section[k] = float(calibrated_val)

            calibrated[section] = calibrated_section

        return calibrated

    # =========================================================
    # BATCH
    # =========================================================
    def run_batch(self, batch_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.run(x) for x in batch_outputs]