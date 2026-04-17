"""
File Name: aggregation_pipeline.py
Module: TruthLens AI - Aggregation Pipeline
Description:
    End-to-end orchestration pipeline for the TruthLens aggregation layer.

    This module coordinates the complete scoring workflow by integrating
    normalization, weight management, score computation, risk assessment,
    and explanation generation.

    Aggregation Flow
    ----------------
        normalize signals
        ↓
        apply weights
        ↓
        compute TruthLens scores
        ↓
        compute categorical risk levels
        ↓
        generate explanations

Dependencies:
    logging
    typing
    numpy
    src.aggregation.score_normalizer
    src.aggregation.weight_manager
    src.aggregation.risk_assessment
    src.aggregation.score_explainer
    src.aggregation.truthlens_score_calculator

Inputs:
    profile: aggregated subsystem outputs

Outputs:
    structured aggregation result containing scores, risks, explanations
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Any

from src.aggregation.score_normalizer import normalize_minmax
from src.aggregation.weight_manager import WeightManager
from src.aggregation.risk_assessment import assess_truthlens_risks
from src.aggregation.score_explainer import ScoreExplainer
from src.aggregation.truthlens_score_calculator import (
    TruthLensScoreCalculator,
)
from src.aggregation.score_schema import TruthLensAggregationOutputModel
from src.analysis.integration_runner import AnalysisIntegrationRunner


logger = logging.getLogger(__name__)


class AggregationPipeline:
    """
    End-to-end aggregation pipeline for TruthLens scoring.
    """

    ANALYSIS_MERGE_MAP = [
        ("bias", "analysis_framing_", "framing"),
        ("bias", "analysis_ideological_", "ideological_language"),
        ("bias", "analysis_context_", "context_omission"),
        ("emotion", "analysis_emotion_target_", "emotion_target"),
        ("narrative", "analysis_narrative_conflict_", "narrative_conflict"),
        ("narrative", "analysis_narrative_propagation_", "narrative_propagation"),
        ("narrative", "analysis_narrative_temporal_", "narrative_temporal"),
        ("narrative", "analysis_propaganda_pattern_", "propaganda_pattern"),
        ("discourse", "analysis_argument_", "argument_mining"),
        ("discourse", "analysis_discourse_", "discourse_coherence"),
        ("discourse", "analysis_rhetorical_", "rhetorical_device"),
        ("discourse", "analysis_source_", "source_attribution"),
    ]

    def __init__(self) -> None:
        self.weight_manager = WeightManager()
        self.score_calculator = TruthLensScoreCalculator()
        self.explainer = ScoreExplainer()
        self.analysis_runner = AnalysisIntegrationRunner()

        logger.info("AggregationPipeline initialized")

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        """
        Numeric check that excludes bool (bool is a subclass of int in Python).
        """
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _inject_analysis_sections(
        self,
        profile: Dict[str, Any],
        analysis_modules: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched = copy.deepcopy(profile)

        for section, prefix, module_key in self.ANALYSIS_MERGE_MAP:
            module_data = analysis_modules.get(module_key)
            if not isinstance(module_data, dict):
                continue

            target = enriched.setdefault(section, {})
            for key, value in module_data.items():
                if self._is_numeric(value):
                    target[f"{prefix}{key}"] = float(value)
                elif isinstance(value, (list, tuple, set)):
                    target[f"{prefix}{key}_count"] = float(len(value))

        return enriched

    def _sanitize_analysis_modules(
        self,
        analysis_modules: Dict[str, Any],
    ) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}

        for module, output in analysis_modules.items():
            if not isinstance(output, dict):
                continue

            safe_output: Dict[str, Any] = {}

            for k, v in output.items():
                if isinstance(v, (int, float, str, bool)):
                    safe_output[k] = v
                elif isinstance(v, (list, tuple, set)):
                    safe_output[k] = len(v)
                elif isinstance(v, dict):
                    safe_output[k] = {
                        sub_k: sub_v
                        for sub_k, sub_v in v.items()
                        if isinstance(sub_v, (int, float, str, bool))
                    }

            sanitized[module] = safe_output

        return sanitized

    def normalize_profile(
        self,
        profile: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Normalize numeric values in each feature section.
        """

        normalized_profile: Dict[str, Dict[str, float]] = {}

        for section, features in profile.items():

            if not isinstance(features, dict):
                continue

            numeric_keys = [
                k for k, v in features.items()
                if self._is_numeric(v)
            ]

            if not numeric_keys:
                normalized_profile[section] = features.copy()
                continue

            values = [features[k] for k in numeric_keys]

            norm_values = normalize_minmax(values)

            normalized_section = dict(features)

            for k, v in zip(numeric_keys, norm_values):
                normalized_section[k] = float(v)

            normalized_profile[section] = normalized_section

        return normalized_profile

    def build_profile_from_prediction(
        self,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(prediction, dict):
            raise TypeError("prediction must be a dictionary")

        profile: Dict[str, Any] = {
            "bias": {
                "bias_prediction": 1.0 if prediction.get("bias") == "bias" else 0.0,
            },
            "ideology": {
                "ideology_left": 1.0 if prediction.get("ideology") == "left" else 0.0,
                "ideology_center": 1.0 if prediction.get("ideology") == "center" else 0.0,
                "ideology_right": 1.0 if prediction.get("ideology") == "right" else 0.0,
            },
            "propaganda": {
                "propaganda_probability": float(
                    prediction.get("propaganda_probability") or 0.0
                ),
            },
            "credibility": {
                "credibility_score": float(
                    prediction.get("credibility_score") or 0.0
                ),
            },
        }

        emotion = prediction.get("emotion")
        if isinstance(emotion, dict):
            profile["emotion"] = {
                k: float(v)
                for k, v in emotion.items()
                if isinstance(v, (int, float))
            }

        credibility_explanation = prediction.get("credibility_explanation")
        if isinstance(credibility_explanation, dict):
            for comp_key, comp_val in credibility_explanation.items():
                if isinstance(comp_val, (int, float)):
                    profile["credibility"][comp_key] = float(comp_val)

        return profile

    def apply_weights(
        self,
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply dynamic weighting to selected subsystem scores.
        """
        # NOTE:
        # This helper remains for backward compatibility with any direct callers.
        # The main pipeline now computes raw and weighted scores explicitly by
        # calling score_calculator with and without weights.
        weights = self.weight_manager.get_weights()

        weighted_scores: Dict[str, float] = {}

        # Weighting is applied only to numeric score entries.
        for key, value in scores.items():
            if not self._is_numeric(value):
                continue
            base_key = key.replace("truthlens_", "").replace("_score", "")

            weight = weights.get(base_key, 1.0)

            weighted_scores[key] = float(value * weight)

        return weighted_scores

    def run(
        self,
        profile: Dict[str, Any],
        *,
        text: str | None = None,
        analysis_modules: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Execute full aggregation pipeline.
        """

        if not isinstance(profile, dict):
            raise ValueError("profile must be a dictionary")

        logger.info("Running TruthLens aggregation pipeline")

        resolved_analysis: Dict[str, Any] = {}
        if isinstance(analysis_modules, dict):
            resolved_analysis = analysis_modules
        elif isinstance(profile.get("analysis_modules"), dict):
            resolved_analysis = profile.get("analysis_modules", {})
        elif isinstance(text, str) and text.strip():
            resolved_analysis = self.analysis_runner.analyze_text(text)

        sanitized_analysis = self._sanitize_analysis_modules(resolved_analysis)
        enriched_profile = self._inject_analysis_sections(profile, sanitized_analysis)
        normalized_profile = self.normalize_profile(enriched_profile)

        # Compute true raw scores (default formula weights)
        raw_scores = self.score_calculator.compute_scores(
            normalized_profile,
            weights=None,
        )

        # Compute configured weighted scores (WeightManager supplied)
        weights = self.weight_manager.get_weights()
        scores = self.score_calculator.compute_scores(
            normalized_profile,
            weights=weights,
        )

        risks = assess_truthlens_risks(scores)

        explanations = self.explainer.explain_profile(
            normalized_profile
        )

        result = {
            "scores": scores,
            "raw_scores": raw_scores,
            "risks": risks,
            "explanations": explanations,
            "analysis_modules": sanitized_analysis,
        }

        validated = TruthLensAggregationOutputModel(**result)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Aggregation pipeline completed")

        return validated.model_dump()

    def run_batch(self, profiles: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        return [self.run(profile) for profile in profiles]
