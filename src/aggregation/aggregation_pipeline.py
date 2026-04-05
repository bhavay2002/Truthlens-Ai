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

import logging
from typing import Dict, Any

from src.aggregation.score_normalizer import normalize_minmax
from src.aggregation.weight_manager import WeightManager
from src.aggregation.risk_assessment import assess_truthlens_risks
from src.aggregation.score_explainer import ScoreExplainer
from src.aggregation.truthlens_score_calculator import (
    TruthLensScoreCalculator,
)
from src.analysis.integration_runner import AnalysisIntegrationRunner


logger = logging.getLogger(__name__)


class AggregationPipeline:
    """
    End-to-end aggregation pipeline for TruthLens scoring.
    """

    def __init__(self) -> None:
        self.weight_manager = WeightManager()
        self.score_calculator = TruthLensScoreCalculator()
        self.explainer = ScoreExplainer()
        self.analysis_runner = AnalysisIntegrationRunner()

        logger.info("AggregationPipeline initialized")

    def _flatten_analysis_outputs(
        self,
        analysis_modules: Dict[str, Any],
    ) -> Dict[str, float]:
        flattened: Dict[str, float] = {}

        for module_name, module_output in analysis_modules.items():
            if not isinstance(module_output, dict):
                continue

            for key, value in module_output.items():
                feature_key = f"analysis_{module_name}_{key}"
                if isinstance(value, (int, float)):
                    flattened[feature_key] = float(value)
                elif isinstance(value, (list, tuple, set)):
                    flattened[f"{feature_key}_count"] = float(len(value))
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float)):
                            flattened[
                                f"{feature_key}_{sub_key}"
                            ] = float(sub_value)

        return flattened

    def _merge_section(
        self,
        profile: Dict[str, Any],
        section: str,
        prefix: str,
        module_data: Dict[str, Any],
    ) -> None:
        target = profile.get(section)
        if not isinstance(target, dict):
            target = {}

        for key, value in module_data.items():
            if isinstance(value, (int, float)):
                target[f"{prefix}{key}"] = float(value)
            elif isinstance(value, (list, tuple, set)):
                target[f"{prefix}{key}_count"] = float(len(value))

        profile[section] = target

    def _inject_analysis_sections(
        self,
        profile: Dict[str, Any],
        analysis_modules: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched = dict(profile)
        enriched["analysis_modules"] = analysis_modules

        analysis_section = self._flatten_analysis_outputs(analysis_modules)
        if analysis_section:
            existing_analysis = enriched.get("analysis")
            if isinstance(existing_analysis, dict):
                merged_analysis = dict(existing_analysis)
                merged_analysis.update(analysis_section)
                enriched["analysis"] = merged_analysis
            else:
                enriched["analysis"] = analysis_section

        self._merge_section(
            enriched,
            "bias",
            "analysis_framing_",
            analysis_modules.get("framing", {}),
        )
        self._merge_section(
            enriched,
            "bias",
            "analysis_ideological_",
            analysis_modules.get("ideological_language", {}),
        )
        self._merge_section(
            enriched,
            "bias",
            "analysis_context_",
            analysis_modules.get("context_omission", {}),
        )

        self._merge_section(
            enriched,
            "emotion",
            "analysis_emotion_target_",
            analysis_modules.get("emotion_target", {}),
        )

        self._merge_section(
            enriched,
            "narrative",
            "analysis_narrative_conflict_",
            analysis_modules.get("narrative_conflict", {}),
        )
        self._merge_section(
            enriched,
            "narrative",
            "analysis_narrative_propagation_",
            analysis_modules.get("narrative_propagation", {}),
        )
        self._merge_section(
            enriched,
            "narrative",
            "analysis_narrative_temporal_",
            analysis_modules.get("narrative_temporal", {}),
        )
        self._merge_section(
            enriched,
            "narrative",
            "analysis_propaganda_pattern_",
            analysis_modules.get("propaganda_pattern", {}),
        )

        self._merge_section(
            enriched,
            "discourse",
            "analysis_argument_",
            analysis_modules.get("argument_mining", {}),
        )
        self._merge_section(
            enriched,
            "discourse",
            "analysis_discourse_",
            analysis_modules.get("discourse_coherence", {}),
        )
        self._merge_section(
            enriched,
            "discourse",
            "analysis_rhetorical_",
            analysis_modules.get("rhetorical_device", {}),
        )
        self._merge_section(
            enriched,
            "discourse",
            "analysis_source_",
            analysis_modules.get("source_attribution", {}),
        )

        return enriched

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
                if isinstance(v, (int, float))
            ]

            if not numeric_keys:
                normalized_profile[section] = features
                continue

            values = [features[k] for k in numeric_keys]

            norm_values = normalize_minmax(values)

            normalized_section = features.copy()

            for k, v in zip(numeric_keys, norm_values):
                normalized_section[k] = float(v)

            normalized_profile[section] = normalized_section

        return normalized_profile

    def apply_weights(
        self,
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply dynamic weighting to selected subsystem scores.
        """

        weights = self.weight_manager.get_weights()

        weighted_scores: Dict[str, float] = {}

        for key, value in scores.items():

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

        enriched_profile = self._inject_analysis_sections(profile, resolved_analysis)
        normalized_profile = self.normalize_profile(enriched_profile)

        scores = self.score_calculator.compute_scores(
            normalized_profile
        )

        weighted_scores = self.apply_weights(scores)

        risks = assess_truthlens_risks(scores)

        explanations = self.explainer.explain_profile(
            normalized_profile
        )

        result = {
            "scores": weighted_scores,
            "raw_scores": scores,
            "risks": risks,
            "explanations": explanations,
            "analysis_modules": resolved_analysis,
        }

        logger.info("Aggregation pipeline completed")

        return result
