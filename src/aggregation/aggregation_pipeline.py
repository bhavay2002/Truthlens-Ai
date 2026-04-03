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


logger = logging.getLogger(__name__)


class AggregationPipeline:
    """
    End-to-end aggregation pipeline for TruthLens scoring.
    """

    def __init__(self) -> None:
        self.weight_manager = WeightManager()
        self.score_calculator = TruthLensScoreCalculator()
        self.explainer = ScoreExplainer()

        logger.info("AggregationPipeline initialized")

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
    ) -> Dict[str, Any]:
        """
        Execute full aggregation pipeline.
        """

        if not isinstance(profile, dict):
            raise ValueError("profile must be a dictionary")

        logger.info("Running TruthLens aggregation pipeline")

        normalized_profile = self.normalize_profile(profile)

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
        }

        logger.info("Aggregation pipeline completed")

        return result
