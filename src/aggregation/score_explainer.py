"""
File Name: score_explainer.py
Module: TruthLens AI - Aggregation Score Explainer
Description:
    Provides explainability utilities for TruthLens scoring outputs.
    The module analyzes feature-level signals used in aggregation and
    produces interpretable explanations describing why a particular
    score was generated.

    It supports:
        • Feature contribution ranking
        • Score breakdown generation
        • Human-readable explanation structures

Dependencies:
    logging
    typing
    numpy

Inputs:
    aggregated feature signals and score dictionary

Outputs:
    explanation dictionaries describing score composition
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class ScoreExplainer:
    """
    Generates explanations for TruthLens scores by identifying
    dominant contributing signals.
    """

    def __init__(self) -> None:
        logger.info("ScoreExplainer initialized")

    def rank_contributors(
        self,
        feature_section: Dict[str, float],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Rank features by magnitude contribution.
        """

        if not isinstance(feature_section, dict) or not feature_section:
            return []

        numeric_items = [
            (k, float(v))
            for k, v in feature_section.items()
            if isinstance(v, (int, float))
        ]

        if not numeric_items:
            return []

        sorted_items = sorted(
            numeric_items,
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return sorted_items[:top_k]

    def explain_section(
        self,
        section_name: str,
        feature_section: Dict[str, float],
        *,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Produce explanation for a specific analysis section.
        """

        top_features = self.rank_contributors(feature_section, top_k)

        return {
            "section": section_name,
            "top_contributors": [
                feature for feature, _ in top_features
            ],
            "feature_scores": dict(top_features),
        }

    def explain_profile(
        self,
        profile: Dict[str, Dict[str, float]],
        *,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate explanation across all analysis sections.
        """

        if not isinstance(profile, dict):
            raise ValueError("profile must be a dictionary")

        explanations: Dict[str, Any] = {}

        for section, features in profile.items():

            if not isinstance(features, dict):
                continue

            explanations[section] = self.explain_section(
                section,
                features,
                top_k=top_k,
            )

        return explanations

    def explain_final_score(
        self,
        scores: Dict[str, float],
        *,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Identify the dominant contributors to the final TruthLens score.
        """

        if not isinstance(scores, dict) or not scores:
            raise ValueError("scores must be a non-empty dictionary")

        items = [
            (k, float(v))
            for k, v in scores.items()
            if isinstance(v, (int, float))
        ]

        sorted_items = sorted(
            items,
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_items = sorted_items[:top_k]

        explanation = {
            "top_contributors": [k for k, _ in top_items],
            "contribution_values": dict(top_items),
        }

        return explanation


def summarize_score_explanation(
    scores: Dict[str, float],
    *,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Utility wrapper for quick explanation generation.
    """

    explainer = ScoreExplainer()

    return explainer.explain_final_score(
        scores,
        top_k=top_k,
    )