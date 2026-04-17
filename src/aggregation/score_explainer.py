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
from typing import Dict, Any, List


logger = logging.getLogger(__name__)

EXPLAINABLE_SECTIONS = {
    "bias",
    "emotion",
    "narrative",
    "discourse",
    "graph",
    "ideology",
    "analysis",
}


class ScoreExplainer:
    def __init__(self) -> None:
        logger.info("ScoreExplainer initialized")

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        return top_k

    def rank_contributors(
        self,
        feature_section: Dict[str, float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        top_k = self._validate_top_k(top_k)

        if not isinstance(feature_section, dict) or not feature_section:
            return []

        numeric_items = [
            (k, float(v))
            for k, v in feature_section.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

        if not numeric_items:
            return []

        sorted_items = sorted(
            numeric_items,
            key=lambda x: (-abs(x[1]), x[0]),
        )[:top_k]

        return [
            {
                "feature": k,
                "value": v,
                "magnitude": abs(v),
                "direction": "positive" if v >= 0 else "negative",
            }
            for k, v in sorted_items
        ]

    def explain_section(
        self,
        section_name: str,
        feature_section: Dict[str, float],
        *,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        top_features = self.rank_contributors(feature_section, top_k)
        return {
            "section": section_name,
            "top_contributors": [item["feature"] for item in top_features],
            "contributors": top_features,
        }

    def explain_profile(
        self,
        profile: Dict[str, Dict[str, float]],
        *,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        top_k = self._validate_top_k(top_k)

        if not isinstance(profile, dict):
            raise ValueError("profile must be a dictionary")

        explanations: Dict[str, Any] = {}
        for section, features in profile.items():
            if section not in EXPLAINABLE_SECTIONS:
                continue
            if not isinstance(features, dict):
                continue
            explanations[section] = self.explain_section(section, features, top_k=top_k)

        return explanations

    def explain_final_score(
        self,
        scores: Dict[str, float],
        *,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        top_k = self._validate_top_k(top_k)

        if not isinstance(scores, dict) or not scores:
            raise ValueError("scores must be a non-empty dictionary")

        items = [
            (k, float(v))
            for k, v in scores.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if not items:
            raise ValueError("scores must contain at least one numeric non-boolean value")

        top_items = sorted(items, key=lambda x: (-abs(x[1]), x[0]))[:top_k]

        return {
            "top_contributors": [k for k, _ in top_items],
            "contributors": [
                {
                    "feature": k,
                    "value": v,
                    "magnitude": abs(v),
                    "direction": "positive" if v >= 0 else "negative",
                }
                for k, v in top_items
            ],
        }


def summarize_score_explanation(
    scores: Dict[str, float],
    *,
    top_k: int = 3,
) -> Dict[str, Any]:
    return ScoreExplainer().explain_final_score(scores, top_k=top_k)
