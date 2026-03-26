"""
File Name: bias_profile_builder.py
Module: Bias Analysis - Profile Construction
Description:
    Builds a structured bias profile from multiple analytical components used
    in the TruthLens AI system. The module aggregates bias features, ideology
    predictions, emotion signals, discourse indicators, and narrative features
    into a unified bias profile representation that can be used for downstream
    analysis, reporting, or model input.

Dependencies:
    logging
    typing
    numpy

Inputs:
    Dictionaries containing outputs from various analytical modules

Outputs:
    Structured bias profile dictionary and optional numerical feature vector
"""

import logging
from typing import Dict, Any

import numpy as np


logger = logging.getLogger(__name__)


class BiasProfileBuilder:
    """
    Aggregates multiple analysis outputs into a unified bias profile.
    """

    def __init__(self) -> None:
        """Initialize bias profile builder."""
        logger.info("BiasProfileBuilder initialized")

    def build_profile(
        self,
        bias_features: Dict[str, float],
        emotion_features: Dict[str, float],
        narrative_features: Dict[str, float],
        discourse_features: Dict[str, float],
        ideology_predictions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Construct a unified bias profile from multiple analytical components."""

        if not isinstance(bias_features, dict):
            raise ValueError("bias_features must be a dictionary")

        if not isinstance(emotion_features, dict):
            raise ValueError("emotion_features must be a dictionary")

        if not isinstance(narrative_features, dict):
            raise ValueError("narrative_features must be a dictionary")

        if not isinstance(discourse_features, dict):
            raise ValueError("discourse_features must be a dictionary")

        if not isinstance(ideology_predictions, dict):
            raise ValueError("ideology_predictions must be a dictionary")

        profile: Dict[str, Any] = {}

        profile["bias"] = self._aggregate_bias(bias_features)

        profile["emotion"] = emotion_features

        profile["narrative"] = narrative_features

        profile["discourse"] = discourse_features

        profile["ideology"] = ideology_predictions

        profile["bias_score"] = self._compute_bias_score(profile)

        return profile

    def _aggregate_bias(self, bias_features: Dict[str, float]) -> Dict[str, float]:
        """Aggregate bias feature signals."""

        normalized_features: Dict[str, float] = {}

        for key, value in bias_features.items():
            if isinstance(value, (int, float)):
                normalized_features[key] = float(value)
            else:
                normalized_features[key] = 0.0

        return normalized_features

    def _compute_bias_score(self, profile: Dict[str, Any]) -> float:
        """Compute a global bias score from profile signals."""

        bias_values = list(profile["bias"].values())

        emotion_values = [
            v for k, v in profile["emotion"].items() if isinstance(v, (int, float))
        ]

        narrative_values = [
            v for k, v in profile["narrative"].items() if isinstance(v, (int, float))
        ]

        discourse_values = [
            v for k, v in profile["discourse"].items() if isinstance(v, (int, float))
        ]

        combined = bias_values + emotion_values + narrative_values + discourse_values

        if not combined:
            return 0.0

        try:
            score = float(np.mean(np.array(combined, dtype=np.float32)))
            return score
        except Exception as exc:
            logger.exception("Bias score computation failed")
            raise RuntimeError("Failed to compute bias score") from exc


def bias_profile_vector(profile: Dict[str, Any]) -> np.ndarray:
    """Convert bias profile into a numerical vector."""

    if not isinstance(profile, dict):
        raise ValueError("profile must be a dictionary")

    values = []

    for section in ["bias", "emotion", "narrative", "discourse"]:
        if section in profile and isinstance(profile[section], dict):
            values.extend(profile[section].values())

    if not values:
        raise ValueError("profile contains no numeric values")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Bias profile vector conversion failed")
        raise RuntimeError("Failed to convert bias profile") from exc