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

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(slots=True)
class BiasProfileConfig:

    bias_weight: float = 1.0
    emotion_weight: float = 1.0
    narrative_weight: float = 1.0
    discourse_weight: float = 1.0
    ideology_weight: float = 0.6

    normalize_values: bool = True
    normalization_method: str = "minmax"   # minmax | zscore


# ---------------------------------------------------------
# Bias Profile Builder
# ---------------------------------------------------------

class BiasProfileBuilder:

    """
    Aggregates signals from multiple analysis modules
    into a structured bias profile used throughout TruthLens.

    The profile supports:
    - interpretable analytics
    - ML feature vectors
    - explainability layers
    """

    PROFILE_SECTIONS = (
        "bias",
        "emotion",
        "narrative",
        "discourse",
        "ideology",
    )

    def __init__(self, config: BiasProfileConfig | None = None) -> None:

        self.config = config or BiasProfileConfig()

        logger.info("BiasProfileBuilder initialized")

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def build_profile(
        self,
        bias_features: Dict[str, float],
        emotion_features: Dict[str, float],
        narrative_features: Dict[str, float],
        discourse_features: Dict[str, float],
        ideology_predictions: Dict[str, float],
    ) -> Dict[str, Any]:

        bias = self._sanitize_numeric_dict(bias_features)
        emotion = self._sanitize_numeric_dict(emotion_features)
        narrative = self._sanitize_numeric_dict(narrative_features)
        discourse = self._sanitize_numeric_dict(discourse_features)
        ideology = self._sanitize_numeric_dict(ideology_predictions)

        profile = {

            "metadata": {
                "created_at": int(time.time()),
                "sections": list(self.PROFILE_SECTIONS),
            },

            "bias": bias,
            "emotion": emotion,
            "narrative": narrative,
            "discourse": discourse,
            "ideology": ideology,
        }

        profile["metrics"] = self._compute_profile_metrics(profile)

        profile["bias_score"] = self._compute_bias_score(profile)

        return profile

    # -----------------------------------------------------
    # Input sanitation
    # -----------------------------------------------------

    def _sanitize_numeric_dict(self, data: Dict[str, Any]) -> Dict[str, float]:

        if not isinstance(data, dict):
            raise ValueError("Input must be dictionary")

        sanitized = {}

        for k, v in data.items():

            if isinstance(v, (int, float, np.number)):
                sanitized[k] = float(v)
            else:
                sanitized[k] = 0.0

        if self.config.normalize_values and sanitized:

            sanitized = self._normalize_values(sanitized)

        return sanitized

    # -----------------------------------------------------
    # Normalization
    # -----------------------------------------------------

    def _normalize_values(self, data: Dict[str, float]) -> Dict[str, float]:

        values = np.array(list(data.values()), dtype=np.float32)

        if values.size == 0:
            return data

        if self.config.normalization_method == "zscore":

            mean = values.mean()
            std = values.std()

            if std < 1e-9:
                return data

            normalized = (values - mean) / std

        else:

            min_v = values.min()
            max_v = values.max()

            if max_v - min_v < 1e-9:
                return data

            normalized = (values - min_v) / (max_v - min_v)

        return {k: float(v) for k, v in zip(data.keys(), normalized)}

    # -----------------------------------------------------
    # Profile Metrics
    # -----------------------------------------------------

    def _compute_profile_metrics(self, profile: Dict[str, Any]) -> Dict[str, float]:

        ideology_values = list(profile["ideology"].values())

        if ideology_values:

            arr = np.array(ideology_values)

            entropy = -np.sum(arr * np.log(arr + 1e-9))
            dominance = float(arr.max())

        else:

            entropy = 0.0
            dominance = 0.0

        bias_values = list(profile["bias"].values())

        variance = float(np.var(bias_values)) if bias_values else 0.0

        return {

            "bias_variance": variance,
            "ideology_entropy": float(entropy),
            "ideology_dominance": dominance,
        }

    # -----------------------------------------------------
    # Bias Score
    # -----------------------------------------------------

    def _compute_bias_score(self, profile: Dict[str, Any]) -> float:

        values: List[float] = []

        values.extend(
            v * self.config.bias_weight
            for v in profile["bias"].values()
        )

        values.extend(
            v * self.config.emotion_weight
            for v in profile["emotion"].values()
        )

        values.extend(
            v * self.config.narrative_weight
            for v in profile["narrative"].values()
        )

        values.extend(
            v * self.config.discourse_weight
            for v in profile["discourse"].values()
        )

        values.extend(
            v * self.config.ideology_weight
            for v in profile["ideology"].values()
        )

        if not values:
            return 0.0

        return float(np.mean(values))


# ---------------------------------------------------------
# Vector Conversion
# ---------------------------------------------------------

def bias_profile_vector(profile: Dict[str, Any]) -> np.ndarray:

    if not isinstance(profile, dict):
        raise ValueError("profile must be dictionary")

    ordered_sections = (
        "bias",
        "emotion",
        "narrative",
        "discourse",
        "ideology",
    )

    values: List[float] = []

    for section in ordered_sections:

        data = profile.get(section, {})

        if isinstance(data, dict):

            for key in sorted(data.keys()):
                values.append(float(data[key]))

    if not values:
        raise ValueError("profile contains no numeric values")

    return np.array(values, dtype=np.float32)