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
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BiasProfileConfig:
    """
    Configuration for bias profile aggregation.
    """

    bias_weight: float = 1.0
    emotion_weight: float = 1.0
    narrative_weight: float = 1.0
    discourse_weight: float = 1.0
    ideology_weight: float = 0.5
    normalize_values: bool = True


class BiasProfileBuilder:
    """
    Aggregates multiple analysis outputs into a unified bias profile.

    This class collects signals produced by various analytical modules
    within TruthLens AI and produces a structured representation of bias
    characteristics suitable for downstream analytics, reporting pipelines,
    or model feature inputs.
    """

    def __init__(self, config: BiasProfileConfig | None = None) -> None:
        """
        Initialize the BiasProfileBuilder.

        Args:
            config: Optional BiasProfileConfig controlling aggregation behavior.
        """
        self.config = config or BiasProfileConfig()
        logger.info("BiasProfileBuilder initialized with config: %s", self.config)

    def build_profile(
        self,
        bias_features: Dict[str, float],
        emotion_features: Dict[str, float],
        narrative_features: Dict[str, float],
        discourse_features: Dict[str, float],
        ideology_predictions: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Construct a unified bias profile from multiple analytical components.

        Args:
            bias_features: Bias-related signals.
            emotion_features: Emotion analysis outputs.
            narrative_features: Narrative pattern indicators.
            discourse_features: Discourse structure metrics.
            ideology_predictions: Ideology classification probabilities.

        Returns:
            Structured bias profile dictionary.
        """

        self._validate_input(bias_features, "bias_features")
        self._validate_input(emotion_features, "emotion_features")
        self._validate_input(narrative_features, "narrative_features")
        self._validate_input(discourse_features, "discourse_features")
        self._validate_input(ideology_predictions, "ideology_predictions")

        profile: Dict[str, Any] = {
            "bias": self._aggregate_bias(bias_features),
            "emotion": self._sanitize_numeric_dict(emotion_features),
            "narrative": self._sanitize_numeric_dict(narrative_features),
            "discourse": self._sanitize_numeric_dict(discourse_features),
            "ideology": self._sanitize_numeric_dict(ideology_predictions),
        }

        profile["bias_score"] = self._compute_bias_score(profile)

        logger.debug("Bias profile built successfully")
        return profile

    def _validate_input(self, data: Any, name: str) -> None:
        """
        Validate dictionary input.

        Args:
            data: Input object to validate.
            name: Parameter name.

        Raises:
            ValueError: If input is invalid.
        """
        if not isinstance(data, dict):
            logger.error("Invalid input for %s: expected dict", name)
            raise ValueError(f"{name} must be a dictionary")

    def _sanitize_numeric_dict(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert values to numeric floats where possible.

        Non-numeric values are replaced with 0.0.

        Args:
            data: Input dictionary.

        Returns:
            Sanitized dictionary with float values.
        """
        sanitized: Dict[str, float] = {}

        for key, value in data.items():
            if isinstance(value, (int, float, np.number)):
                sanitized[key] = float(value)
            else:
                logger.debug("Non-numeric value detected for key '%s'", key)
                sanitized[key] = 0.0

        if self.config.normalize_values and sanitized:
            sanitized = self._normalize_values(sanitized)

        return sanitized

    def _aggregate_bias(self, bias_features: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate bias feature signals.

        Args:
            bias_features: Raw bias feature signals.

        Returns:
            Normalized bias feature dictionary.
        """
        return self._sanitize_numeric_dict(bias_features)

    def _normalize_values(self, data: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize values to range [0,1].

        Args:
            data: Numeric dictionary.

        Returns:
            Normalized dictionary.
        """
        values = np.array(list(data.values()), dtype=np.float32)

        if values.size == 0:
            return data

        min_val = float(np.min(values))
        max_val = float(np.max(values))

        if max_val - min_val == 0:
            return data

        normalized = (values - min_val) / (max_val - min_val)

        return {k: float(v) for k, v in zip(data.keys(), normalized)}

    def _compute_bias_score(self, profile: Dict[str, Any]) -> float:
        """
        Compute a global bias score from profile signals.

        Args:
            profile: Structured bias profile.

        Returns:
            Global bias score.
        """

        try:
            bias_vals = list(profile["bias"].values())
            emotion_vals = list(profile["emotion"].values())
            narrative_vals = list(profile["narrative"].values())
            discourse_vals = list(profile["discourse"].values())
            ideology_vals = list(profile["ideology"].values())

            weighted_values: List[float] = []

            weighted_values.extend(
                [v * self.config.bias_weight for v in bias_vals]
            )
            weighted_values.extend(
                [v * self.config.emotion_weight for v in emotion_vals]
            )
            weighted_values.extend(
                [v * self.config.narrative_weight for v in narrative_vals]
            )
            weighted_values.extend(
                [v * self.config.discourse_weight for v in discourse_vals]
            )
            weighted_values.extend(
                [v * self.config.ideology_weight for v in ideology_vals]
            )

            if not weighted_values:
                logger.warning("No values available for bias score computation")
                return 0.0

            score = float(np.mean(np.array(weighted_values, dtype=np.float32)))

            return score

        except Exception as exc:
            logger.exception("Bias score computation failed")
            raise RuntimeError("Failed to compute bias score") from exc


def bias_profile_vector(profile: Dict[str, Any]) -> np.ndarray:
    """
    Convert bias profile into a numerical feature vector.

    Args:
        profile: Structured bias profile.

    Returns:
        NumPy vector representation.
    """

    if not isinstance(profile, dict):
        raise ValueError("profile must be a dictionary")

    values: List[float] = []

    for section in ("bias", "emotion", "narrative", "discourse", "ideology"):
        section_data = profile.get(section)

        if isinstance(section_data, dict):
            for value in section_data.values():
                if isinstance(value, (int, float, np.number)):
                    values.append(float(value))

    if not values:
        logger.error("Bias profile contains no numeric values")
        raise ValueError("profile contains no numeric values")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector

    except Exception as exc:
        logger.exception("Bias profile vector conversion failed")
        raise RuntimeError("Failed to convert bias profile") from exc