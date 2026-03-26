"""
File Name: emotion_score.py
Module: Emotion Analysis - Composite Scoring
Description:
    Computes composite emotion scores used in the TruthLens AI system. The module
    aggregates emotion signals from multiple analysis components such as emotion
    distributions, intensity estimations, and polarization signals. The resulting
    scores provide a normalized emotional strength measure suitable for ranking,
    comparison, and downstream model features.

Dependencies:
    logging
    typing
    numpy

Inputs:
    Emotion feature dictionary

Outputs:
    Emotion scoring metrics and numerical score vector
"""

import logging
from typing import Dict

import numpy as np


logger = logging.getLogger(__name__)


class EmotionScoreCalculator:
    """
    Computes aggregated emotional scoring metrics from emotion features.
    """

    def __init__(self) -> None:
        """Initialize emotion score calculator."""

        logger.info("EmotionScoreCalculator initialized")

    def compute_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """Compute composite emotional metrics from feature inputs."""

        if not isinstance(features, dict) or not features:
            raise ValueError("features must be a non-empty dictionary")

        positive = self._safe_value(features.get("emotion_positive_score", 0.0))
        negative = self._safe_value(features.get("emotion_negative_score", 0.0))
        intensity = self._safe_value(features.get("emotion_intensity", 0.0))

        emotional_load = positive + negative

        if emotional_load == 0:
            polarity = 0.0
        else:
            polarity = (positive - negative) / emotional_load

        emotional_strength = emotional_load * intensity

        volatility = self._estimate_volatility(features)

        scores = {
            "emotion_positive_score": float(positive),
            "emotion_negative_score": float(negative),
            "emotion_intensity_score": float(intensity),
            "emotion_emotional_load": float(emotional_load),
            "emotion_polarity_score": float(polarity),
            "emotion_emotional_strength": float(emotional_strength),
            "emotion_volatility_score": float(volatility),
        }

        return scores

    def _estimate_volatility(self, features: Dict[str, float]) -> float:
        """Estimate emotional volatility using emotion distribution variance."""

        emotion_values = []

        for key, value in features.items():
            if key.startswith("emotion_") and isinstance(value, (int, float)):
                emotion_values.append(float(value))

        if not emotion_values:
            return 0.0

        try:
            return float(np.std(np.array(emotion_values, dtype=np.float32)))
        except Exception as exc:
            logger.exception("Volatility estimation failed")
            raise RuntimeError("Emotion volatility calculation failed") from exc

    def _safe_value(self, value: float) -> float:
        """Validate and normalize numeric input values."""

        if not isinstance(value, (int, float)):
            return 0.0

        return float(value)


def emotion_score_vector(scores: Dict[str, float]) -> np.ndarray:
    """Convert emotion score dictionary into numeric vector."""

    if not isinstance(scores, dict) or not scores:
        raise ValueError("scores must be a non-empty dictionary")

    try:
        vector = np.array(list(scores.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Emotion score vector conversion failed")
        raise RuntimeError("Failed to convert emotion scores") from exc