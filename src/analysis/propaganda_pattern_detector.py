"""
File Name: propaganda_pattern_detector.py
Module: Narrative Analysis - Propaganda Pattern Detection
Description:
    High-level propaganda pattern detection module for the TruthLens AI system.
    This module aggregates signals from multiple analytical subsystems including
    emotion analysis, narrative analysis, rhetorical device detection, and
    argument mining to estimate the likelihood of common propaganda patterns.

    The detector focuses on macro-level patterns such as fear amplification,
    scapegoating narratives, and polarization framing. These scores provide
    interpretable indicators of manipulative discourse strategies used in
    propaganda and ideological messaging.

Dependencies:
    logging
    typing
    dataclasses
    numpy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PropagandaPatternConfig:
    """
    Configuration for PropagandaPatternDetector.
    """

    fear_weight_emotion: float = 0.4
    fear_weight_rhetoric: float = 0.4
    fear_weight_narrative: float = 0.2

    scapegoat_weight_rhetoric: float = 0.6
    scapegoat_weight_argument: float = 0.4

    polarization_weight_narrative: float = 0.6
    polarization_weight_rhetoric: float = 0.4


class PropagandaPatternDetector:
    """
    Detects high-level propaganda patterns by combining multiple analytical signals.
    """

    def __init__(self, config: PropagandaPatternConfig | None = None) -> None:
        """
        Initialize detector configuration.
        """

        self.config = config or PropagandaPatternConfig()

        logger.info("PropagandaPatternDetector initialized")

    def analyze(
        self,
        emotion_features: Dict[str, float] | None = None,
        narrative_features: Dict[str, float] | None = None,
        rhetorical_features: Dict[str, float] | None = None,
        argument_features: Dict[str, float] | None = None,
    ) -> Dict[str, float]:
        """
        Compute propaganda pattern scores using multiple subsystem outputs.

        Args:
            emotion_features: Emotion analysis output
            narrative_features: Narrative analysis output
            rhetorical_features: Rhetorical device output
            argument_features: Argument mining output

        Returns:
            Dictionary containing propaganda pattern scores.
        """

        emotion_features = emotion_features or {}
        narrative_features = narrative_features or {}
        rhetorical_features = rhetorical_features or {}
        argument_features = argument_features or {}

        fear_score = self._compute_fear_propaganda(
            emotion_features,
            narrative_features,
            rhetorical_features,
        )

        scapegoat_score = self._compute_scapegoating(
            rhetorical_features,
            argument_features,
        )

        polarization_score = self._compute_polarization(
            narrative_features,
            rhetorical_features,
        )

        return {
            "fear_propaganda_score": float(fear_score),
            "scapegoating_score": float(scapegoat_score),
            "polarization_score": float(polarization_score),
        }

    def _compute_fear_propaganda(
        self,
        emotion: Dict[str, float],
        narrative: Dict[str, float],
        rhetoric: Dict[str, float],
    ) -> float:
        """
        Estimate fear-driven propaganda intensity.
        """

        emotion_signal = emotion.get("emotion_fear", 0.0)

        rhetoric_signal = rhetoric.get("rhetoric_fear_appeal_score", 0.0)

        narrative_signal = narrative.get("narrative_conflict_term_ratio", 0.0)

        score = (
            emotion_signal * self.config.fear_weight_emotion
            + rhetoric_signal * self.config.fear_weight_rhetoric
            + narrative_signal * self.config.fear_weight_narrative
        )

        return float(score)

    def _compute_scapegoating(
        self,
        rhetoric: Dict[str, float],
        argument: Dict[str, float],
    ) -> float:
        """
        Estimate scapegoating narrative patterns.
        """

        rhetoric_signal = rhetoric.get("rhetoric_scapegoating_score", 0.0)

        argument_signal = argument.get("argument_contrast_ratio", 0.0)

        score = (
            rhetoric_signal * self.config.scapegoat_weight_rhetoric
            + argument_signal * self.config.scapegoat_weight_argument
        )

        return float(score)

    def _compute_polarization(
        self,
        narrative: Dict[str, float],
        rhetoric: Dict[str, float],
    ) -> float:
        """
        Estimate polarization framing intensity.
        """

        narrative_signal = narrative.get("narrative_polarization_ratio", 0.0)

        rhetoric_signal = rhetoric.get("rhetoric_loaded_language_score", 0.0)

        score = (
            narrative_signal * self.config.polarization_weight_narrative
            + rhetoric_signal * self.config.polarization_weight_rhetoric
        )

        return float(score)


def propaganda_pattern_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert propaganda pattern features into numeric vector.

    Args:
        features: Propaganda feature dictionary.

    Returns:
        NumPy vector representation.
    """

    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary")

    if not features:
        raise ValueError("features must be a non-empty dictionary")

    values = []

    for key, value in features.items():
        if isinstance(value, (int, float, np.number)):
            values.append(float(value))
        else:
            logger.warning("Non-numeric propaganda feature skipped: %s", key)

    if not values:
        raise ValueError("No numeric propaganda features found")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Propaganda pattern vector conversion failed")
        raise RuntimeError("Failed to convert propaganda features") from exc