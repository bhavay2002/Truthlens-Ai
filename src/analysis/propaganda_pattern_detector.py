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

from src.analysis.feature_schema import PROPAGANDA_PATTERN_KEYS, make_vector


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class PropagandaPatternConfig:

    # Fear propaganda
    fear_weight_emotion: float = 0.35
    fear_weight_rhetoric: float = 0.35
    fear_weight_narrative: float = 0.30

    # Scapegoating
    scapegoat_weight_rhetoric: float = 0.55
    scapegoat_weight_argument: float = 0.45

    # Polarization
    polarization_weight_narrative: float = 0.60
    polarization_weight_rhetoric: float = 0.40

    # Emotional amplification
    emotion_amplification_weight: float = 0.6
    rhetoric_amplification_weight: float = 0.4

    # Narrative imbalance
    narrative_claim_weight: float = 0.5
    narrative_evidence_weight: float = 0.5


# ------------------------------------------------------------
# Detector
# ------------------------------------------------------------

class PropagandaPatternDetector:

    """
    Detect macro-level propaganda patterns by aggregating signals
    from multiple analytical modules.
    """

    def __init__(self, config: PropagandaPatternConfig | None = None):

        self.config = config or PropagandaPatternConfig()

        logger.info("PropagandaPatternDetector initialized")

    # ------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------

    def analyze(
        self,
        emotion_features: Dict[str, float] | None = None,
        narrative_features: Dict[str, float] | None = None,
        rhetorical_features: Dict[str, float] | None = None,
        argument_features: Dict[str, float] | None = None,
        information_features: Dict[str, float] | None = None,
    ) -> Dict[str, float]:

        emotion = emotion_features or {}
        narrative = narrative_features or {}
        rhetoric = rhetorical_features or {}
        argument = argument_features or {}
        info = information_features or {}

        features = {}

        features["fear_propaganda_score"] = self._fear_propaganda(
            emotion, narrative, rhetoric
        )

        features["scapegoating_score"] = self._scapegoating(
            rhetoric, argument
        )

        features["polarization_score"] = self._polarization(
            narrative, rhetoric
        )

        features["emotional_amplification_score"] = self._emotional_amplification(
            emotion, rhetoric
        )

        features["narrative_imbalance_score"] = self._narrative_imbalance(
            argument, info
        )

        return features

    # ------------------------------------------------------------
    # Fear Propaganda
    # ------------------------------------------------------------

    def _fear_propaganda(
        self,
        emotion: Dict[str, float],
        narrative: Dict[str, float],
        rhetoric: Dict[str, float],
    ) -> float:

        emotion_signal = emotion.get("emotion_fear", 0.0)
        rhetoric_signal = rhetoric.get("rhetoric_fear_appeal_score", 0.0)
        narrative_signal = narrative.get("narrative_conflict_term_ratio", 0.0)

        score = (
            emotion_signal * self.config.fear_weight_emotion
            + rhetoric_signal * self.config.fear_weight_rhetoric
            + narrative_signal * self.config.fear_weight_narrative
        )

        return float(score)

    # ------------------------------------------------------------
    # Scapegoating
    # ------------------------------------------------------------

    def _scapegoating(
        self,
        rhetoric: Dict[str, float],
        argument: Dict[str, float],
    ) -> float:

        rhetoric_signal = rhetoric.get("rhetoric_scapegoating_score", 0.0)
        contrast_signal = argument.get("argument_contrast_ratio", 0.0)

        score = (
            rhetoric_signal * self.config.scapegoat_weight_rhetoric
            + contrast_signal * self.config.scapegoat_weight_argument
        )

        return float(score)

    # ------------------------------------------------------------
    # Polarization
    # ------------------------------------------------------------

    def _polarization(
        self,
        narrative: Dict[str, float],
        rhetoric: Dict[str, float],
    ) -> float:

        narrative_signal = narrative.get("narrative_polarization_ratio", 0.0)
        rhetoric_signal = rhetoric.get("rhetoric_loaded_language_score", 0.0)

        score = (
            narrative_signal * self.config.polarization_weight_narrative
            + rhetoric_signal * self.config.polarization_weight_rhetoric
        )

        return float(score)

    # ------------------------------------------------------------
    # Emotional Amplification
    # ------------------------------------------------------------

    def _emotional_amplification(
        self,
        emotion: Dict[str, float],
        rhetoric: Dict[str, float],
    ) -> float:

        anger = emotion.get("emotion_anger", 0.0)
        fear = emotion.get("emotion_fear", 0.0)

        rhetoric_intensity = rhetoric.get("rhetoric_emotional_intensity", 0.0)

        emotion_signal = (anger + fear) / 2

        score = (
            emotion_signal * self.config.emotion_amplification_weight
            + rhetoric_intensity * self.config.rhetoric_amplification_weight
        )

        return float(score)

    # ------------------------------------------------------------
    # Narrative Imbalance
    # ------------------------------------------------------------

    def _narrative_imbalance(
        self,
        argument: Dict[str, float],
        info: Dict[str, float],
    ) -> float:

        claim_density = argument.get("argument_claim_ratio", 0.0)

        evidence_density = info.get("factual_density", 0.0)

        score = (
            claim_density * self.config.narrative_claim_weight
            - evidence_density * self.config.narrative_evidence_weight
        )

        return float(max(score, 0.0))


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def propaganda_pattern_vector(features: Dict[str, float]) -> np.ndarray:

    return make_vector(features, PROPAGANDA_PATTERN_KEYS)