from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.analysis.feature_schema import (
    PROPAGANDA_PATTERN_KEYS,
    make_vector,
    validate_features,
)

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

    # Safety
    clip_outputs: bool = True
    clip_range: tuple[float, float] = (0.0, 1.0)

    # 🔥 NEW: debug + validation
    enable_validation: bool = True
    enable_debug_metadata: bool = False


# ------------------------------------------------------------
# Detector
# ------------------------------------------------------------

class PropagandaPatternDetector:

    def __init__(self, config: PropagandaPatternConfig | None = None):
        self.config = config or PropagandaPatternConfig()
        logger.info("PropagandaPatternDetector initialized")

    # ------------------------------------------------------------
    # Safe feature access
    # ------------------------------------------------------------

    def _get_feature(self, features: Dict, *keys: str, default: float = 0.0) -> float:
        for key in keys:
            value = features.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if np.isnan(value) or np.isinf(value):
                    return default
                return float(value)
        return default

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

        debug = {}

        # --------------------------------------------------
        # Core signals
        # --------------------------------------------------

        fear = self._fear_propaganda(emotion, narrative, rhetoric)
        scapegoat = self._scapegoating(rhetoric, argument)
        polar = self._polarization(narrative, rhetoric)
        amplification = self._emotional_amplification(emotion, rhetoric)
        imbalance = self._narrative_imbalance(argument, info)

        features = {
            "fear_propaganda_score": fear,
            "scapegoating_score": scapegoat,
            "polarization_score": polar,
            "emotional_amplification_score": amplification,
            "narrative_imbalance_score": imbalance,
        }

        # --------------------------------------------------
        # Optional debug metadata
        # --------------------------------------------------

        if self.config.enable_debug_metadata:
            debug["components"] = {
                "fear": fear,
                "scapegoating": scapegoat,
                "polarization": polar,
                "amplification": amplification,
                "imbalance": imbalance,
            }

        # --------------------------------------------------
        # Clipping (stability)
        # --------------------------------------------------

        if self.config.clip_outputs:
            features = self._clip(features)

        # --------------------------------------------------
        # Validation (CRITICAL)
        # --------------------------------------------------

        if self.config.enable_validation:
            valid = validate_features(features, PROPAGANDA_PATTERN_KEYS)
            if not valid:
                logger.warning("Propaganda features failed validation")

        if self.config.enable_debug_metadata:
            features["_debug"] = debug

        return features

    # ------------------------------------------------------------
    # Feature Computations
    # ------------------------------------------------------------

    def _fear_propaganda(self, emotion, narrative, rhetoric) -> float:

        emotion_signal = self._get_feature(
            emotion, "emotion_fear", "emotion_expression_ratio"
        )

        rhetoric_signal = self._get_feature(
            rhetoric, "rhetoric_fear_appeal_score"
        )

        narrative_signal = self._get_feature(
            narrative, "conflict_verb_ratio", "polarization_ratio"
        )

        return float(
            emotion_signal * self.config.fear_weight_emotion
            + rhetoric_signal * self.config.fear_weight_rhetoric
            + narrative_signal * self.config.fear_weight_narrative
        )

    def _scapegoating(self, rhetoric, argument) -> float:

        rhetoric_signal = self._get_feature(
            rhetoric, "rhetoric_scapegoating_score"
        )

        contrast_signal = self._get_feature(
            argument, "argument_contrast_ratio"
        )

        return float(
            rhetoric_signal * self.config.scapegoat_weight_rhetoric
            + contrast_signal * self.config.scapegoat_weight_argument
        )

    def _polarization(self, narrative, rhetoric) -> float:

        narrative_signal = self._get_feature(
            narrative, "polarization_ratio"
        )

        rhetoric_signal = self._get_feature(
            rhetoric, "rhetoric_loaded_language_score"
        )

        return float(
            narrative_signal * self.config.polarization_weight_narrative
            + rhetoric_signal * self.config.polarization_weight_rhetoric
        )

    def _emotional_amplification(self, emotion, rhetoric) -> float:

        anger = self._get_feature(emotion, "emotion_anger")
        fear = self._get_feature(emotion, "emotion_fear", "emotion_expression_ratio")

        rhetoric_intensity = self._get_feature(
            rhetoric, "rhetoric_emotional_intensity"
        )

        emotion_signal = (anger + fear) / 2

        return float(
            emotion_signal * self.config.emotion_amplification_weight
            + rhetoric_intensity * self.config.rhetoric_amplification_weight
        )

    def _narrative_imbalance(self, argument, info) -> float:

        claim_density = self._get_feature(
            argument, "argument_claim_ratio"
        )

        evidence_density = self._get_feature(
            info, "factual_density"
        )

        score = (
            claim_density * self.config.narrative_claim_weight
            - evidence_density * self.config.narrative_evidence_weight
        )

        return float(max(score, 0.0))

    # ------------------------------------------------------------
    # Output Safety
    # ------------------------------------------------------------

    def _clip(self, features: Dict[str, float]) -> Dict[str, float]:

        low, high = self.config.clip_range

        return {
            k: float(np.clip(v, low, high))
            for k, v in features.items()
        }


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def propaganda_pattern_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, PROPAGANDA_PATTERN_KEYS)