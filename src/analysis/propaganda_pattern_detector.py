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

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class PropagandaPatternConfig:

    fear_weight_emotion: float = 0.35
    fear_weight_rhetoric: float = 0.35
    fear_weight_narrative: float = 0.30

    scapegoat_weight_rhetoric: float = 0.55
    scapegoat_weight_argument: float = 0.45

    polarization_weight_narrative: float = 0.60
    polarization_weight_rhetoric: float = 0.40

    emotion_amplification_weight: float = 0.6
    rhetoric_amplification_weight: float = 0.4

    narrative_claim_weight: float = 0.5
    narrative_evidence_weight: float = 0.5

    clip_outputs: bool = True
    clip_range: tuple[float, float] = (0.0, 1.0)

    enable_validation: bool = True
    enable_debug_metadata: bool = False


# =========================================================
# DETECTOR
# =========================================================

class PropagandaPatternDetector:

    def __init__(self, config: PropagandaPatternConfig | None = None):
        self.config = config or PropagandaPatternConfig()

    # --------------------------------------------------------

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

        # -------------------------
        # CORE SIGNALS
        # -------------------------
        raw = {
            "fear": self._fear(emotion, narrative, rhetoric),
            "scapegoating": self._scapegoating(rhetoric, argument),
            "polarization": self._polarization(narrative, rhetoric),
            "amplification": self._amplification(emotion, rhetoric),
            "imbalance": self._imbalance(argument, info),
        }

        # -------------------------
        # NORMALIZATION
        # -------------------------
        dist = self._normalize(raw)

        # -------------------------
        # GLOBAL INTENSITY
        # -------------------------
        intensity = sum(raw.values()) / (len(raw) + EPS)

        # -------------------------
        # DIVERSITY
        # -------------------------
        diversity = self._entropy(dist)

        features = {
            "fear_propaganda_score": dist["fear"],
            "scapegoating_score": dist["scapegoating"],
            "polarization_score": dist["polarization"],
            "emotional_amplification_score": dist["amplification"],
            "narrative_imbalance_score": dist["imbalance"],
            "propaganda_intensity": self._safe(intensity),
            "propaganda_diversity": self._safe(diversity),
        }

        # -------------------------
        # CLIP
        # -------------------------
        if self.config.clip_outputs:
            features = self._clip(features)

        # -------------------------
        # VALIDATION
        # -------------------------
        if self.config.enable_validation:
            validate_features(features, PROPAGANDA_PATTERN_KEYS)

        return features

    # =========================================================
    # SIGNALS (CONFIG-DRIVEN)
    # =========================================================

    def _fear(self, emotion, narrative, rhetoric):

        e = self._get(emotion, "emotion_expression_ratio")
        r = self._get(rhetoric, "rhetoric_fear_appeal_score")
        n = self._get(narrative, "conflict_intensity", "polarization_ratio")

        return (
            e * self.config.fear_weight_emotion +
            r * self.config.fear_weight_rhetoric +
            n * self.config.fear_weight_narrative
        )

    def _scapegoating(self, rhetoric, argument):

        r = self._get(rhetoric, "rhetoric_scapegoating_score")
        a = self._get(argument, "argument_contrast_ratio")

        return (
            r * self.config.scapegoat_weight_rhetoric +
            a * self.config.scapegoat_weight_argument
        )

    def _polarization(self, narrative, rhetoric):

        n = self._get(narrative, "polarization_ratio")
        r = self._get(rhetoric, "rhetoric_loaded_language_score")

        return (
            n * self.config.polarization_weight_narrative +
            r * self.config.polarization_weight_rhetoric
        )

    def _amplification(self, emotion, rhetoric):

        vals = [
            self._get(emotion, "emotion_expression_ratio"),
            self._get(emotion, "dominant_emotion_strength"),
        ]

        e = sum(vals) / max(len(vals), 1)
        r = self._get(rhetoric, "rhetoric_emotional_appeal_score")

        return (
            e * self.config.emotion_amplification_weight +
            r * self.config.rhetoric_amplification_weight
        )

    def _imbalance(self, argument, info):

        claim = self._get(argument, "argument_claim_ratio")
        evidence = self._get(info, "factual_density")

        # bounded ratio
        return claim / (claim + evidence + EPS)

    # =========================================================
    # UTILS
    # =========================================================

    def _get(self, features: Dict, *keys: str, default: float = 0.0):

        for k in keys:
            v = features.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return float(v)

        return default

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:

        values = np.array(list(scores.values()), dtype=np.float32)

        total = float(values.sum())

        if total < EPS:
            return {k: 0.0 for k in scores}

        norm = values / (total + EPS)

        return dict(zip(scores.keys(), norm.astype(float)))

    def _entropy(self, dist: Dict[str, float]) -> float:

        values = np.array(list(dist.values()), dtype=np.float32)

        if values.sum() < EPS:
            return 0.0

        probs = values / (values.sum() + EPS)

        entropy = -np.sum(probs * np.log(probs + EPS))
        max_entropy = np.log(len(probs))

        return float(entropy / (max_entropy + EPS))

    def _clip(self, features: Dict[str, float]) -> Dict[str, float]:

        low, high = self.config.clip_range

        return {k: float(np.clip(v, low, high)) for k, v in features.items()}

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))


# =========================================================
# VECTOR
# =========================================================

def propaganda_pattern_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, PROPAGANDA_PATTERN_KEYS)