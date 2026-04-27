# src/features/bias_features.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------
# Lexicons (weighted)
# ---------------------------------------------------------

LOADED_LANGUAGE = {...}
SUBJECTIVE_WORDS = {...}
UNCERTAINTY_WORDS = {...}
POLARIZING_WORDS = {...}
EVALUATIVE_WORDS = {...}

NEGATIONS = {"not", "no", "never", "n't"}


# ---------------------------------------------------------
# Negation-aware weighting
# ---------------------------------------------------------

def _negation_factor(tokens: List[str], idx: int, window: int = 3) -> float:
    """
    Returns scaling factor based on negation proximity.
    """
    start = max(0, idx - window)
    if any(t in NEGATIONS for t in tokens[start:idx]):
        return 0.3
    return 1.0


# ---------------------------------------------------------
# Weighted ratio
# ---------------------------------------------------------

def _weighted_ratio(tokens: List[str], lexicon: Dict[str, float]) -> float:

    if not tokens:
        return 0.0

    score = 0.0

    for i, token in enumerate(tokens):
        if token in lexicon:
            weight = lexicon[token]
            weight *= _negation_factor(tokens, i)
            score += weight

    return score / (len(tokens) + EPS)


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class BiasFeaturesV2(BaseFeature):

    name: str = "bias_features_v2"
    group: str = "bias"  # 🔥 REQUIRED for pipeline
    description: str = "Advanced bias detection (normalized + entropy)"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = context.tokens or _tokenize(text)
        n = len(tokens)

        if n == 0:
            return {}

        # -----------------------------
        # Raw signals
        # -----------------------------

        raw = {
            "loaded": _weighted_ratio(tokens, LOADED_LANGUAGE),
            "subjective": _weighted_ratio(tokens, SUBJECTIVE_WORDS),
            "uncertainty": _weighted_ratio(tokens, UNCERTAINTY_WORDS),
            "polarization": _weighted_ratio(tokens, POLARIZING_WORDS),
            "evaluative": _weighted_ratio(tokens, EVALUATIVE_WORDS),
        }

        # -----------------------------
        # NORMALIZATION (CRITICAL)
        # -----------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = float(values.sum())

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -----------------------------
        # ENTROPY (diversity)
        # -----------------------------

        probs = np.array(list(dist.values()), dtype=np.float32)

        if probs.sum() < EPS:
            entropy = 0.0
        else:
            entropy_raw = -np.sum(probs * np.log(probs + EPS))
            entropy = entropy_raw / (np.log(len(probs)) + EPS)

        # -----------------------------
        # Structural signals (FIXED)
        # -----------------------------

        exclam = text.count("!")
        exclamation_density = exclam / (n + EPS)

        caps_ratio = sum(
            1 for w in text.split() if w.isupper() and len(w) > 2
        ) / (n + EPS)

        # -----------------------------
        # Intensity
        # -----------------------------

        intensity = float(np.mean(list(raw.values())))

        # -----------------------------
        # OUTPUT
        # -----------------------------

        return {
            "bias_loaded": self._safe(dist["loaded"]),
            "bias_subjective": self._safe(dist["subjective"]),
            "bias_uncertainty": self._safe(dist["uncertainty"]),
            "bias_polarization": self._safe(dist["polarization"]),
            "bias_evaluative": self._safe(dist["evaluative"]),
            "bias_intensity": self._safe(intensity),
            "bias_diversity": self._safe(entropy),
            "bias_caps_ratio": self._safe(caps_ratio),
            "bias_exclamation_density": self._safe(exclamation_density),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

# Backward-compat alias used across the inference layer.
BiasFeatures = BiasFeaturesV2

