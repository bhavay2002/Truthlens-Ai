# src/features/emotion/emotion_lexicon_features.py

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    EMOTION_TERMS,
)

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# -----------------------------------------------------
# Tokenizer
# -----------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


# -----------------------------------------------------
# Reverse lookup
# -----------------------------------------------------

WORD_TO_EMOTION: Dict[str, str] = {
    word: emotion
    for emotion, words in EMOTION_TERMS.items()
    for word in words
}


# -----------------------------------------------------
# Feature extractor
# -----------------------------------------------------

@dataclass
@register_feature
class EmotionLexiconFeatures(BaseFeature):

    name: str = "emotion_lexicon_features"
    group: str = "emotion"
    description: str = "Calibrated lexicon emotion features"

    # -------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = context.tokens or _tokenize(text)
        n_tokens = len(tokens)

        if n_tokens == 0:
            return {}

        # -------------------------
        # Counts
        # -------------------------

        counts = {emotion: 0 for emotion in EMOTION_LABELS}

        for t in tokens:
            emo = WORD_TO_EMOTION.get(t)
            if emo:
                counts[emo] += 1

        total_hits = sum(counts.values())

        # -------------------------
        # Distribution
        # -------------------------

        values = np.array([counts[e] for e in EMOTION_LABELS], dtype=np.float32)

        if total_hits > 0:
            dist = values / (total_hits + EPS)
        else:
            dist = np.zeros_like(values)

        # -------------------------
        # Coverage (CRITICAL)
        # -------------------------

        coverage = total_hits / (n_tokens + EPS)

        # -------------------------
        # Intensity (STRONGER)
        # -------------------------

        l2_intensity = float(np.linalg.norm(dist))

        max_intensity = float(np.max(dist))

        # -------------------------
        # Diversity
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # Entropy (FIXED)
        # -------------------------

        if dist.sum() > 0:
            entropy_raw = -np.sum(dist * np.log(dist + EPS))
            entropy = entropy_raw / (np.log(len(dist)) + EPS)
        else:
            entropy = 0.0

        # -------------------------
        # Output
        # -------------------------

        features: Dict[str, float] = {}

        for i, emotion in enumerate(EMOTION_LABELS):
            features[f"lexicon_emotion_{emotion}"] = self._safe(dist[i])

        features.update({
            "lexicon_emotion_coverage": self._safe(coverage),
            "lexicon_emotion_intensity_l2": self._safe(l2_intensity),
            "lexicon_emotion_intensity_max": self._safe(max_intensity),
            "lexicon_emotion_diversity": self._safe(diversity),
            "lexicon_emotion_entropy": self._safe(entropy),
        })

        return features

    # -------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))