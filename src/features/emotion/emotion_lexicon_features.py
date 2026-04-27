# src/features/emotion/emotion_lexicon_features.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_matcher import LexiconMatcher, to_token_array
from src.features.base.numerics import normalized_entropy
from src.features.base.tokenization import ensure_tokens_word

from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    EMOTION_TERMS,
)

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# -----------------------------------------------------
# Reverse lookup
# -----------------------------------------------------

WORD_TO_EMOTION: Dict[str, str] = {
    word: emotion
    for emotion, words in EMOTION_TERMS.items()
    for word in words
}


# -----------------------------------------------------
# Vectorized matchers (one per emotion label, built once)
# -----------------------------------------------------

_EMOTION_LEX_MATCHERS = {
    emotion: LexiconMatcher(EMOTION_TERMS.get(emotion, ()), name=emotion)
    for emotion in EMOTION_LABELS
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

        tokens = ensure_tokens_word(context, text)
        n_tokens = len(tokens)

        if n_tokens == 0:
            return {}

        # -------------------------
        # Counts (vectorized)
        # -------------------------

        tokens_arr = to_token_array(tokens)
        counts = {
            emotion: _EMOTION_LEX_MATCHERS[emotion].count_in_tokens(tokens_arr)
            for emotion in EMOTION_LABELS
        }

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

        entropy = normalized_entropy(dist)

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

    # -------------------------------------------------

    def extract_batch(self, contexts):
        return [self.extract(ctx) for ctx in contexts]