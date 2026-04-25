# src/features/emotion/emotion_features.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict

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


# -------------------------------------------------------
# Reverse lookup
# -------------------------------------------------------

WORD_TO_EMOTION = {
    word: emotion
    for emotion, words in EMOTION_TERMS.items()
    for word in words
}


# -------------------------------------------------------
# Emotion groups (OPTIONAL BUT IMPORTANT)
# -------------------------------------------------------

POSITIVE_EMOTIONS = {
    "joy", "trust", "love", "optimism"
}

NEGATIVE_EMOTIONS = {
    "anger", "fear", "sadness", "disgust"
}


# -------------------------------------------------------
# Lexicon detector
# -------------------------------------------------------

def _lexicon_emotions(text: str):

    tokens = re.findall(r"\b\w+\b", text.lower())

    counts = {emotion: 0 for emotion in EMOTION_LABELS}

    for token in tokens:
        emo = WORD_TO_EMOTION.get(token)
        if emo:
            counts[emo] += 1

    total_hits = sum(counts.values())
    total_tokens = len(tokens)

    return counts, total_hits, total_tokens


# -------------------------------------------------------
# Feature extractor
# -------------------------------------------------------

@dataclass
@register_feature
class EmotionFeatures(BaseFeature):

    name: str = "emotion_features"
    group: str = "emotion"

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        counts, total_hits, total_tokens = _lexicon_emotions(text)

        if total_tokens == 0:
            return {}

        # -------------------------
        # DISTRIBUTION (normalized)
        # -------------------------

        values = np.array([counts[e] for e in EMOTION_LABELS], dtype=np.float32)

        if total_hits == 0:
            dist = np.zeros_like(values)
        else:
            dist = values / (total_hits + EPS)

        # -------------------------
        # COVERAGE (CRITICAL)
        # -------------------------

        coverage = total_hits / (total_tokens + EPS)

        # -------------------------
        # ENTROPY
        # -------------------------

        if dist.sum() > 0:
            entropy_raw = -np.sum(dist * np.log(dist + EPS))
            entropy = entropy_raw / (np.log(len(dist)) + EPS)
        else:
            entropy = 0.0

        # -------------------------
        # INTENSITY (FIXED)
        # -------------------------

        intensity = float(np.linalg.norm(dist))  # stable

        # -------------------------
        # POLARITY
        # -------------------------

        pos = sum(dist[EMOTION_LABELS.index(e)] for e in POSITIVE_EMOTIONS if e in EMOTION_LABELS)
        neg = sum(dist[EMOTION_LABELS.index(e)] for e in NEGATIVE_EMOTIONS if e in EMOTION_LABELS)

        polarity = pos - neg  # [-1, 1] approx

        # -------------------------
        # DOMINANT
        # -------------------------

        dominant_idx = int(np.argmax(dist))
        dominant_emotion = EMOTION_LABELS[dominant_idx]

        # -------------------------
        # OUTPUT
        # -------------------------

        features: Dict[str, float] = {}

        for i, emotion in enumerate(EMOTION_LABELS):
            features[f"emotion_{emotion}"] = self._safe(dist[i])

        features.update({
            "emotion_coverage": self._safe(coverage),
            "emotion_intensity": self._safe(intensity),
            "emotion_entropy": self._safe(entropy),
            "emotion_polarity": self._safe((polarity + 1) / 2),  # normalize to [0,1]
        })

        features[f"emotion_dominant_{dominant_emotion}"] = 1.0

        return features

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))