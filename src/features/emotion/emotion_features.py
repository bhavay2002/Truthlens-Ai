"""
File Name: emotion_features.py
Module: Feature Engineering - Emotion Features
Description:
    Extracts emotion-related features from text using transformer-based
    emotion classification models or a lexicon-based fallback. These
    features capture the emotional distribution, polarity, and emotional
    intensity signals within the text.

    The module integrates with the TruthLens feature system through the
    BaseFeature abstraction and FeatureRegistry, enabling automatic
    discovery and execution within feature pipelines.

Dependencies:
    dataclasses
    typing
    logging
    numpy
    transformers (optional)
    torch (optional)

Inputs:
    FeatureContext containing input text

Outputs:
    Dict[str, float] representing emotion distribution and summary metrics
"""

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


# -------------------------------------------------------
# Build Reverse Emotion Lookup
# -------------------------------------------------------

WORD_TO_EMOTION = {}

for emotion, words in EMOTION_TERMS.items():
    for word in words:
        WORD_TO_EMOTION[word] = emotion


# -------------------------------------------------------
# Fast Lexicon Emotion Detector
# -------------------------------------------------------

def _lexicon_emotions(text: str) -> Dict[str, float]:
    """
    Fast lexicon-based emotion detection using reverse lookup.
    Complexity: O(tokens)
    """

    tokens = re.findall(r"\b\w+\b", text.lower())

    counts = {emotion: 0 for emotion in EMOTION_LABELS}

    for token in tokens:

        emotion = WORD_TO_EMOTION.get(token)

        if emotion:
            counts[emotion] += 1

    total = sum(counts.values()) or 1

    return {emotion: counts[emotion] / total for emotion in EMOTION_LABELS}


# -------------------------------------------------------
# Feature Extractor
# -------------------------------------------------------

@dataclass
@register_feature
class EmotionFeatures(BaseFeature):

    name: str = "emotion_features"
    description: str = "20-class emotion distribution and intensity features"

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        emotion_scores = _lexicon_emotions(context.text)

        values = np.array(list(emotion_scores.values()), dtype=np.float32)

        dominant_idx = int(np.argmax(values))
        dominant_emotion = EMOTION_LABELS[dominant_idx]

        intensity = float(np.max(values) - np.mean(values))

        features: Dict[str, float] = {}

        # Emotion distribution
        for emotion, score in emotion_scores.items():
            features[f"emotion_{emotion}"] = float(score)

        # Emotion intensity
        features["emotion_intensity"] = intensity

        # Dominant emotion
        features[f"emotion_dominant_{dominant_emotion}"] = 1.0

        logger.debug(
            "Emotion features extracted | dominant=%s intensity=%.4f",
            dominant_emotion,
            intensity,
        )

        return features