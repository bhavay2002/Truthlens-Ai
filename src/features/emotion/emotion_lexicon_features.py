"""
File Name: emotion_lexicon_features.py
Module: Feature Engineering - Emotion Lexicon Features
Description:
    Extracts emotion-related features using a lexicon-based approach.
    This module computes emotion scores by matching tokens against
    predefined emotion lexicons and producing normalized emotion
    distributions.

    The implementation is lightweight, deterministic, and suitable
    for environments where transformer models are unavailable or
    expensive to run. It integrates with the TruthLens feature
    extraction framework via BaseFeature and FeatureRegistry.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing text and optional tokens

Outputs:
    Dict[str, float] containing lexicon-based emotion features
"""

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


# -----------------------------------------------------
# Tokenizer
# -----------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lightweight tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


# -----------------------------------------------------
# Reverse Emotion Lookup (fast)
# -----------------------------------------------------

WORD_TO_EMOTION: Dict[str, str] = {}

for emotion, words in EMOTION_TERMS.items():
    for word in words:
        WORD_TO_EMOTION[word] = emotion


# -----------------------------------------------------
# Feature Extractor
# -----------------------------------------------------

@dataclass
@register_feature
class EmotionLexiconFeatures(BaseFeature):
    """
    Extract emotion features using lexicon matching.

    Output Features
    ---------------
    lexicon_emotion_<emotion>
    lexicon_emotion_intensity
    lexicon_emotion_diversity
    lexicon_emotion_density
    lexicon_emotion_entropy
    """

    name: str = "emotion_lexicon_features"
    description: str = "Lexicon-based emotion feature extractor"

    # -------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for emotion lexicon extraction")
            return {}

        counts: Dict[str, int] = {emotion: 0 for emotion in EMOTION_LABELS}

        for token in tokens:

            emotion = WORD_TO_EMOTION.get(token)

            if emotion:
                counts[emotion] += 1

        total_emotion_tokens = sum(counts.values())

        if total_emotion_tokens == 0:
            total_emotion_tokens = 1

        features: Dict[str, float] = {}

        # -------------------------------------------------
        # Emotion distribution
        # -------------------------------------------------

        for emotion, count in counts.items():
            features[f"lexicon_emotion_{emotion}"] = count / total_emotion_tokens

        # -------------------------------------------------
        # Emotion intensity
        # -------------------------------------------------

        max_count = max(counts.values())
        features["lexicon_emotion_intensity"] = max_count / total_emotion_tokens

        # -------------------------------------------------
        # Emotion diversity
        # -------------------------------------------------

        active_emotions = sum(1 for v in counts.values() if v > 0)

        features["lexicon_emotion_diversity"] = active_emotions / len(EMOTION_LABELS)

        # -------------------------------------------------
        # Emotion density (emotion tokens / total tokens)
        # -------------------------------------------------

        features["lexicon_emotion_density"] = total_emotion_tokens / len(tokens)

        # -------------------------------------------------
        # Emotion entropy
        # -------------------------------------------------

        values = np.array(list(features[f"lexicon_emotion_{e}"] for e in EMOTION_LABELS))

        eps = 1e-9
        entropy = -np.sum(values * np.log(values + eps))

        features["lexicon_emotion_entropy"] = float(entropy)

        logger.debug(
            "Emotion lexicon features extracted | tokens=%d emotion_tokens=%d",
            len(tokens),
            total_emotion_tokens,
        )

        return features