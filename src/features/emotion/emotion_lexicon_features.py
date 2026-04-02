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

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------
# Emotion Lexicon
# ---------------------------------------------------------------------

EMOTION_LEXICON: Dict[str, List[str]] = {
    "anger": ["anger", "angry", "rage", "furious", "hate", "hostile"],
    "fear": ["fear", "scared", "terror", "panic", "afraid"],
    "joy": ["joy", "happy", "delight", "smile", "pleased"],
    "sadness": ["sad", "sorrow", "grief", "cry", "depressed"],
    "surprise": ["surprised", "astonished", "shocked"],
    "disgust": ["disgust", "repulsive", "nasty"],
}


# ---------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------

@dataclass
@register_feature
class EmotionLexiconFeatures(BaseFeature):
    """
    Extracts emotion features using lexicon matching.

    Output Features
    ---------------
    - lexicon_emotion_<emotion>
    - lexicon_emotion_intensity
    - lexicon_emotion_diversity
    """

    name: str = "emotion_lexicon_features"
    description: str = "Lexicon-based emotion feature extractor"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract emotion lexicon features from text.

        Parameters
        ----------
        context : FeatureContext

        Returns
        -------
        Dict[str, float]
        """

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for emotion lexicon extraction")
            return {}

        token_counter = Counter(tokens)

        emotion_counts: Dict[str, int] = {emotion: 0 for emotion in EMOTION_LEXICON}

        for emotion, lex_words in EMOTION_LEXICON.items():
            for word in lex_words:
                emotion_counts[emotion] += token_counter.get(word, 0)

        total_emotion_tokens = sum(emotion_counts.values())

        if total_emotion_tokens == 0:
            total_emotion_tokens = 1

        features: Dict[str, float] = {}

        for emotion, count in emotion_counts.items():
            features[f"lexicon_emotion_{emotion}"] = float(count / total_emotion_tokens)

        # Emotion intensity
        intensity = max(emotion_counts.values()) / total_emotion_tokens
        features["lexicon_emotion_intensity"] = float(intensity)

        # Emotion diversity
        non_zero = sum(1 for v in emotion_counts.values() if v > 0)
        features["lexicon_emotion_diversity"] = float(non_zero / len(emotion_counts))

        logger.debug(
            "Emotion lexicon features extracted | total_tokens=%d",
            len(tokens),
        )

        return features