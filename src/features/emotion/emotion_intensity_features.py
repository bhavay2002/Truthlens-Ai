"""
File Name: emotion_intensity_features.py
Module: Feature Engineering - Emotion Intensity Features
Description:
    Computes emotion intensity metrics from text using transformer-based
    emotion predictions when available, or a lexicon-based fallback.
    The goal of this module is to measure the strength, concentration,
    and volatility of emotional signals within a piece of text.

    These features are particularly useful for detecting emotional
    manipulation, propaganda, outrage narratives, and polarizing language.

    The module integrates with the TruthLens feature framework via
    BaseFeature and FeatureRegistry.

Dependencies:
    dataclasses
    typing
    logging
    numpy
    transformers (optional)
    torch (optional)
    re

Inputs:
    FeatureContext containing input text

Outputs:
    Dict[str, float] representing emotion intensity statistics
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


# ------------------------------------------------------------
# Build Reverse Emotion Lookup (fast lexicon lookup)
# ------------------------------------------------------------

WORD_TO_EMOTION = {}

for emotion, words in EMOTION_TERMS.items():
    for word in words:
        WORD_TO_EMOTION[word] = emotion


# ------------------------------------------------------------
# Optional Transformer Emotion Model
# ------------------------------------------------------------

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    _model.eval()

    TRANSFORMER_AVAILABLE = True

    TRANSFORMER_LABELS = [
        "anger",
        "disgust",
        "fear",
        "joy",
        "neutral",
        "sadness",
        "surprise",
    ]

except Exception:  # noqa: BLE001
    TRANSFORMER_AVAILABLE = False
    _tokenizer = None
    _model = None

    logger.warning(
        "Transformer emotion model not available. "
        "EmotionIntensityFeatures will use lexicon fallback."
    )


# ------------------------------------------------------------
# Fast Lexicon-based Emotion Detection
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Feature Extractor
# ------------------------------------------------------------

@dataclass
@register_feature
class EmotionIntensityFeatures(BaseFeature):
    """
    Computes higher-level emotion intensity statistics.

    Output Features
    ---------------
    emotion_intensity_max
    emotion_intensity_mean
    emotion_intensity_std
    emotion_intensity_range
    emotion_intensity_entropy
    """

    name: str = "emotion_intensity_features"
    description: str = "Emotion strength and concentration metrics"

    # --------------------------------------------------------

    def _transformer_emotions(self, text: str) -> Dict[str, float]:
        """
        Compute emotion distribution using transformer model.
        Maps transformer outputs into TruthLens 20-emotion schema.
        """
        if not TRANSFORMER_AVAILABLE or _tokenizer is None or _model is None:
            return {emotion: 0.0 for emotion in EMOTION_LABELS}

        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = _model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()

        scores = {emotion: 0.0 for emotion in EMOTION_LABELS}

        for label, prob in zip(TRANSFORMER_LABELS, probs):

            if label in scores:
                scores[label] = float(prob)

        return scores

    # --------------------------------------------------------

    def _hybrid_emotions(self, text: str) -> Dict[str, float]:
        """
        Combine transformer and lexicon emotion signals.
        """

        transformer_scores = {}
        lexicon_scores = _lexicon_emotions(text)

        if TRANSFORMER_AVAILABLE:
            transformer_scores = self._transformer_emotions(text)

        scores = {}

        for emotion in EMOTION_LABELS:

            t = transformer_scores.get(emotion, 0.0)
            l = lexicon_scores.get(emotion, 0.0)

            scores[emotion] = 0.7 * t + 0.3 * l
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    # --------------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")
        if not context.text.strip():
            return {}

        scores = self._hybrid_emotions(context.text)

        values = np.array(list(scores.values()), dtype=np.float32)

        max_intensity = float(np.max(values))
        mean_intensity = float(np.mean(values))
        std_intensity = float(np.std(values))
        range_intensity = float(np.max(values) - np.min(values))

        # Entropy of emotion distribution
        eps = 1e-9
        entropy = float(-np.sum(values * np.log(values + eps)))

        features: Dict[str, float] = {
            "emotion_intensity_max": max_intensity,
            "emotion_intensity_mean": mean_intensity,
            "emotion_intensity_std": std_intensity,
            "emotion_intensity_range": range_intensity,
            "emotion_intensity_entropy": entropy,
        }

        logger.debug(
            "Emotion intensity features extracted | max=%.4f mean=%.4f",
            max_intensity,
            mean_intensity,
        )

        return features