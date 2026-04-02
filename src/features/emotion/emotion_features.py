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
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optional Transformer Emotion Model
# ---------------------------------------------------------------------

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    _model.eval()

    TRANSFORMER_AVAILABLE = True

    EMOTION_LABELS = [
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
        "Transformer emotion model not available. Using lexicon fallback."
    )

    EMOTION_LABELS = [
        "anger",
        "fear",
        "joy",
        "sadness",
    ]


# ---------------------------------------------------------------------
# Lexicon Fallback
# ---------------------------------------------------------------------

LEXICON = {
    "anger": {"angry", "rage", "furious", "hate"},
    "fear": {"fear", "terror", "scared", "panic"},
    "joy": {"happy", "joy", "delight", "smile"},
    "sadness": {"sad", "cry", "sorrow", "grief"},
}


def _lexicon_emotions(text: str) -> Dict[str, float]:
    """
    Basic lexicon-based emotion scoring.
    """

    tokens = text.lower().split()
    counts = {emotion: 0 for emotion in LEXICON}

    for token in tokens:
        for emotion, words in LEXICON.items():
            if token in words:
                counts[emotion] += 1

    total = sum(counts.values()) or 1

    return {k: v / total for k, v in counts.items()}


# ---------------------------------------------------------------------
# Emotion Feature Extractor
# ---------------------------------------------------------------------

@dataclass
@register_feature
class EmotionFeatures(BaseFeature):
    """
    Extracts emotion distribution features from text.

    Output features include:
    - emotion probabilities
    - dominant emotion
    - emotional intensity
    """

    name: str = "emotion_features"
    description: str = "Emotion distribution and emotional intensity features"

    def _transformer_emotions(self, text: str) -> Dict[str, float]:
        """
        Compute emotions using transformer classifier.
        """

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

        return {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract emotion-related features.
        """

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        if TRANSFORMER_AVAILABLE:
            emotion_scores = self._transformer_emotions(context.text)
        else:
            emotion_scores = _lexicon_emotions(context.text)

        values = np.array(list(emotion_scores.values()))

        dominant_idx = int(np.argmax(values))
        dominant_emotion = list(emotion_scores.keys())[dominant_idx]

        intensity = float(np.max(values) - np.mean(values))

        features: Dict[str, float] = {}

        for emotion, score in emotion_scores.items():
            features[f"emotion_{emotion}"] = float(score)

        features["emotion_intensity"] = intensity
        features[f"emotion_dominant_{dominant_emotion}"] = 1.0

        logger.debug(
            "Emotion features extracted | dominant=%s intensity=%.4f",
            dominant_emotion,
            intensity,
        )

        return features