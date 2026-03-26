"""
File Name: emotion_polarization.py
Module: Emotion Analysis - Polarization Detection
Description:
    Computes emotional polarization signals in text for the TruthLens AI system.
    The module measures the balance between positive and negative emotional
    signals and detects emotionally polarized discourse. These features are
    useful for identifying manipulative rhetoric, ideological framing, and
    emotionally charged narratives in media or political communication.

Dependencies:
    logging
    typing
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Emotion polarization feature dictionary and numerical vector
"""

import logging
from typing import Dict, Optional

import numpy as np
import spacy

from src.features.emotion.emotion_detector import EmotionDetector


logger = logging.getLogger(__name__)


class EmotionPolarizationAnalyzer:
    """
    Analyzes emotional polarization between positive and negative emotions.
    """

    POSITIVE_EMOTIONS = {"joy", "trust", "anticipation"}
    NEGATIVE_EMOTIONS = {"anger", "fear", "sadness", "disgust"}

    def __init__(
        self,
        emotion_detector: Optional[EmotionDetector] = None,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        """Initialize polarization analyzer and NLP pipeline."""

        self.emotion_detector = emotion_detector or EmotionDetector()

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        logger.info("EmotionPolarizationAnalyzer initialized")

    def analyze(self, text: str) -> Dict[str, float]:
        """Compute polarization metrics from emotion distribution."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        try:
            emotion_result = self.emotion_detector.detect(text)
        except Exception as exc:
            logger.exception("Emotion detection failed")
            raise RuntimeError("Emotion analysis failed") from exc

        positive_score = self._aggregate_emotions(
            emotion_result, self.POSITIVE_EMOTIONS
        )

        negative_score = self._aggregate_emotions(
            emotion_result, self.NEGATIVE_EMOTIONS
        )

        total = positive_score + negative_score

        if total == 0:
            balance = 0.0
        else:
            balance = (positive_score - negative_score) / total

        polarization_strength = abs(balance)

        features = {
            "emotion_positive_score": float(positive_score),
            "emotion_negative_score": float(negative_score),
            "emotion_polarization_balance": float(balance),
            "emotion_polarization_strength": float(polarization_strength),
        }

        return features

    def _aggregate_emotions(
        self,
        emotion_distribution: Dict[str, float],
        target_emotions: set,
    ) -> float:
        """Aggregate emotion scores for a defined emotion group."""

        score = 0.0

        for emotion in target_emotions:
            key = f"emotion_{emotion}"
            score += emotion_distribution.get(key, 0.0)

        return float(score)


def polarization_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert polarization feature dictionary into a numerical vector."""

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = np.array(list(features.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Polarization vector conversion failed")
        raise RuntimeError("Failed to convert polarization features") from exc
