"""
File Name: emotion_trajectory.py
Module: Emotion Analysis - Trajectory Modeling
Description:
    Models the evolution of emotions across text segments for the TruthLens AI
    system. The module analyzes how emotional signals change throughout a text,
    capturing temporal emotional patterns such as escalation, decline, volatility,
    and shifts. These trajectory features are useful for narrative analysis,
    propaganda detection, and discourse-level emotion modeling.

Dependencies:
    logging
    typing
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Emotion trajectory feature dictionary and optional numerical trajectory vector
"""

import logging
from typing import Dict, List

import numpy as np
import spacy

from emotion_detector import EmotionDetector


logger = logging.getLogger(__name__)


class EmotionTrajectoryAnalyzer:
    """
    Analyzes emotional progression across segments of text.
    """

    def __init__(
        self,
        emotion_detector: EmotionDetector = None,
        spacy_model: str = "en_core_web_sm",
        segment_size: int = 2,
    ) -> None:
        """Initialize trajectory analyzer and NLP pipeline."""

        if not isinstance(segment_size, int) or segment_size <= 0:
            raise ValueError("segment_size must be a positive integer")

        self.segment_size = segment_size
        self.emotion_detector = emotion_detector or EmotionDetector()

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        logger.info("EmotionTrajectoryAnalyzer initialized")

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze emotion trajectory patterns across text segments."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        doc = self.nlp(text)

        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

        if not sentences:
            return self._empty_features()

        segments = self._create_segments(sentences)

        emotion_scores: List[float] = []

        for segment in segments:
            combined = " ".join(segment)

            try:
                result = self.emotion_detector.detect(combined)
            except Exception as exc:
                logger.exception("Emotion detection failed for segment")
                raise RuntimeError("Segment emotion analysis failed") from exc

            intensity = result.get("emotion_intensity", 0.0)

            emotion_scores.append(float(intensity))

        trajectory = self._compute_trajectory_metrics(emotion_scores)

        return trajectory

    def _create_segments(self, sentences: List[str]) -> List[List[str]]:
        """Split sentences into sequential segments."""

        segments: List[List[str]] = []

        for i in range(0, len(sentences), self.segment_size):
            segment = sentences[i : i + self.segment_size]
            segments.append(segment)

        return segments

    def _compute_trajectory_metrics(self, scores: List[float]) -> Dict[str, float]:
        """Compute trajectory statistics from emotion intensity sequence."""

        if not scores:
            return self._empty_features()

        scores_array = np.array(scores, dtype=np.float32)

        slope = self._estimate_trend(scores_array)

        volatility = float(np.std(scores_array))

        peak = float(np.max(scores_array))

        minimum = float(np.min(scores_array))

        mean_intensity = float(np.mean(scores_array))

        return {
            "emotion_trend": slope,
            "emotion_volatility": volatility,
            "emotion_peak": peak,
            "emotion_min": minimum,
            "emotion_mean": mean_intensity,
        }

    def _estimate_trend(self, values: np.ndarray) -> float:
        """Estimate trajectory slope using simple linear regression."""

        if len(values) <= 1:
            return 0.0

        x = np.arange(len(values))

        try:
            slope, _ = np.polyfit(x, values, 1)
            return float(slope)
        except Exception:
            return 0.0

    def _empty_features(self) -> Dict[str, float]:
        """Return default trajectory feature values."""

        return {
            "emotion_trend": 0.0,
            "emotion_volatility": 0.0,
            "emotion_peak": 0.0,
            "emotion_min": 0.0,
            "emotion_mean": 0.0,
        }


def trajectory_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert trajectory feature dictionary into numerical vector."""

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = np.array(list(features.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Trajectory vector conversion failed")
        raise RuntimeError("Failed to convert trajectory features") from exc