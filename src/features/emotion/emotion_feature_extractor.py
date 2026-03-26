"""
File Name: emotion_feature_extractor.py
Module: Emotion Analysis - Feature Extraction

Description:
    Aggregates outputs from multiple emotion analysis components
    to produce a unified feature representation for TruthLens AI.
"""

import logging
from typing import Dict, Optional, Iterable, List

import numpy as np
import torch

from src.features.emotion.emotion_detector import EmotionDetector
from src.features.emotion.emotion_intensity import EmotionIntensityEstimator
from src.features.emotion.emotion_patterns import EmotionPatternDetector


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Deterministic Feature Schema
# ---------------------------------------------------------

EMOTION_FEATURE_SCHEMA = [
    "emotion_anger",
    "emotion_fear",
    "emotion_joy",
    "emotion_sadness",
    "emotion_surprise",
    "emotion_disgust",
    "emotion_trust",
    "emotion_anticipation",
    "emotion_intensity",

    "emotion_lexicon_intensity",
    "intensifier_ratio",
    "exclamation_intensity",
    "question_intensity",
    "ellipsis_intensity",
    "capitalization_intensity",
    "adjective_amplification",
    "adverb_amplification",
    "repetition_intensity",

    "emotion_contrast_ratio",
    "emotion_escalation_ratio",
    "emotion_negation_ratio",
    "emotion_repetition_ratio",
    "emotion_exclamation_ratio",
    "emotion_question_ratio",
    "emotion_ellipsis_ratio",
    "emotion_sentence_variability",
]


class EmotionFeatureExtractor:

    def __init__(
        self,
        emotion_detector: Optional[EmotionDetector] = None,
        intensity_estimator: Optional[EmotionIntensityEstimator] = None,
        pattern_detector: Optional[EmotionPatternDetector] = None,
        device: Optional[str] = None,
    ):

        self.emotion_detector = emotion_detector or EmotionDetector()
        self.intensity_estimator = intensity_estimator or EmotionIntensityEstimator()
        self.pattern_detector = pattern_detector or EmotionPatternDetector()

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("EmotionFeatureExtractor initialized")

    # -----------------------------------------------------

    def extract_features(self, text: str) -> Dict[str, float]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be non-empty")

        emotion = self.emotion_detector.detect(text)
        intensity = self.intensity_estimator.estimate(text)
        patterns = self.pattern_detector.detect_patterns(text)

        features: Dict[str, float] = {}

        features.update(emotion)
        features.update(intensity)
        features.update(patterns)

        return features

    # -----------------------------------------------------

    def extract_vector(self, text: str) -> np.ndarray:

        features = self.extract_features(text)

        vector = np.array(
            [features.get(name, 0.0) for name in EMOTION_FEATURE_SCHEMA],
            dtype=np.float32,
        )

        return vector

    # -----------------------------------------------------

    def extract_tensor(self, text: str) -> torch.Tensor:

        vector = self.extract_vector(text)

        return torch.tensor(vector, dtype=torch.float32, device=self.device)

    # -----------------------------------------------------

    def extract_batch(self, texts: Iterable[str]) -> List[np.ndarray]:

        results = []

        for text in texts:
            results.append(self.extract_vector(text))

        return results
