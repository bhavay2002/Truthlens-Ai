"""
File Name: emotion_feature_extractor.py
Module: Emotion Analysis - Feature Extraction
Description:
    Combines multiple emotion analysis components to produce a unified emotion
    feature representation for the TruthLens AI system. This module integrates
    emotion classification outputs, emotion intensity estimation, and emotion
    discourse pattern detection to generate a comprehensive feature vector
    suitable for downstream machine learning models.

Dependencies:
    logging
    typing
    numpy
    torch

Inputs:
    Raw text string

Outputs:
    Aggregated emotion feature dictionary and numerical feature vector
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch

from emotion_detector import EmotionDetector
from emotion_intensity import EmotionIntensityEstimator
from emotion_patterns import EmotionPatternDetector


logger = logging.getLogger(__name__)


class EmotionFeatureExtractor:
    """
    Aggregates multiple emotion analysis signals into a unified feature set.
    """

    def __init__(
        self,
        emotion_detector: Optional[EmotionDetector] = None,
        intensity_estimator: Optional[EmotionIntensityEstimator] = None,
        pattern_detector: Optional[EmotionPatternDetector] = None,
    ) -> None:
        """Initialize emotion feature extractor components."""

        self.emotion_detector = emotion_detector or EmotionDetector()
        self.intensity_estimator = intensity_estimator or EmotionIntensityEstimator()
        self.pattern_detector = pattern_detector or EmotionPatternDetector()

        logger.info("EmotionFeatureExtractor initialized")

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract unified emotion features from text."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        features: Dict[str, float] = {}

        try:
            emotion_distribution = self.emotion_detector.detect(text)
            intensity_features = self.intensity_estimator.estimate(text)
            pattern_features = self.pattern_detector.detect_patterns(text)
        except Exception as exc:
            logger.exception("Emotion feature extraction failed")
            raise RuntimeError("Failed to extract emotion features") from exc

        features.update(emotion_distribution)
        features.update(intensity_features)
        features.update(pattern_features)

        return features

    def extract_vector(self, text: str) -> np.ndarray:
        """Convert extracted emotion features into a numeric vector."""

        features = self.extract_features(text)

        try:
            vector = np.array(list(features.values()), dtype=np.float32)
            return vector
        except Exception as exc:
            logger.exception("Emotion feature vector creation failed")
            raise RuntimeError("Failed to convert emotion features to vector") from exc

    def extract_tensor(self, text: str) -> torch.Tensor:
        """Convert extracted emotion features into a PyTorch tensor."""

        vector = self.extract_vector(text)

        try:
            tensor = torch.tensor(vector, dtype=torch.float32)
            return tensor
        except Exception as exc:
            logger.exception("Emotion tensor conversion failed")
            raise RuntimeError("Failed to convert emotion vector to tensor") from exc