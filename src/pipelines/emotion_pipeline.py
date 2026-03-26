 """
File Name: emotion_pipeline.py
Module: Emotion Analysis - Pipeline
Description:
    Implements the unified emotion analysis pipeline used in the TruthLens AI
    system. The pipeline orchestrates multiple emotion analysis components
    including emotion detection, emotion intensity estimation, emotion pattern
    analysis, emotion polarization detection, emotion trajectory modeling, and
    emotion target analysis. It produces a consolidated emotional feature
    representation suitable for downstream models and analytical modules.

Dependencies:
    logging
    typing
    numpy
    emotion_detector
    emotion_intensity
    emotion_patterns
    emotion_polarization
    emotion_trajectory
    emotion_target_analysis

Inputs:
    Raw text string

Outputs:
    Aggregated emotion feature dictionary and numerical feature vector
"""

import logging
from typing import Dict, Any

import numpy as np

from emotion_detector import EmotionDetector
from emotion_intensity import EmotionIntensityEstimator
from emotion_patterns import EmotionPatternDetector
from emotion_polarization import EmotionPolarizationAnalyzer
from emotion_trajectory import EmotionTrajectoryAnalyzer
from emotion_target_analysis import EmotionTargetAnalyzer


logger = logging.getLogger(__name__)


class EmotionPipeline:
    """
    Unified pipeline for extracting emotion-related features.
    """

    def __init__(self) -> None:
        """Initialize emotion analysis components."""

        self.detector = EmotionDetector()
        self.intensity = EmotionIntensityEstimator()
        self.patterns = EmotionPatternDetector()
        self.polarization = EmotionPolarizationAnalyzer()
        self.trajectory = EmotionTrajectoryAnalyzer()
        self.targets = EmotionTargetAnalyzer()

        logger.info("EmotionPipeline initialized")

    def analyze(self, text: str) -> Dict[str, Any]:
        """Run full emotion analysis pipeline."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            emotion_distribution = self.detector.detect(text)

            intensity_features = self.intensity.estimate(text)

            pattern_features = self.patterns.detect_patterns(text)

            polarization_features = self.polarization.analyze(text)

            trajectory_features = self.trajectory.analyze(text)

            target_features = self.targets.analyze(text)

        except Exception as exc:
            logger.exception("Emotion pipeline failed")
            raise RuntimeError("Emotion analysis pipeline execution failed") from exc

        features = {
            "emotion_distribution": emotion_distribution,
            "emotion_intensity": intensity_features,
            "emotion_patterns": pattern_features,
            "emotion_polarization": polarization_features,
            "emotion_trajectory": trajectory_features,
            "emotion_targets": target_features,
        }

        return features

    def extract_vector(self, text: str) -> np.ndarray:
        """Convert emotion features into a numeric vector."""

        features = self.analyze(text)

        values = []

        for section in features.values():
            if isinstance(section, dict):
                values.extend(
                    v for v in section.values()
                    if isinstance(v, (int, float))
                )

        if not values:
            raise ValueError("No numeric emotion features extracted")

        try:
            vector = np.array(values, dtype=np.float32)
            return vector
        except Exception as exc:
            logger.exception("Emotion feature vector conversion failed")
            raise RuntimeError("Failed to convert emotion features") from exc