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
import re
from typing import Any, Dict

import numpy as np

from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
from src.features.emotion.emotion_detector import EmotionDetector
from src.features.emotion.emotion_intensity import EmotionIntensityEstimator

try:
    from src.features.emotion.emotion_polarization import (
        EmotionPolarizationAnalyzer,
    )
except Exception:  # pragma: no cover - optional dependency drift
    EmotionPolarizationAnalyzer = None  # type: ignore[assignment]

try:
    from src.features.emotion.emotion_trajectory import EmotionTrajectoryAnalyzer
except Exception:  # pragma: no cover - optional dependency drift
    EmotionTrajectoryAnalyzer = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class EmotionPipeline:
    """
    Unified pipeline for extracting emotion-related features.
    """

    def __init__(self) -> None:
        """Initialize emotion analysis components."""

        self.detector = EmotionDetector()
        self.intensity = EmotionIntensityEstimator()
        self.targets = EmotionTargetAnalyzer()
        self.polarization = self._build_optional_component(
            "EmotionPolarizationAnalyzer",
            EmotionPolarizationAnalyzer,
        )
        self.trajectory = self._build_optional_component(
            "EmotionTrajectoryAnalyzer",
            EmotionTrajectoryAnalyzer,
        )

        logger.info("EmotionPipeline initialized")

    def _build_optional_component(self, name: str, cls):
        if cls is None:
            logger.warning("%s unavailable. Falling back to heuristic features.", name)
            return None

        try:
            return cls()
        except Exception as exc:
            logger.warning(
                "%s initialization failed (%s). Falling back to heuristic features.",
                name,
                exc,
            )
            return None

    def _compute_pattern_features(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"\b[a-zA-Z]+\b", text)
        if not tokens:
            return {
                "emotion_pattern_exclamation_ratio": 0.0,
                "emotion_pattern_question_ratio": 0.0,
                "emotion_pattern_caps_ratio": 0.0,
                "emotion_pattern_repetition_ratio": 0.0,
            }

        lower_tokens = [token.lower() for token in tokens]
        unique_tokens = set(lower_tokens)
        repeated_tokens = len(tokens) - len(unique_tokens)
        all_caps_tokens = sum(1 for token in tokens if token.isupper() and len(token) > 1)

        text_length = max(len(text), 1)
        token_length = max(len(tokens), 1)
        return {
            "emotion_pattern_exclamation_ratio": float(text.count("!") / text_length),
            "emotion_pattern_question_ratio": float(text.count("?") / text_length),
            "emotion_pattern_caps_ratio": float(all_caps_tokens / token_length),
            "emotion_pattern_repetition_ratio": float(repeated_tokens / token_length),
        }

    def _fallback_polarization(
        self,
        emotion_distribution: Dict[str, float],
    ) -> Dict[str, float]:
        positive_keys = (
            "emotion_joy",
            "emotion_trust",
            "emotion_anticipation",
        )
        negative_keys = (
            "emotion_anger",
            "emotion_fear",
            "emotion_sadness",
            "emotion_disgust",
        )

        positive = float(sum(float(emotion_distribution.get(key, 0.0)) for key in positive_keys))
        negative = float(sum(float(emotion_distribution.get(key, 0.0)) for key in negative_keys))
        total = positive + negative
        balance = 0.0 if total == 0.0 else (positive - negative) / total

        return {
            "emotion_positive_score": positive,
            "emotion_negative_score": negative,
            "emotion_polarization_balance": float(balance),
            "emotion_polarization_strength": float(abs(balance)),
        }

    def _fallback_trajectory(self, text: str) -> Dict[str, float]:
        sentences = [segment.strip() for segment in re.split(r"[.!?]+", text) if segment.strip()]
        if not sentences:
            return {
                "emotion_trend": 0.0,
                "emotion_volatility": 0.0,
                "emotion_peak": 0.0,
                "emotion_min": 0.0,
                "emotion_mean": 0.0,
            }

        scores: list[float] = []
        for sentence in sentences:
            result = self.detector.detect(sentence)
            scores.append(float(result.get("emotion_intensity", 0.0)))

        if len(scores) == 1:
            trend = 0.0
        else:
            x = np.arange(len(scores), dtype=np.float32)
            y = np.asarray(scores, dtype=np.float32)
            trend = float(np.polyfit(x, y, 1)[0])

        scores_arr = np.asarray(scores, dtype=np.float32)
        return {
            "emotion_trend": trend,
            "emotion_volatility": float(np.std(scores_arr)),
            "emotion_peak": float(np.max(scores_arr)),
            "emotion_min": float(np.min(scores_arr)),
            "emotion_mean": float(np.mean(scores_arr)),
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """Run full emotion analysis pipeline."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            emotion_distribution = self.detector.detect(text)

            intensity_features = self.intensity.estimate(text)

            pattern_features = self._compute_pattern_features(text)

            polarization_features = (
                self.polarization.analyze(text)
                if self.polarization is not None
                else self._fallback_polarization(emotion_distribution)
            )

            trajectory_features = (
                self.trajectory.analyze(text)
                if self.trajectory is not None
                else self._fallback_trajectory(text)
            )

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

        values: list[float] = []

        for section_name in (
            "emotion_distribution",
            "emotion_intensity",
            "emotion_patterns",
            "emotion_polarization",
            "emotion_trajectory",
            "emotion_targets",
        ):
            section = features.get(section_name, {})
            if isinstance(section, dict):
                for key in sorted(section.keys()):
                    value = section[key]
                    if isinstance(value, (int, float)):
                        values.append(float(value))

        if not values:
            raise ValueError("No numeric emotion features extracted")

        try:
            vector = np.array(values, dtype=np.float32)
            return vector
        except Exception as exc:
            logger.exception("Emotion feature vector conversion failed")
            raise RuntimeError("Failed to convert emotion features") from exc
