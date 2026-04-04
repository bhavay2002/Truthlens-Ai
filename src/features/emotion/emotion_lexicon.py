"""
Wrapper module providing the EmotionLexiconAnalyzer interface expected by the API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.features.base.base_feature import FeatureContext
from src.features.emotion.emotion_lexicon_features import EmotionLexiconFeatures
from src.features.emotion.emotion_schema import EMOTION_LABELS


@dataclass
class EmotionResult:
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    emotion_distribution: Dict[str, float]


class EmotionLexiconAnalyzer:
    """High-level emotion analysis interface using lexicon-based features."""

    def __init__(self) -> None:
        self._extractor = EmotionLexiconFeatures()

    def analyze(self, text: str) -> EmotionResult:
        context = FeatureContext(text=text)
        features = self._extractor.extract(context)

        emotion_scores: Dict[str, float] = {}
        for emotion in EMOTION_LABELS:
            key = f"lexicon_emotion_{emotion}"
            emotion_scores[emotion] = round(features.get(key, 0.0), 4)

        if emotion_scores:
            dominant = max(emotion_scores, key=lambda k: emotion_scores[k])
        else:
            dominant = "neutral"

        total = sum(emotion_scores.values()) or 1.0
        distribution = {k: round(v / total, 4) for k, v in emotion_scores.items()}

        return EmotionResult(
            dominant_emotion=dominant,
            emotion_scores=emotion_scores,
            emotion_distribution=distribution,
        )
