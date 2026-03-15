"""
TruthLens AI
Emotion Detection Pipeline

Central orchestration module for emotion analysis.

Combines signals from:
    • emotion_lexicon.py
    • emotion_intensity.py
    • manipulation_patterns.py

Capabilities
------------
• Emotion distribution
• Dominant emotion detection
• Emotion intensity scoring
• Emotional manipulation detection
• Emotion entropy (diversity metric)
"""

from __future__ import annotations

from typing import Dict
from dataclasses import dataclass
import math

from .emotion_lexicon import EmotionLexiconAnalyzer
from .emotion_intensity import EmotionIntensityAnalyzer
from .manipulation_patterns import detect_emotion_manipulation


# ---------------------------------------------------------
# Result Structure
# ---------------------------------------------------------

@dataclass
class EmotionDetectionResult:
    emotion_scores: Dict[str, float]
    dominant_emotion: str
    emotion_entropy: float
    emotion_intensity: float
    manipulation_score: float
    manipulation_type: str


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def compute_emotion_entropy(emotion_scores: Dict[str, float]) -> float:
    """
    Compute entropy of emotion distribution.

    High entropy -> diverse emotions
    Low entropy -> single dominant emotion
    """

    entropy = 0.0

    for value in emotion_scores.values():

        if value > 0:
            entropy -= value * math.log(value)

    return round(entropy, 4)


def get_dominant_emotion(emotion_scores: Dict[str, float]) -> str:

    if not emotion_scores:
        return "neutral"

    dominant = max(emotion_scores, key=emotion_scores.get)

    if emotion_scores[dominant] == 0:
        return "neutral"

    return dominant


# ---------------------------------------------------------
# Emotion Detection Pipeline
# ---------------------------------------------------------

class EmotionDetector:
    """
    Main emotion analysis engine for TruthLens AI.
    """

    def __init__(self):

        self.lexicon_analyzer = EmotionLexiconAnalyzer()
        self.intensity_analyzer = EmotionIntensityAnalyzer()

    # -----------------------------------------------------

    def analyze(self, text: str) -> EmotionDetectionResult:

        # -------------------------------------------------
        # 1. Base emotion detection
        # -------------------------------------------------

        lexicon_result = self.lexicon_analyzer.analyze(text)

        emotion_scores = lexicon_result.emotion_scores

        # -------------------------------------------------
        # 2. Emotion intensity
        # -------------------------------------------------

        intensity_result = self.intensity_analyzer.analyze(text)

        intensity_score = intensity_result.intensity_score

        # -------------------------------------------------
        # 3. Emotional manipulation
        # -------------------------------------------------

        manipulation_result = detect_emotion_manipulation(text)

        manipulation_score = manipulation_result["manipulation_score"]
        manipulation_type = manipulation_result["manipulation_type"]

        # -------------------------------------------------
        # 4. Dominant emotion
        # -------------------------------------------------

        dominant_emotion = get_dominant_emotion(emotion_scores)

        # -------------------------------------------------
        # 5. Emotion amplification (intensity adjusted)
        # -------------------------------------------------

        weighted_emotions = {
            emotion: round(score * (1 + intensity_score), 4)
            for emotion, score in emotion_scores.items()
        }

        # -------------------------------------------------
        # 6. Emotion entropy
        # -------------------------------------------------

        entropy = compute_emotion_entropy(weighted_emotions)

        return EmotionDetectionResult(
            emotion_scores=weighted_emotions,
            dominant_emotion=dominant_emotion,
            emotion_entropy=entropy,
            emotion_intensity=intensity_score,
            manipulation_score=manipulation_score,
            manipulation_type=manipulation_type
        )


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    detector = EmotionDetector()

    text = """
    This shocking crisis has created massive fear and panic.
    The government has failed completely!
    Act now before it's too late!
    """

    result = detector.analyze(text)

    print(result)