"""
TruthLens AI
Emotion Intensity Detection Module

Measures the strength of emotional expression in text.

Signals Used
------------
• ALL CAPS emphasis
• Exclamation punctuation
• Intensifier adverbs
• Repeated punctuation patterns

Outputs
-------
intensity_score
signal_breakdown
sentence_intensity
highlighted_tokens
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

# ---------------------------------------------------------
# Intensifier Lexicon
# ---------------------------------------------------------

INTENSIFIER_ADVERBS = {
    "very",
    "extremely",
    "incredibly",
    "highly",
    "deeply",
    "absolutely",
    "totally",
    "completely",
    "remarkably",
    "terribly",
    "really",
    "so",
}


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

WEIGHTS = {
    "caps": 0.35,
    "exclamation": 0.30,
    "intensifier": 0.20,
    "repeated_punctuation": 0.15,
}


# ---------------------------------------------------------
# Data Structures
# ---------------------------------------------------------


@dataclass
class EmotionIntensityResult:
    intensity_score: float
    signal_breakdown: Dict[str, float]
    sentence_intensity: List[Dict]
    highlighted_tokens: List[Dict]


# ---------------------------------------------------------
# Text Processing Utilities
# ---------------------------------------------------------


def tokenize_words(text: str) -> List[str]:

    return re.findall(r"\b\w+\b", text)


def tokenize_sentences(text: str) -> List[str]:

    sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# Signal Detection
# ---------------------------------------------------------


def detect_caps_words(tokens: List[str]) -> int:

    return sum(1 for token in tokens if token.isupper() and len(token) > 2)


def detect_exclamations(text: str) -> int:

    return text.count("!")


def detect_intensifiers(tokens: List[str]) -> List[int]:

    positions = []

    for idx, token in enumerate(tokens):

        if token.lower() in INTENSIFIER_ADVERBS:
            positions.append(idx)

    return positions


def detect_repeated_punctuation(text: str) -> int:

    patterns = re.findall(r"[!?]{2,}", text)

    return len(patterns)


# ---------------------------------------------------------
# Sentence-Level Intensity
# ---------------------------------------------------------


def compute_sentence_intensity(sentence: str) -> float:

    tokens = tokenize_words(sentence)

    if not tokens:
        return 0.0

    caps = detect_caps_words(tokens)
    intensifiers = len(detect_intensifiers(tokens))
    exclamations = sentence.count("!")

    score = caps * 0.4 + intensifiers * 0.35 + exclamations * 0.25

    return round(score / len(tokens), 4)


# ---------------------------------------------------------
# Emotion Intensity Analyzer
# ---------------------------------------------------------


class EmotionIntensityAnalyzer:
    """
    Detects emotional amplification signals in text.
    """

    def analyze(self, text: str) -> EmotionIntensityResult:

        tokens = tokenize_words(text)

        if not tokens:

            return EmotionIntensityResult(
                intensity_score=0.0,
                signal_breakdown={},
                sentence_intensity=[],
                highlighted_tokens=[],
            )

        total_words = len(tokens)

        # Signal counts
        caps_count = detect_caps_words(tokens)
        exclamation_count = detect_exclamations(text)
        repeated_punct = detect_repeated_punctuation(text)
        intensifier_positions = detect_intensifiers(tokens)

        intensifier_count = len(intensifier_positions)

        # Ratios
        caps_ratio = caps_count / total_words
        exclamation_ratio = exclamation_count / total_words
        intensifier_ratio = intensifier_count / total_words
        repeated_ratio = repeated_punct / total_words

        # Weighted intensity score
        intensity_score = (
            WEIGHTS["caps"] * caps_ratio
            + WEIGHTS["exclamation"] * exclamation_ratio
            + WEIGHTS["intensifier"] * intensifier_ratio
            + WEIGHTS["repeated_punctuation"] * repeated_ratio
        )

        intensity_score = round(intensity_score, 4)

        # Sentence-level analysis
        sentences = tokenize_sentences(text)

        sentence_scores = []

        for sentence in sentences:

            score = compute_sentence_intensity(sentence)

            sentence_scores.append({"sentence": sentence, "intensity": score})

        # Token highlights
        highlights = []

        for pos in intensifier_positions:

            highlights.append(
                {"token": tokens[pos], "type": "intensifier", "position": pos}
            )

        signal_breakdown = {
            "caps_ratio": round(caps_ratio, 4),
            "exclamation_ratio": round(exclamation_ratio, 4),
            "intensifier_ratio": round(intensifier_ratio, 4),
            "repeated_punctuation_ratio": round(repeated_ratio, 4),
        }

        return EmotionIntensityResult(
            intensity_score=intensity_score,
            signal_breakdown=signal_breakdown,
            sentence_intensity=sentence_scores,
            highlighted_tokens=highlights,
        )


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    analyzer = EmotionIntensityAnalyzer()

    text = """
    THIS is absolutely shocking!!!
    The government has made an incredibly terrible decision!!!
    """

    result = analyzer.analyze(text)

    print(result)
