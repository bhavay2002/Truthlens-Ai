"""
File Name: emotion_patterns.py
Module: Emotion Analysis - Pattern Detection

Description:
    Extracts rhetorical and discourse-level emotion pattern signals.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

import numpy as np


PATTERN_SCHEMA = [
    "emotion_contrast_ratio",
    "emotion_escalation_ratio",
    "emotion_negation_ratio",
    "emotion_repetition_ratio",
    "emotion_exclamation_ratio",
    "emotion_question_ratio",
    "emotion_ellipsis_ratio",
    "emotion_sentence_variability",
]

CONTRAST_MARKERS = {
    "but",
    "however",
    "yet",
    "although",
    "though",
    "whereas",
    "while",
}

ESCALATION_MARKERS = {
    "increasingly",
    "more",
    "most",
    "intensifying",
    "escalating",
    "worse",
    "worst",
    "growing",
}

NEGATION_MARKERS = {"not", "never", "no", "none", "without", "cannot"}

_TOKEN_REGEX = re.compile(r"\b[a-z']+\b")
_SENTENCE_SPLIT_REGEX = re.compile(r"[.!?]+")
_ELLIPSIS_REGEX = re.compile(r"\.\.\.+")


class EmotionPatternDetector:
    """
    Detect rhetorical emotion patterns that are useful for feature engineering.
    """

    def detect_patterns(self, text: str) -> Dict[str, float]:
        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

        tokens = _TOKEN_REGEX.findall(text.lower())
        if not tokens:
            return self._empty_features()

        token_count = float(len(tokens))
        text_length = float(max(len(text), 1))

        contrast_count = sum(token in CONTRAST_MARKERS for token in tokens)
        escalation_count = sum(token in ESCALATION_MARKERS for token in tokens)
        negation_count = sum(token in NEGATION_MARKERS for token in tokens)

        counts = Counter(tokens)
        repeated_tokens = sum(count - 1 for count in counts.values() if count > 1)

        exclamation_count = text.count("!")
        question_count = text.count("?")
        ellipsis_count = len(_ELLIPSIS_REGEX.findall(text))

        sentence_variability = self._sentence_variability(text)

        return {
            "emotion_contrast_ratio": round(contrast_count / token_count, 4),
            "emotion_escalation_ratio": round(escalation_count / token_count, 4),
            "emotion_negation_ratio": round(negation_count / token_count, 4),
            "emotion_repetition_ratio": round(repeated_tokens / token_count, 4),
            "emotion_exclamation_ratio": round(exclamation_count / text_length, 4),
            "emotion_question_ratio": round(question_count / text_length, 4),
            "emotion_ellipsis_ratio": round(ellipsis_count / text_length, 4),
            "emotion_sentence_variability": round(sentence_variability, 4),
        }

    def _sentence_variability(self, text: str) -> float:
        sentences: List[str] = [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_REGEX.split(text)
            if sentence.strip()
        ]

        if len(sentences) <= 1:
            return 0.0

        lengths = [
            len(_TOKEN_REGEX.findall(sentence.lower()))
            for sentence in sentences
        ]

        if not lengths:
            return 0.0

        avg_length = float(np.mean(lengths))
        if avg_length == 0.0:
            return 0.0

        return float(np.std(lengths) / avg_length)

    def _empty_features(self) -> Dict[str, float]:
        return {name: 0.0 for name in PATTERN_SCHEMA}
