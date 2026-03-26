"""
File Name: emotion_lexicon.py
Module: Emotion Analysis - Lexicon Scoring

Description:
    Lightweight lexicon-based emotion analyzer used by feature and API layers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set


EMOTION_ORDER = [
    "anger",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "disgust",
    "trust",
    "anticipation",
]

NEGATIONS = {"not", "never", "no", "none", "without"}
INTENSIFIERS = {
    "very",
    "extremely",
    "really",
    "highly",
    "absolutely",
    "deeply",
    "incredibly",
    "totally",
}

DEFAULT_NRC_LEXICON: Dict[str, Set[str]] = {
    "anger": {"angry", "rage", "outrage", "furious", "hostile"},
    "fear": {"fear", "afraid", "panic", "threat", "terrified"},
    "joy": {"joy", "happy", "delight", "celebrate", "pleased"},
    "sadness": {"sad", "grief", "mourn", "tragic", "depressed"},
    "surprise": {"surprised", "shocking", "unexpected", "sudden"},
    "disgust": {"disgust", "revolting", "repulsive", "nauseating"},
    "trust": {"trust", "reliable", "credible", "honest", "faith"},
    "anticipation": {"expect", "await", "forecast", "upcoming", "anticipate"},
}

_TOKEN_REGEX = re.compile(r"\b[a-z']+\b")


@dataclass(frozen=True)
class EmotionLexiconResult:
    dominant_emotion: str
    emotion_scores: Dict[str, float]
    emotion_distribution: Dict[str, float]


class EmotionLexiconAnalyzer:
    """
    Lexicon-based emotion analyzer with simple negation/intensifier handling.
    """

    def __init__(
        self,
        emotion_lexicon: Optional[Dict[str, Iterable[str]]] = None,
    ) -> None:
        self.emotion_lexicon = self._normalize_lexicon(emotion_lexicon)

    def _normalize_lexicon(
        self,
        emotion_lexicon: Optional[Dict[str, Iterable[str]]],
    ) -> Dict[str, Set[str]]:
        if emotion_lexicon is None:
            return {
                emotion: set(words)
                for emotion, words in DEFAULT_NRC_LEXICON.items()
            }

        normalized: Dict[str, Set[str]] = {emotion: set() for emotion in EMOTION_ORDER}

        for emotion, words in emotion_lexicon.items():
            if not isinstance(emotion, str):
                continue

            emotion_name = emotion.strip().lower()
            if not emotion_name:
                continue

            if emotion_name not in normalized:
                normalized[emotion_name] = set()

            if not isinstance(words, Iterable):
                continue

            normalized[emotion_name].update(
                token.strip().lower()
                for token in words
                if isinstance(token, str) and token.strip()
            )

        return normalized

    def analyze(self, text: str) -> EmotionLexiconResult:
        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

        tokens = _TOKEN_REGEX.findall(text.lower())

        base_emotions = list(dict.fromkeys([*EMOTION_ORDER, *self.emotion_lexicon.keys()]))

        if not tokens:
            empty = {emotion: 0.0 for emotion in base_emotions}
            return EmotionLexiconResult(
                dominant_emotion="neutral",
                emotion_scores=empty,
                emotion_distribution=empty.copy(),
            )

        raw_scores: Dict[str, float] = {emotion: 0.0 for emotion in base_emotions}

        for index, token in enumerate(tokens):
            previous = tokens[index - 1] if index > 0 else ""
            modifier = 1.0

            if previous in INTENSIFIERS:
                modifier = 1.5

            if previous in NEGATIONS:
                modifier *= -1.0

            for emotion in base_emotions:
                lexicon = self.emotion_lexicon.get(emotion, set())
                if token in lexicon:
                    raw_scores[emotion] += modifier

        token_count = max(len(tokens), 1)
        emotion_scores = {
            emotion: round(score / token_count, 4)
            for emotion, score in raw_scores.items()
        }

        total_signal = sum(abs(score) for score in raw_scores.values())

        if total_signal == 0:
            emotion_distribution = {
                emotion: 0.0 for emotion in base_emotions
            }
            dominant_emotion = "neutral"
        else:
            emotion_distribution = {
                emotion: round(abs(raw_scores[emotion]) / total_signal, 4)
                for emotion in base_emotions
            }
            dominant_emotion = max(
                emotion_distribution,
                key=emotion_distribution.get,
            )

        return EmotionLexiconResult(
            dominant_emotion=dominant_emotion,
            emotion_scores=emotion_scores,
            emotion_distribution=emotion_distribution,
        )

    def analyze_batch(self, texts: Iterable[str]) -> List[EmotionLexiconResult]:
        return [self.analyze(text) for text in texts]
