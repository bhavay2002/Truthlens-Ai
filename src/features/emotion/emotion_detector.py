"""
File Name: emotion_detector.py
Module: Emotion Analysis

Description:
     emotion detection module for TruthLens AI.

    Provides:
        • lexicon-based emotion inference
        • emotion intensity estimation
        • negation-aware emotion scoring
        • batch processing
        • transformer-compatible feature vectors

"""

import logging
from collections import Counter
from typing import Dict, List, Optional, Iterable

import numpy as np
import spacy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Feature schema for deterministic ML input
# ---------------------------------------------------------

EMOTION_SCHEMA = [
    "emotion_anger",
    "emotion_fear",
    "emotion_joy",
    "emotion_sadness",
    "emotion_surprise",
    "emotion_disgust",
    "emotion_trust",
    "emotion_anticipation",
    "emotion_intensity",
]


# ---------------------------------------------------------
# Linguistic helpers
# ---------------------------------------------------------

NEGATIONS = {"not", "never", "no", "none"}

INTENSIFIERS = {
    "very",
    "extremely",
    "highly",
    "deeply",
    "incredibly",
    "strongly",
}


# ---------------------------------------------------------
# Emotion Detector
# ---------------------------------------------------------


class EmotionDetector:

    DEFAULT_EMOTIONS = [
        "anger",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "disgust",
        "trust",
        "anticipation",
    ]

    def __init__(
        self,
        emotion_lexicon: Optional[Dict[str, List[str]]] = None,
        spacy_model: str = "en_core_web_sm",
    ):

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy load failed")
            raise RuntimeError("spaCy initialization failed") from exc

        self.emotion_lexicon = self._normalize_lexicon(emotion_lexicon)

        logger.info("EmotionDetector initialized")

    # -----------------------------------------------------

    def _normalize_lexicon(
        self,
        emotion_lexicon: Optional[Dict[str, List[str]]],
    ) -> Dict[str, set]:

        normalized = {}

        if emotion_lexicon is None:
            for e in self.DEFAULT_EMOTIONS:
                normalized[e] = set()
            return normalized

        for emotion, words in emotion_lexicon.items():

            normalized[emotion.lower()] = {
                w.strip().lower()
                for w in words
                if isinstance(w, str) and w.strip()
            }

        return normalized

    # -----------------------------------------------------

    def detect(self, text: str) -> Dict[str, float]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        doc = self.nlp(text)

        tokens = [t.text.lower() for t in doc if t.is_alpha]

        emotion_counts = self._emotion_counts(tokens)

        distribution = self._normalize_distribution(emotion_counts)

        intensity = self._compute_intensity(tokens)

        result = {}

        for emotion in self.DEFAULT_EMOTIONS:
            result[f"emotion_{emotion}"] = distribution.get(emotion, 0.0)

        result["emotion_intensity"] = intensity

        return result

    # -----------------------------------------------------

    def detect_batch(
        self,
        texts: Iterable[str],
    ) -> List[Dict[str, float]]:

        results = []

        for doc in self.nlp.pipe(texts):

            tokens = [t.text.lower() for t in doc if t.is_alpha]

            emotion_counts = self._emotion_counts(tokens)

            distribution = self._normalize_distribution(emotion_counts)

            intensity = self._compute_intensity(tokens)

            record = {}

            for emotion in self.DEFAULT_EMOTIONS:
                record[f"emotion_{emotion}"] = distribution.get(emotion, 0.0)

            record["emotion_intensity"] = intensity

            results.append(record)

        return results

    # -----------------------------------------------------

    def _emotion_counts(self, tokens: List[str]) -> Counter:

        counts = Counter()

        for i, token in enumerate(tokens):

            modifier = 1.0

            if i > 0 and tokens[i - 1] in INTENSIFIERS:
                modifier = 1.5

            if i > 0 and tokens[i - 1] in NEGATIONS:
                modifier = -1.0

            for emotion, lexicon in self.emotion_lexicon.items():

                if token in lexicon:
                    counts[emotion] += modifier

        return counts

    # -----------------------------------------------------

    def _normalize_distribution(
        self,
        emotion_counts: Counter,
    ) -> Dict[str, float]:

        total = sum(abs(v) for v in emotion_counts.values())

        if total == 0:
            return {e: 0.0 for e in self.DEFAULT_EMOTIONS}

        distribution = {}

        for emotion in self.DEFAULT_EMOTIONS:
            distribution[emotion] = emotion_counts.get(emotion, 0) / total

        return distribution

    # -----------------------------------------------------

    def _compute_intensity(self, tokens: List[str]) -> float:

        if not tokens:
            return 0.0

        hits = 0

        for token in tokens:
            for lexicon in self.emotion_lexicon.values():

                if token in lexicon:
                    hits += 1
                    break

        return hits / len(tokens)


# ---------------------------------------------------------
# Vector Conversion
# ---------------------------------------------------------


def emotion_vector_to_numpy(
    emotion_dict: Dict[str, float]
) -> np.ndarray:

    vector = np.array(
        [emotion_dict.get(name, 0.0) for name in EMOTION_SCHEMA],
        dtype=np.float32,
    )

    return vector
