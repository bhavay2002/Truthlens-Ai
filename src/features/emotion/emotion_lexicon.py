"""
File Name: emotion_intensity.py
Module: Emotion Analysis - Intensity Estimation

Description:
    Emotional intensity estimator used in TruthLens AI.
    Extracts interpretable signals indicating emotional strength and
    rhetorical amplification within text.

"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Iterable

import numpy as np
import spacy
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Feature schema (deterministic ordering)
# ---------------------------------------------------------

INTENSITY_SCHEMA = [
    "emotion_lexicon_intensity",
    "intensifier_ratio",
    "exclamation_intensity",
    "question_intensity",
    "ellipsis_intensity",
    "capitalization_intensity",
    "adjective_amplification",
    "adverb_amplification",
    "repetition_intensity",
]


NEGATIONS = {"not", "never", "no", "none"}

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


class EmotionIntensityEstimator:

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

    # -----------------------------------------------------

    def _normalize_lexicon(
        self,
        emotion_lexicon: Optional[Dict[str, List[str]]],
    ) -> Dict[str, set]:

        normalized = {}

        if emotion_lexicon is None:
            return normalized

        for emotion, words in emotion_lexicon.items():

            normalized[emotion] = {
                w.strip().lower()
                for w in words
                if isinstance(w, str)
            }

        return normalized

    # -----------------------------------------------------

    def estimate(self, text: str) -> Dict[str, float]:

        doc = self.nlp(text)

        tokens = [t.text.lower() for t in doc if t.is_alpha]

        features = {}

        features.update(self._lexicon_intensity(tokens))
        features.update(self._intensifier_features(tokens))
        features.update(self._punctuation_features(text))
        features.update(self._capitalization_features(text))
        features.update(self._syntactic_features(doc))
        features.update(self._repetition_features(tokens))

        return features

    # -----------------------------------------------------

    def estimate_batch(
        self,
        texts: Iterable[str],
    ) -> List[Dict[str, float]]:

        results = []

        for doc in self.nlp.pipe(texts):

            tokens = [t.text.lower() for t in doc if t.is_alpha]

            features = {}

            features.update(self._lexicon_intensity(tokens))
            features.update(self._intensifier_features(tokens))
            features.update(self._punctuation_features(doc.text))
            features.update(self._capitalization_features(doc.text))
            features.update(self._syntactic_features(doc))
            features.update(self._repetition_features(tokens))

            results.append(features)

        return results

    # -----------------------------------------------------

    def _lexicon_intensity(self, tokens: List[str]) -> Dict[str, float]:

        if not tokens or not self.emotion_lexicon:
            return {"emotion_lexicon_intensity": 0.0}

        hits = 0

        for token in tokens:
            for lexicon in self.emotion_lexicon.values():

                if token in lexicon:
                    hits += 1
                    break

        return {"emotion_lexicon_intensity": hits / len(tokens)}

    # -----------------------------------------------------

    def _intensifier_features(self, tokens: List[str]):

        count = sum(token in INTENSIFIERS for token in tokens)

        return {"intensifier_ratio": count / max(len(tokens), 1)}

    # -----------------------------------------------------

    def _punctuation_features(self, text: str):

        exclam = len(re.findall(r"!", text))
        quest = len(re.findall(r"\?", text))
        ellip = len(re.findall(r"\.\.\.", text))

        length = max(len(text), 1)

        return {
            "exclamation_intensity": exclam / length,
            "question_intensity": quest / length,
            "ellipsis_intensity": ellip / length,
        }

    # -----------------------------------------------------

    def _capitalization_features(self, text: str):

        caps = re.findall(r"\b[A-Z]{2,}\b", text)

        return {
            "capitalization_intensity": len(caps) / max(len(text.split()), 1)
        }

    # -----------------------------------------------------

    def _syntactic_features(self, doc: Doc):

        adjectives = [t for t in doc if t.pos_ == "ADJ"]
        adverbs = [t for t in doc if t.pos_ == "ADV"]

        length = max(len(doc), 1)

        return {
            "adjective_amplification": len(adjectives) / length,
            "adverb_amplification": len(adverbs) / length,
        }

    # -----------------------------------------------------

    def _repetition_features(self, tokens: List[str]):

        counts = Counter(tokens)

        repeated = sum(1 for token, c in counts.items() if c > 1)

        return {"repetition_intensity": repeated / max(len(tokens), 1)}


# ---------------------------------------------------------
# Vector Conversion
# ---------------------------------------------------------


def emotion_intensity_vector(features: Dict[str, float]) -> np.ndarray:

    return np.array(
        [features.get(name, 0.0) for name in INTENSITY_SCHEMA],
        dtype=np.float32,
    )