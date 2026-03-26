"""
File Name: bias_features.py
Module: Feature Engineering - Bias Detection

Description:
    Extracts interpretable linguistic signals associated with bias,
    persuasion, and opinionated language. These features complement
    transformer-based representations by providing transparent,
    explainable indicators useful for bias detection models in
    the TruthLens AI system.

    Designed for:
        • scalable feature pipelines
        • integration with multi-task NLP architectures

Dependencies:
    logging
    re
    typing
    collections
    numpy
    spacy
"""

import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Iterable

import numpy as np
import spacy
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Feature schema (deterministic order for ML models)
# -------------------------------------------------------------------------

FEATURE_SCHEMA: List[str] = [
    "bias_lexicon_count",
    "bias_lexicon_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "pronoun_ratio",
    "hedge_ratio",
    "modal_ratio",
    "intensifier_ratio",
    "dep_amod_ratio",
    "dep_advmod_ratio",
    "dep_nsubj_ratio",
    "dep_dobj_ratio",
    "exclamation_ratio",
    "question_ratio",
    "capital_word_ratio",
]


# -------------------------------------------------------------------------
# Linguistic lexicons
# -------------------------------------------------------------------------

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "likely", "seems", "appears",
    "suggests", "reportedly", "arguably", "apparently"
}

INTENSIFIERS = {
    "very", "extremely", "highly", "deeply", "strongly",
    "completely", "totally", "absolutely"
}

MODAL_VERBS = {
    "must", "should", "could", "might", "may", "would"
}


# -------------------------------------------------------------------------
# Bias Feature Extractor
# -------------------------------------------------------------------------


class BiasFeatureExtractor:
    """
    Extract interpretable bias-related linguistic signals.

    This module provides explainable signals that complement
    transformer-based bias detection models.
    """

    def __init__(
        self,
        bias_lexicon: Optional[List[str]] = None,
        spacy_model: str = "en_core_web_sm",
        disable_spacy_components: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize NLP pipeline and lexicons.

        Args:
            bias_lexicon:
                Optional list of bias-indicating words
            spacy_model:
                spaCy model name
            disable_spacy_components:
                Components to disable for performance
        """

        if bias_lexicon and not isinstance(bias_lexicon, list):
            raise ValueError("bias_lexicon must be a list")

        disable_spacy_components = disable_spacy_components or []

        try:
            self.nlp = spacy.load(spacy_model, disable=disable_spacy_components)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("spaCy initialization error") from exc

        self.bias_lexicon = set(bias_lexicon or [])

        logger.info("BiasFeatureExtractor initialized")

    # ---------------------------------------------------------------------

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract all bias-related features from a text document.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        doc = self._process_text(text)

        features: Dict[str, float] = {}

        features.update(self._lexicon_features(doc))
        features.update(self._sentiment_features(doc))
        features.update(self._subjectivity_features(doc))
        features.update(self._hedging_features(doc))
        features.update(self._syntactic_features(doc))
        features.update(self._punctuation_features(text))

        return features

    # ---------------------------------------------------------------------

    def extract_batch(self, texts: Iterable[str]) -> List[Dict[str, float]]:
        """
        Batch feature extraction for efficient large-scale processing.
        """

        results: List[Dict[str, float]] = []

        for doc in self.nlp.pipe(texts):
            features: Dict[str, float] = {}

            features.update(self._lexicon_features(doc))
            features.update(self._sentiment_features(doc))
            features.update(self._subjectivity_features(doc))
            features.update(self._hedging_features(doc))
            features.update(self._syntactic_features(doc))
            features.update(self._punctuation_features(doc.text))

            results.append(features)

        return results

    # ---------------------------------------------------------------------

    def _process_text(self, text: str) -> Doc:
        """Run spaCy pipeline with safe error handling."""

        try:
            return self.nlp(text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing error") from exc

    # ---------------------------------------------------------------------

    def _lexicon_features(self, doc: Doc) -> Dict[str, float]:

        tokens = [t.text.lower() for t in doc if t.is_alpha]
        token_count = len(tokens)

        if token_count == 0:
            return {
                "bias_lexicon_count": 0.0,
                "bias_lexicon_ratio": 0.0,
            }

        bias_count = sum(token in self.bias_lexicon for token in tokens)

        return {
            "bias_lexicon_count": float(bias_count),
            "bias_lexicon_ratio": float(bias_count / token_count),
        }

    # ---------------------------------------------------------------------

    def _sentiment_features(self, doc: Doc) -> Dict[str, float]:

        adjectives = [t for t in doc if t.pos_ == "ADJ"]
        adverbs = [t for t in doc if t.pos_ == "ADV"]

        length = max(len(doc), 1)

        return {
            "adjective_ratio": len(adjectives) / length,
            "adverb_ratio": len(adverbs) / length,
        }

    # ---------------------------------------------------------------------

    def _subjectivity_features(self, doc: Doc) -> Dict[str, float]:

        pronouns = [t for t in doc if t.pos_ == "PRON"]

        length = max(len(doc), 1)

        return {
            "pronoun_ratio": len(pronouns) / length,
        }

    # ---------------------------------------------------------------------

    def _hedging_features(self, doc: Doc) -> Dict[str, float]:

        tokens = [t.text.lower() for t in doc]

        length = max(len(tokens), 1)

        hedge_count = sum(t in HEDGE_WORDS for t in tokens)
        modal_count = sum(t in MODAL_VERBS for t in tokens)
        intensifier_count = sum(t in INTENSIFIERS for t in tokens)

        return {
            "hedge_ratio": hedge_count / length,
            "modal_ratio": modal_count / length,
            "intensifier_ratio": intensifier_count / length,
        }

    # ---------------------------------------------------------------------

    def _syntactic_features(self, doc: Doc) -> Dict[str, float]:

        dependency_counts = Counter(t.dep_ for t in doc)
        total_tokens = max(len(doc), 1)

        return {
            "dep_amod_ratio": dependency_counts.get("amod", 0) / total_tokens,
            "dep_advmod_ratio": dependency_counts.get("advmod", 0) / total_tokens,
            "dep_nsubj_ratio": dependency_counts.get("nsubj", 0) / total_tokens,
            "dep_dobj_ratio": dependency_counts.get("dobj", 0) / total_tokens,
        }

    # ---------------------------------------------------------------------

    def _punctuation_features(self, text: str) -> Dict[str, float]:

        exclamations = len(re.findall(r"!", text))
        questions = len(re.findall(r"\?", text))
        capital_words = len(re.findall(r"\b[A-Z]{2,}\b", text))

        text_length = max(len(text), 1)

        return {
            "exclamation_ratio": exclamations / text_length,
            "question_ratio": questions / text_length,
            "capital_word_ratio": capital_words / text_length,
        }


# -------------------------------------------------------------------------
# Feature Vector Utilities
# -------------------------------------------------------------------------


def normalize_feature_vector(feature_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert feature dictionary into normalized vector.

    Uses deterministic ordering defined by FEATURE_SCHEMA.
    """

    if not feature_dict:
        raise ValueError("feature_dict cannot be empty")

    vector = np.array(
        [feature_dict.get(name, 0.0) for name in FEATURE_SCHEMA],
        dtype=np.float32,
    )

    std = np.std(vector)

    if std == 0:
        return vector

    return (vector - np.mean(vector)) / std