"""
File Name: discourse_features.py
Module: Discourse Analysis - Feature Extraction
Description:
    Extracts discourse-level linguistic features used for narrative and bias
    analysis in the TruthLens AI system. The module identifies structural
    discourse signals such as sentence complexity, discourse markers,
    argumentative structures, cohesion indicators, and rhetorical patterns.
    These features complement transformer models by capturing document-level
    language organization and discourse dynamics.

Dependencies:
    logging
    re
    typing
    collections
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Dictionary of discourse-level features and numerical feature vector
"""

import logging
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import spacy


logger = logging.getLogger(__name__)


class DiscourseFeatureExtractor:
    """
    Extracts discourse structure and cohesion features from text.
    """

    DISCOURSE_MARKERS = {
        "however",
        "therefore",
        "thus",
        "moreover",
        "furthermore",
        "nevertheless",
        "meanwhile",
        "consequently",
        "instead",
        "otherwise",
        "although",
        "because",
        "since",
        "while",
    }

    CONTRAST_MARKERS = {
        "but",
        "however",
        "although",
        "though",
        "yet",
        "nevertheless",
    }

    CAUSAL_MARKERS = {
        "because",
        "since",
        "therefore",
        "thus",
        "consequently",
        "hence",
    }

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize discourse feature extractor."""

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        logger.info("DiscourseFeatureExtractor initialized")

    def extract(self, text: str) -> Dict[str, float]:
        """Extract discourse-related linguistic features."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        try:
            doc = self.nlp(text)
        except Exception as exc:
            logger.exception("spaCy text processing failed")
            raise RuntimeError("Text processing failed") from exc

        tokens = [token.text.lower() for token in doc if token.is_alpha]

        features: Dict[str, float] = {}

        features.update(self._sentence_structure(doc))
        features.update(self._discourse_marker_features(tokens))
        features.update(self._cohesion_features(tokens))
        features.update(self._syntactic_complexity(doc))
        features.update(self._punctuation_structure(text))

        return features

    def _sentence_structure(self, doc) -> Dict[str, float]:
        """Measure sentence-level structural properties."""

        sentences = list(doc.sents)

        if not sentences:
            return {
                "discourse_sentence_count": 0.0,
                "discourse_avg_sentence_length": 0.0,
            }

        lengths = [len(sentence) for sentence in sentences]

        return {
            "discourse_sentence_count": float(len(sentences)),
            "discourse_avg_sentence_length": float(np.mean(lengths)),
        }

    def _discourse_marker_features(self, tokens: List[str]) -> Dict[str, float]:
        """Detect discourse marker usage in text."""

        if not tokens:
            return {"discourse_marker_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.DISCOURSE_MARKERS)

        ratio = count / max(len(tokens), 1)

        contrast_count = sum(1 for token in tokens if token in self.CONTRAST_MARKERS)
        causal_count = sum(1 for token in tokens if token in self.CAUSAL_MARKERS)

        return {
            "discourse_marker_ratio": float(ratio),
            "discourse_contrast_ratio": float(contrast_count / max(len(tokens), 1)),
            "discourse_causal_ratio": float(causal_count / max(len(tokens), 1)),
        }

    def _cohesion_features(self, tokens: List[str]) -> Dict[str, float]:
        """Measure lexical cohesion through word repetition."""

        if not tokens:
            return {"discourse_lexical_cohesion": 0.0}

        counts = Counter(tokens)

        repeated = sum(1 for _, count in counts.items() if count > 1)

        ratio = repeated / max(len(tokens), 1)

        return {"discourse_lexical_cohesion": float(ratio)}

    def _syntactic_complexity(self, doc) -> Dict[str, float]:
        """Estimate syntactic complexity using dependency structure."""

        dep_counts = Counter(token.dep_ for token in doc)

        total = max(len(doc), 1)

        return {
            "discourse_clause_ratio": float(dep_counts.get("ccomp", 0) / total),
            "discourse_modifier_ratio": float(dep_counts.get("amod", 0) / total),
            "discourse_adverbial_ratio": float(dep_counts.get("advmod", 0) / total),
        }

    def _punctuation_structure(self, text: str) -> Dict[str, float]:
        """Capture structural punctuation patterns."""

        commas = len(re.findall(r",", text))
        semicolons = len(re.findall(r";", text))
        colons = len(re.findall(r":", text))

        length = max(len(text), 1)

        return {
            "discourse_comma_ratio": float(commas / length),
            "discourse_semicolon_ratio": float(semicolons / length),
            "discourse_colon_ratio": float(colons / length),
        }


def discourse_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert discourse feature dictionary into numeric vector."""

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = np.array(list(features.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Discourse vector conversion failed")
        raise RuntimeError("Failed to convert discourse features") from exc