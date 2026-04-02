"""
File Name: ideological_language_detector.py
Module: Ideology Analysis - Ideological Language Detection
Description:
    Detects ideological language patterns in text for the TruthLens AI system.
    The module identifies lexical signals associated with common political
    ideology narratives such as liberty/freedom rhetoric, equality/social
    justice framing, traditionalist language, and anti-elite rhetoric.

    These features help strengthen ideology classification models by providing
    interpretable signals derived directly from discourse.

Dependencies:
    logging
    typing
    dataclasses
    collections
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Dictionary of ideological language signals and optional numerical vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IdeologicalLanguageConfig:
    """
    Configuration for IdeologicalLanguageDetector.
    """

    spacy_model: str = "en_core_web_sm"


class IdeologicalLanguageDetector:
    """
    Detects ideological language signals in political discourse.
    """

    LIBERTY_TERMS = {
        "liberty",
        "freedom",
        "rights",
        "individual",
        "independence",
        "free",
    }

    EQUALITY_TERMS = {
        "equality",
        "justice",
        "fairness",
        "equity",
        "inclusion",
        "diversity",
    }

    TRADITION_TERMS = {
        "tradition",
        "heritage",
        "values",
        "family",
        "nation",
        "culture",
    }

    ELITE_TERMS = {
        "elite",
        "establishment",
        "bureaucrats",
        "politicians",
        "powerful",
        "globalists",
    }

    def __init__(self, config: IdeologicalLanguageConfig | None = None) -> None:
        """
        Initialize NLP pipeline for ideological language detection.
        """

        self.config = config or IdeologicalLanguageConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "IdeologicalLanguageDetector initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze ideological language signals in text.

        Args:
            text: Input text.

        Returns:
            Dictionary containing ideology language metrics.
        """

        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("Input text must be non-empty")

        try:
            doc: Doc = self.nlp(text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        tokens: List[str] = [
            token.text.lower() for token in doc if token.is_alpha
        ]

        features: Dict[str, float] = {}

        features.update(self._term_ratio(tokens, self.LIBERTY_TERMS, "liberty_language_ratio"))
        features.update(self._term_ratio(tokens, self.EQUALITY_TERMS, "equality_language_ratio"))
        features.update(self._term_ratio(tokens, self.TRADITION_TERMS, "tradition_language_ratio"))
        features.update(self._term_ratio(tokens, self.ELITE_TERMS, "anti_elite_language_ratio"))

        logger.debug("Ideological language features computed")

        return features

    def _term_ratio(
        self,
        tokens: List[str],
        lexicon: set,
        feature_name: str,
    ) -> Dict[str, float]:
        """
        Compute ideological lexical ratio.
        """

        if not tokens:
            return {feature_name: 0.0}

        counts = Counter(tokens)

        hits = sum(counts[token] for token in counts if token in lexicon)

        ratio = hits / max(len(tokens), 1)

        return {feature_name: float(ratio)}


def ideological_language_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert ideological language features into numeric vector.
    """

    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary")

    if not features:
        raise ValueError("features must be a non-empty dictionary")

    values: List[float] = []

    for key, value in features.items():
        if isinstance(value, (int, float, np.number)):
            values.append(float(value))
        else:
            logger.warning("Non-numeric ideology feature skipped: %s", key)

    if not values:
        raise ValueError("No numeric ideology values found")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Ideological language vector conversion failed")
        raise RuntimeError("Failed to convert ideology features") from exc