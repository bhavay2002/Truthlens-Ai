"""
File Name: information_density_analyzer.py
Module: Discourse Analysis - Information Density Analysis
Description:
    Measures informational versus rhetorical density in text for the TruthLens AI
    system. The module estimates how much of a document consists of factual
    statements, opinionated language, claims, and rhetorical signals.

    These signals help differentiate factual reporting from opinion-driven or
    rhetorically persuasive writing. The extracted metrics support bias
    detection, propaganda analysis, and discourse modeling.

Dependencies:
    logging
    typing
    dataclasses
    collections
    numpy
    spacy
    re

Inputs:
    Raw text string

Outputs:
    Information density feature dictionary and optional numerical vector
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InformationDensityConfig:
    """
    Configuration for InformationDensityAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"


class InformationDensityAnalyzer:
    """
    Measures factual vs rhetorical information density within text.
    """

    FACTUAL_TERMS = {
        "data",
        "report",
        "study",
        "research",
        "statistics",
        "analysis",
        "according",
        "evidence",
        "survey",
        "official",
    }

    OPINION_TERMS = {
        "believe",
        "think",
        "argue",
        "claim",
        "suggest",
        "feel",
        "likely",
        "possibly",
        "perhaps",
        "opinion",
    }

    CLAIM_TERMS = {
        "therefore",
        "thus",
        "hence",
        "consequently",
        "so",
        "clearly",
        "obviously",
    }

    RHETORICAL_TERMS = {
        "outrageous",
        "shocking",
        "dangerous",
        "disaster",
        "catastrophe",
        "crisis",
        "threat",
        "corrupt",
        "evil",
    }

    RHETORICAL_PATTERN = re.compile(r"[!?]+")

    def __init__(self, config: InformationDensityConfig | None = None) -> None:
        """
        Initialize NLP pipeline for information density analysis.
        """

        self.config = config or InformationDensityConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "InformationDensityAnalyzer initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze factual and rhetorical density in text.

        Args:
            text: Input text.

        Returns:
            Dictionary of information density metrics.
        """

        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("Input text must be non-empty")

        try:
            doc: Doc = self.nlp(cleaned_text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        tokens: List[str] = [
            token.text.lower() for token in doc if token.is_alpha
        ]

        features: Dict[str, float] = {}

        features.update(self._term_ratio(tokens, self.FACTUAL_TERMS, "factual_density"))
        features.update(self._term_ratio(tokens, self.OPINION_TERMS, "opinion_density"))
        features.update(self._term_ratio(tokens, self.CLAIM_TERMS, "claim_density"))
        features.update(self._term_ratio(tokens, self.RHETORICAL_TERMS, "rhetorical_density"))

        features.update(self._punctuation_rhetoric(cleaned_text))

        logger.debug("Information density features computed")

        return features

    def _term_ratio(
        self,
        tokens: List[str],
        lexicon: set,
        feature_name: str,
    ) -> Dict[str, float]:
        """
        Compute lexical density ratio.
        """

        if not tokens:
            return {feature_name: 0.0}

        counts = Counter(tokens)

        hits = sum(counts[token] for token in counts if token in lexicon)

        ratio = hits / max(len(tokens), 1)

        return {feature_name: float(ratio)}

    def _punctuation_rhetoric(self, text: str) -> Dict[str, float]:
        """
        Capture rhetorical punctuation intensity.
        """

        matches = self.RHETORICAL_PATTERN.findall(text)

        length = max(len(text.split()), 1)

        score = len(matches) / length

        return {"rhetorical_punctuation_density": float(score)}


def information_density_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert information density features into numeric vector.
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
            logger.warning("Non-numeric information density feature skipped: %s", key)

    if not values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Information density vector conversion failed")
        raise RuntimeError("Failed to convert information density features") from exc