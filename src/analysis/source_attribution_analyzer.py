"""
File Name: source_attribution_analyzer.py
Module: Discourse Analysis - Source Attribution Detection
Description:
    Detects attribution patterns in text for the TruthLens AI system. The module
    analyzes how information sources are referenced, identifying expert
    attribution, anonymous source usage, and credibility indicators. These
    signals help determine whether claims are supported by identifiable sources
    or vague references, which is important for misinformation analysis.

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
    Source attribution feature dictionary and optional numerical vector
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SourceAttributionConfig:
    """
    Configuration for SourceAttributionAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"


class SourceAttributionAnalyzer:
    """
    Detects attribution patterns indicating who is cited as authority.
    """

    EXPERT_TERMS = {
        "expert",
        "experts",
        "analyst",
        "analysts",
        "researcher",
        "researchers",
        "scientist",
        "scientists",
        "professor",
        "economist",
        "official",
        "authority",
    }

    ANONYMOUS_TERMS = {
        "sources",
        "source",
        "insiders",
        "officials",
        "people",
        "critics",
        "observers",
    }

    CREDIBILITY_TERMS = {
        "report",
        "study",
        "data",
        "analysis",
        "evidence",
        "according",
        "statistics",
        "survey",
    }

    QUOTE_PATTERN = re.compile(r'"')

    def __init__(self, config: SourceAttributionConfig | None = None) -> None:
        """
        Initialize NLP pipeline.
        """

        self.config = config or SourceAttributionConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "SourceAttributionAnalyzer initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze source attribution patterns.

        Args:
            text: Input text.

        Returns:
            Dictionary containing attribution signals.
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

        features.update(self._term_ratio(tokens, self.EXPERT_TERMS, "expert_attribution_ratio"))
        features.update(self._term_ratio(tokens, self.ANONYMOUS_TERMS, "anonymous_source_ratio"))
        features.update(self._term_ratio(tokens, self.CREDIBILITY_TERMS, "credibility_indicator_ratio"))
        features.update(self._quote_ratio(text))

        return features

    def _term_ratio(
        self,
        tokens: List[str],
        lexicon: set,
        feature_name: str,
    ) -> Dict[str, float]:
        """
        Compute lexical attribution signal ratio.
        """

        if not tokens:
            return {feature_name: 0.0}

        count = sum(1 for token in tokens if token in lexicon)

        ratio = count / max(len(tokens), 1)

        return {feature_name: float(ratio)}

    def _quote_ratio(self, text: str) -> Dict[str, float]:
        """
        Detect quotation usage indicating attributed speech.
        """

        quotes = len(self.QUOTE_PATTERN.findall(text))

        length = max(len(text), 1)

        return {"quotation_ratio": float(quotes / length)}


def source_attribution_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert attribution features into numeric vector.
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
            logger.warning("Non-numeric attribution feature skipped: %s", key)

    if not values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Source attribution vector conversion failed")
        raise RuntimeError("Failed to convert attribution features") from exc