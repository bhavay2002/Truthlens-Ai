"""
File Name: narrative_temporal_analyzer.py
Module: Narrative Analysis - Temporal Narrative Structure
Description:
    Analyzes temporal narrative structure within text for the TruthLens AI
    system. The module detects linguistic signals related to past framing,
    crisis escalation, and urgency language. These features help identify
    narratives that attempt to create panic, urgency, or historical framing
    to influence interpretation of events.

    Temporal narrative signals are particularly important in propaganda,
    crisis reporting, and political messaging where urgency or escalation
    framing is used to shape audience perception.

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
    Dictionary of temporal narrative features and optional numerical vector
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
class NarrativeTemporalConfig:
    """
    Configuration for NarrativeTemporalAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"


class NarrativeTemporalAnalyzer:
    """
    Detects temporal narrative patterns such as past framing,
    crisis escalation language, and urgency signals.
    """

    PAST_TERMS = {
        "previously",
        "earlier",
        "historically",
        "once",
        "before",
        "formerly",
        "past",
        "years",
        "decades",
    }

    CRISIS_TERMS = {
        "crisis",
        "collapse",
        "disaster",
        "catastrophe",
        "breakdown",
        "emergency",
        "meltdown",
        "chaos",
    }

    URGENCY_TERMS = {
        "immediately",
        "urgent",
        "now",
        "rapidly",
        "quickly",
        "instantly",
        "critical",
        "pressing",
    }

    def __init__(self, config: NarrativeTemporalConfig | None = None) -> None:
        """
        Initialize NLP pipeline for temporal narrative analysis.
        """

        self.config = config or NarrativeTemporalConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "NarrativeTemporalAnalyzer initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze temporal narrative signals in text.

        Args:
            text: Input text.

        Returns:
            Dictionary containing temporal narrative features.
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

        features.update(self._term_ratio(tokens, self.PAST_TERMS, "past_framing_ratio"))
        features.update(self._term_ratio(tokens, self.CRISIS_TERMS, "crisis_escalation_ratio"))
        features.update(self._term_ratio(tokens, self.URGENCY_TERMS, "urgency_language_ratio"))

        logger.debug("Temporal narrative features computed")

        return features

    def _term_ratio(
        self,
        tokens: List[str],
        lexicon: set,
        feature_name: str,
    ) -> Dict[str, float]:
        """
        Compute ratio of temporal narrative terms.
        """

        if not tokens:
            return {feature_name: 0.0}

        counts = Counter(tokens)

        hits = sum(counts[token] for token in counts if token in lexicon)

        ratio = hits / max(len(tokens), 1)

        return {feature_name: float(ratio)}


def narrative_temporal_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert temporal narrative features into numeric vector.
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
            logger.warning("Non-numeric temporal feature skipped: %s", key)

    if not values:
        raise ValueError("No numeric temporal values found")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Temporal vector conversion failed")
        raise RuntimeError("Failed to convert temporal features") from exc