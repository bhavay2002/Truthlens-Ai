"""
File Name: rhetorical_device_detector.py
Module: Discourse Analysis - Rhetorical Device Detection
Description:
    Detects rhetorical persuasion techniques in text for the TruthLens AI system.
    The module identifies linguistic signals commonly associated with persuasive
    rhetoric used in propaganda, political messaging, and biased discourse.

    The detector focuses on rhetorical patterns including exaggeration,
    loaded language, emotional appeal, fear appeal, scapegoating, and
    false dilemmas. These features help quantify persuasive intensity and
    manipulation strategies present in text.

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
    Rhetorical feature dictionary and optional numerical vector
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RhetoricalDeviceConfig:
    """
    Configuration for RhetoricalDeviceDetector.
    """

    spacy_model: str = "en_core_web_sm"


class RhetoricalDeviceDetector:
    """
    Detects rhetorical persuasion techniques used in discourse.
    """

    EXAGGERATION_TERMS = {
        "always",
        "never",
        "everyone",
        "no one",
        "completely",
        "totally",
        "absolutely",
        "disaster",
        "catastrophe",
    }

    LOADED_LANGUAGE_TERMS = {
        "corrupt",
        "traitor",
        "radical",
        "extreme",
        "dangerous",
        "evil",
        "outrageous",
        "shocking",
        "disgrace",
    }

    EMOTIONAL_APPEAL_TERMS = {
        "heartbreaking",
        "tragic",
        "devastating",
        "hope",
        "fear",
        "anger",
        "pain",
        "suffering",
    }

    FEAR_APPEAL_TERMS = {
        "threat",
        "danger",
        "risk",
        "crisis",
        "attack",
        "collapse",
        "terror",
        "fear",
    }

    SCAPEGOAT_PATTERNS = {
        "they are responsible",
        "they caused",
        "blame them",
        "their fault",
    }

    FALSE_DILEMMA_PATTERNS = {
        "either",
        "or else",
        "no alternative",
        "only choice",
        "nothing else",
    }

    def __init__(self, config: RhetoricalDeviceConfig | None = None) -> None:
        """
        Initialize NLP pipeline for rhetorical analysis.

        Args:
            config: Optional configuration.
        """

        self.config = config or RhetoricalDeviceConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "RhetoricalDeviceDetector initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze rhetorical persuasion techniques in text.

        Args:
            text: Input text.

        Returns:
            Dictionary of rhetorical device features.
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

        features.update(self._term_ratio(tokens, self.EXAGGERATION_TERMS, "rhetoric_exaggeration_score"))
        features.update(self._term_ratio(tokens, self.LOADED_LANGUAGE_TERMS, "rhetoric_loaded_language_score"))
        features.update(self._term_ratio(tokens, self.EMOTIONAL_APPEAL_TERMS, "rhetoric_emotional_appeal_score"))
        features.update(self._term_ratio(tokens, self.FEAR_APPEAL_TERMS, "rhetoric_fear_appeal_score"))

        features.update(self._pattern_score(cleaned_text, self.SCAPEGOAT_PATTERNS, "rhetoric_scapegoating_score"))
        features.update(self._pattern_score(cleaned_text, self.FALSE_DILEMMA_PATTERNS, "rhetoric_false_dilemma_score"))

        return features

    def _term_ratio(
        self,
        tokens: List[str],
        lexicon: set,
        feature_name: str,
    ) -> Dict[str, float]:
        """
        Compute lexical rhetorical signal ratio.
        """

        if not tokens:
            return {feature_name: 0.0}

        counts = Counter(tokens)

        hits = sum(counts[token] for token in counts if token in lexicon)

        ratio = hits / max(len(tokens), 1)

        return {feature_name: float(ratio)}

    def _pattern_score(
        self,
        text: str,
        patterns: set,
        feature_name: str,
    ) -> Dict[str, float]:
        """
        Detect phrase-level rhetorical patterns.
        """

        text_lower = text.lower()

        hits = sum(1 for pattern in patterns if pattern in text_lower)

        length = max(len(text.split()), 1)

        score = hits / length

        return {feature_name: float(score)}


def rhetorical_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert rhetorical features into numeric vector.
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
            logger.warning("Non-numeric rhetorical feature skipped: %s", key)

    if not values:
        raise ValueError("No numeric rhetorical values found")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Rhetorical vector conversion failed")
        raise RuntimeError("Failed to convert rhetorical features") from exc
