"""
File Name: narrative_conflict.py
Module: Narrative Analysis - Conflict Detection
Description:
    Detects narrative conflict structures within text for the TruthLens AI system.
    The module analyzes linguistic signals that indicate opposing actors,
    ideological clashes, threat framing, and adversarial narrative structures.
    These signals help identify conflict-driven narratives frequently used in
    propaganda, political messaging, and ideological discourse.

Dependencies:
    logging
    typing
    collections
    numpy
    spacy
    re

Inputs:
    Raw text string

Outputs:
    Narrative conflict feature dictionary and optional numerical vector
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
class NarrativeConflictConfig:
    """
    Configuration for NarrativeConflictAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"
    normalize_ratios: bool = True


class NarrativeConflictAnalyzer:
    """
    Detects adversarial narrative structures and conflict framing patterns.
    """

    CONFLICT_TERMS = {
        "conflict",
        "fight",
        "battle",
        "attack",
        "war",
        "struggle",
        "crisis",
        "threat",
        "enemy",
        "clash",
        "oppose",
        "confront",
    }

    OPPOSITION_MARKERS = {
        "versus",
        "against",
        "vs",
        "oppose",
        "opposed",
        "conflict",
    }

    POLARIZATION_TERMS = {
        "us",
        "them",
        "they",
        "others",
        "enemy",
        "opponent",
    }

    EXCLAMATION_PATTERN = re.compile(r"!")
    QUESTION_PATTERN = re.compile(r"\?")

    def __init__(self, config: NarrativeConflictConfig | None = None) -> None:
        """
        Initialize NLP pipeline used for narrative conflict analysis.

        Args:
            config: Optional configuration object.
        """

        self.config = config or NarrativeConflictConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "NarrativeConflictAnalyzer initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze text for narrative conflict signals.

        Args:
            text: Input text.

        Returns:
            Dictionary containing narrative conflict features.
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

        features.update(self._conflict_term_features(tokens))
        features.update(self._opposition_features(tokens))
        features.update(self._polarization_features(tokens))
        features.update(self._entity_opposition(doc))
        features.update(self._punctuation_conflict(cleaned_text))

        logger.debug("Narrative conflict features computed")

        return features

    def _conflict_term_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Measure frequency of conflict-related terms.
        """

        if not tokens:
            return {"narrative_conflict_term_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.CONFLICT_TERMS)

        ratio = count / len(tokens)

        return {"narrative_conflict_term_ratio": float(ratio)}

    def _opposition_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Detect explicit opposition framing.
        """

        if not tokens:
            return {"narrative_opposition_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.OPPOSITION_MARKERS)

        ratio = count / len(tokens)

        return {"narrative_opposition_ratio": float(ratio)}

    def _polarization_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Measure polarized group framing.
        """

        if not tokens:
            return {"narrative_polarization_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.POLARIZATION_TERMS)

        ratio = count / len(tokens)

        return {"narrative_polarization_ratio": float(ratio)}

    def _entity_opposition(self, doc: Doc) -> Dict[str, float]:
        """
        Estimate potential conflict between named entities.
        """

        entities = [ent.text.lower() for ent in doc.ents]

        if not entities:
            return {"narrative_entity_conflict_ratio": 0.0}

        entity_counts = Counter(entities)

        repeated_entities = sum(
            1 for _, count in entity_counts.items() if count > 1
        )

        ratio = repeated_entities / max(len(entities), 1)

        return {"narrative_entity_conflict_ratio": float(ratio)}

    def _punctuation_conflict(self, text: str) -> Dict[str, float]:
        """
        Capture punctuation emphasis associated with conflict rhetoric.
        """

        exclamations = len(self.EXCLAMATION_PATTERN.findall(text))
        questions = len(self.QUESTION_PATTERN.findall(text))

        length = max(len(text), 1)

        return {
            "narrative_conflict_exclamation_ratio": float(exclamations / length),
            "narrative_conflict_question_ratio": float(questions / length),
        }


def narrative_conflict_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert narrative conflict features into numeric vector.
    """

    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary")

    if not features:
        raise ValueError("features must be a non-empty dictionary")

    numeric_values: List[float] = []

    for key, value in features.items():
        if isinstance(value, (int, float, np.number)):
            numeric_values.append(float(value))
        else:
            logger.warning("Non-numeric narrative conflict feature skipped: %s", key)

    if not numeric_values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(numeric_values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Narrative conflict vector conversion failed")
        raise RuntimeError(
            "Failed to convert narrative conflict features"
        ) from exc