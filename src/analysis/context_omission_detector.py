"""
File Name: context_omission_detector.py
Module: Discourse Analysis - Context Omission Detection
Description:
    Detects potential context omission patterns in text for the TruthLens AI
    system. The module analyzes linguistic signals that often indicate that
    important contextual information may be missing, simplified, or selectively
    presented. It examines discourse cues such as vague references, missing
    attribution, limited evidence markers, and abrupt claims without supporting
    context.

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
    Dictionary of context omission features and optional numerical vector
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ContextOmissionConfig:
    """
    Configuration for ContextOmissionDetector.
    """

    spacy_model: str = "en_core_web_sm"
    normalize_ratios: bool = True


class ContextOmissionDetector:
    """
    Detects linguistic patterns associated with missing or incomplete context.
    """

    VAGUE_REFERENCES = {
        "they",
        "people",
        "many",
        "some",
        "others",
        "experts",
        "critics",
        "sources",
        "analysts",
    }

    ATTRIBUTION_MARKERS = {
        "according",
        "reported",
        "stated",
        "claimed",
        "said",
        "noted",
        "explained",
    }

    EVIDENCE_MARKERS = {
        "data",
        "study",
        "report",
        "research",
        "analysis",
        "evidence",
        "statistics",
    }

    QUOTE_PATTERN = re.compile(r'"')

    def __init__(self, config: ContextOmissionConfig | None = None) -> None:
        """
        Initialize NLP pipeline for context omission detection.

        Args:
            config: Optional configuration for detector.
        """

        self.config = config or ContextOmissionConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "ContextOmissionDetector initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze text for signals of missing contextual information.

        Args:
            text: Input text.

        Returns:
            Dictionary containing context omission features.
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

        features.update(self._vague_reference_features(tokens))
        features.update(self._attribution_features(tokens))
        features.update(self._evidence_features(tokens))
        features.update(self._quote_features(cleaned_text))
        features.update(self._entity_context_features(doc))

        logger.debug("Context omission features computed")

        return features

    def _vague_reference_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Measure frequency of vague references.

        Args:
            tokens: Tokenized words.

        Returns:
            Feature dictionary.
        """

        if not tokens:
            return {"context_vague_reference_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.VAGUE_REFERENCES)

        ratio = count / len(tokens)

        return {"context_vague_reference_ratio": float(ratio)}

    def _attribution_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Detect attribution signals referencing external sources.

        Args:
            tokens: Tokenized words.

        Returns:
            Feature dictionary.
        """

        if not tokens:
            return {"context_attribution_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.ATTRIBUTION_MARKERS)

        ratio = count / len(tokens)

        return {"context_attribution_ratio": float(ratio)}

    def _evidence_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Measure presence of evidence or research references.

        Args:
            tokens: Tokenized words.

        Returns:
            Feature dictionary.
        """

        if not tokens:
            return {"context_evidence_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.EVIDENCE_MARKERS)

        ratio = count / len(tokens)

        return {"context_evidence_ratio": float(ratio)}

    def _quote_features(self, text: str) -> Dict[str, float]:
        """
        Detect quotation usage indicating cited statements.

        Args:
            text: Input text.

        Returns:
            Feature dictionary.
        """

        quote_count = len(self.QUOTE_PATTERN.findall(text))
        length = max(len(text), 1)

        ratio = quote_count / length

        return {"context_quote_ratio": float(ratio)}

    def _entity_context_features(self, doc: Doc) -> Dict[str, float]:
        """
        Measure named entity presence as contextual grounding.

        Args:
            doc: spaCy document.

        Returns:
            Feature dictionary.
        """

        entities = list(doc.ents)

        total_tokens = max(len(doc), 1)

        entity_ratio = len(entities) / total_tokens

        entity_types = Counter(ent.label_ for ent in entities)

        diversity = len(entity_types)

        return {
            "context_entity_ratio": float(entity_ratio),
            "context_entity_type_diversity": float(diversity),
        }


def context_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert context omission features into a numerical vector.

    Args:
        features: Dictionary of context features.

    Returns:
        NumPy feature vector.
    """

    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary")

    if not features:
        raise ValueError("features dictionary cannot be empty")

    numeric_values: List[float] = []

    for key, value in features.items():
        if isinstance(value, (int, float, np.number)):
            numeric_values.append(float(value))
        else:
            logger.warning("Non-numeric feature skipped: %s", key)

    if not numeric_values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(numeric_values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Context feature vector conversion failed")
        raise RuntimeError("Failed to convert context features") from exc