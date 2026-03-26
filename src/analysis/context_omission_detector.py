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

import logging
import re
from collections import Counter
from typing import Dict, List

import numpy as np
import spacy


logger = logging.getLogger(__name__)


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

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize NLP pipeline for context omission detection."""

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        logger.info("ContextOmissionDetector initialized")

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze text for signals of missing contextual information."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        try:
            doc = self.nlp(text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        tokens = [token.text.lower() for token in doc if token.is_alpha]

        features: Dict[str, float] = {}

        features.update(self._vague_reference_features(tokens))
        features.update(self._attribution_features(tokens))
        features.update(self._evidence_features(tokens))
        features.update(self._quote_features(text))
        features.update(self._entity_context_features(doc))

        return features

    def _vague_reference_features(self, tokens: List[str]) -> Dict[str, float]:
        """Measure frequency of vague references."""

        if not tokens:
            return {"context_vague_reference_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.VAGUE_REFERENCES)

        ratio = count / max(len(tokens), 1)

        return {"context_vague_reference_ratio": float(ratio)}

    def _attribution_features(self, tokens: List[str]) -> Dict[str, float]:
        """Detect attribution signals referencing external sources."""

        if not tokens:
            return {"context_attribution_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.ATTRIBUTION_MARKERS)

        ratio = count / max(len(tokens), 1)

        return {"context_attribution_ratio": float(ratio)}

    def _evidence_features(self, tokens: List[str]) -> Dict[str, float]:
        """Measure presence of evidence or research references."""

        if not tokens:
            return {"context_evidence_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.EVIDENCE_MARKERS)

        ratio = count / max(len(tokens), 1)

        return {"context_evidence_ratio": float(ratio)}

    def _quote_features(self, text: str) -> Dict[str, float]:
        """Detect quotation usage indicating cited statements."""

        quotes = len(re.findall(r'"', text))

        length = max(len(text), 1)

        return {"context_quote_ratio": float(quotes / length)}

    def _entity_context_features(self, doc) -> Dict[str, float]:
        """Measure named entity presence as contextual grounding."""

        entities = [ent.text for ent in doc.ents]

        total_tokens = max(len(doc), 1)

        entity_ratio = len(entities) / total_tokens

        entity_types = Counter(ent.label_ for ent in doc.ents)

        diversity = len(entity_types)

        return {
            "context_entity_ratio": float(entity_ratio),
            "context_entity_type_diversity": float(diversity),
        }


def context_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert context omission features into a numerical vector."""

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = np.array(list(features.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Context feature vector conversion failed")
        raise RuntimeError("Failed to convert context features") from exc