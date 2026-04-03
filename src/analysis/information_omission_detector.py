"""
File Name: information_omission_detector.py
Module: Discourse Analysis - Information Omission Detection
Description:
    Detects advanced information omission patterns in text for the TruthLens AI
    system. This module extends basic context omission detection by identifying
    missing counterarguments, one-sided framing, and incomplete evidence chains.

    These signals help detect narratives where opposing viewpoints are absent,
    evidence is weak or incomplete, or arguments are presented without balanced
    perspectives. Such patterns are common in propaganda, misinformation, and
    highly biased discourse.

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
    Dictionary of information omission features and optional numerical vector
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
class InformationOmissionConfig:
    """
    Configuration for InformationOmissionDetector.
    """

    spacy_model: str = "en_core_web_sm"


class InformationOmissionDetector:
    """
    Detects advanced information omission signals in discourse.
    """

    COUNTERARGUMENT_MARKERS = {
        "however",
        "but",
        "although",
        "though",
        "nevertheless",
        "on the other hand",
        "yet",
    }

    EVIDENCE_MARKERS = {
        "evidence",
        "data",
        "study",
        "research",
        "report",
        "analysis",
        "statistics",
    }

    CLAIM_MARKERS = {
        "therefore",
        "thus",
        "clearly",
        "obviously",
        "shows",
        "proves",
        "demonstrates",
    }

    def __init__(self, config: InformationOmissionConfig | None = None) -> None:
        """
        Initialize NLP pipeline.
        """

        self.config = config or InformationOmissionConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "InformationOmissionDetector initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze omission patterns in text.

        Args:
            text: Input text.

        Returns:
            Dictionary of omission indicators.
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

        features.update(self._missing_counterarguments(tokens))
        features.update(self._one_sided_framing(tokens))
        features.update(self._evidence_chain_strength(tokens))

        logger.debug("Information omission features computed")

        return features

    def _missing_counterarguments(self, tokens: List[str]) -> Dict[str, float]:
        """
        Estimate likelihood of missing counterarguments.
        """

        if not tokens:
            return {"missing_counterargument_score": 0.0}

        counter_hits = sum(
            1 for token in tokens if token in self.COUNTERARGUMENT_MARKERS
        )

        score = 1.0 - (counter_hits / max(len(tokens), 1))

        return {"missing_counterargument_score": float(score)}

    def _one_sided_framing(self, tokens: List[str]) -> Dict[str, float]:
        """
        Detect strong claim language without balancing signals.
        """

        if not tokens:
            return {"one_sided_framing_score": 0.0}

        claim_hits = sum(
            1 for token in tokens if token in self.CLAIM_MARKERS
        )

        counter_hits = sum(
            1 for token in tokens if token in self.COUNTERARGUMENT_MARKERS
        )

        if claim_hits == 0:
            return {"one_sided_framing_score": 0.0}

        score = claim_hits / max(counter_hits + 1, 1)

        return {"one_sided_framing_score": float(score)}

    def _evidence_chain_strength(self, tokens: List[str]) -> Dict[str, float]:
        """
        Estimate completeness of evidence chains.
        """

        if not tokens:
            return {"incomplete_evidence_score": 0.0}

        counts = Counter(tokens)

        evidence_hits = sum(
            counts[token] for token in counts if token in self.EVIDENCE_MARKERS
        )

        ratio = evidence_hits / max(len(tokens), 1)

        score = 1.0 - ratio

        return {"incomplete_evidence_score": float(score)}


def information_omission_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert omission features into numeric vector.
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
            logger.warning("Non-numeric omission feature skipped: %s", key)

    if not values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Omission vector conversion failed")
        raise RuntimeError("Failed to convert omission features") from exc
