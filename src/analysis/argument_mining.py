"""
File Name: argument_mining.py
Module: Discourse Analysis - Argument Mining
Description:
    Implements argument mining utilities for the TruthLens AI system. The module
    extracts argumentation structures from text by identifying claims, premises,
    supporting statements, and argumentative discourse markers. These features
    support higher-level narrative and propaganda analysis by modeling how
    arguments are constructed within text.

Dependencies:
    logging
    typing
    collections
    dataclasses
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Argumentation feature dictionary and optional numerical vector
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
class ArgumentMiningConfig:
    """
    Configuration for ArgumentMiningAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"


class ArgumentMiningAnalyzer:
    """
    Extracts argument structure signals from text.
    """

    CLAIM_MARKERS = {
        "therefore",
        "thus",
        "hence",
        "consequently",
        "so",
        "clearly",
        "obviously",
    }

    PREMISE_MARKERS = {
        "because",
        "since",
        "given",
        "as",
        "considering",
        "due",
    }

    SUPPORT_MARKERS = {
        "for example",
        "for instance",
        "evidence",
        "demonstrates",
        "shows",
    }

    CONTRAST_MARKERS = {
        "however",
        "but",
        "although",
        "though",
        "yet",
        "nevertheless",
    }

    def __init__(self, config: ArgumentMiningConfig | None = None) -> None:
        """
        Initialize NLP pipeline used for argument mining.

        Args:
            config: Optional configuration object.
        """

        self.config = config or ArgumentMiningConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "ArgumentMiningAnalyzer initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze argumentative structures in text.

        Args:
            text: Input text.

        Returns:
            Dictionary containing argument mining features.
        """

        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("Input text must be a non-empty string")

        try:
            doc: Doc = self.nlp(cleaned_text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        tokens: List[str] = [
            token.text.lower() for token in doc if token.is_alpha
        ]

        features: Dict[str, float] = {}

        features.update(self._claim_features(tokens))
        features.update(self._premise_features(tokens))
        features.update(self._support_features(cleaned_text))
        features.update(self._contrast_features(tokens))
        features.update(self._argument_density(doc))

        logger.debug("Argument mining features computed")

        return features

    def _claim_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Detect claim-related discourse markers.
        """

        if not tokens:
            return {"argument_claim_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.CLAIM_MARKERS)

        ratio = count / len(tokens)

        return {"argument_claim_ratio": float(ratio)}

    def _premise_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Detect premise indicators supporting arguments.
        """

        if not tokens:
            return {"argument_premise_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.PREMISE_MARKERS)

        ratio = count / len(tokens)

        return {"argument_premise_ratio": float(ratio)}

    def _support_features(self, text: str) -> Dict[str, float]:
        """
        Detect supporting evidence patterns.
        """

        text_lower = text.lower()

        support_hits = sum(
            1 for marker in self.SUPPORT_MARKERS if marker in text_lower
        )

        length = max(len(text.split()), 1)

        return {"argument_support_ratio": float(support_hits / length)}

    def _contrast_features(self, tokens: List[str]) -> Dict[str, float]:
        """
        Detect counterargument or contrast signals.
        """

        if not tokens:
            return {"argument_contrast_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.CONTRAST_MARKERS)

        ratio = count / len(tokens)

        return {"argument_contrast_ratio": float(ratio)}

    def _argument_density(self, doc: Doc) -> Dict[str, float]:
        """
        Estimate argument density using verbs and clauses.
        """

        verbs = [token for token in doc if token.pos_ == "VERB"]
        clauses = [token for token in doc if token.dep_ in {"ccomp", "xcomp"}]

        total_tokens = max(len(doc), 1)

        return {
            "argument_verb_density": float(len(verbs) / total_tokens),
            "argument_clause_density": float(len(clauses) / total_tokens),
        }


def argument_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert argument features dictionary into numeric vector.
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
            logger.warning("Non-numeric argument feature skipped: %s", key)

    if not numeric_values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(numeric_values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Argument feature vector conversion failed")
        raise RuntimeError("Failed to convert argument features") from exc