"""
File Name: narrative_propagation.py
Module: Narrative Analysis - Propagation Dynamics
Description:
    Analyzes how narrative signals propagate within a piece of text. The module
    estimates narrative spread, reinforcement, and persistence by tracking
    repeated narrative frames, thematic continuity, and cross-sentence narrative
    reinforcement. These signals help the TruthLens AI system identify how
    strongly a narrative is being pushed and whether it is repeatedly reinforced
    throughout the discourse.

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
    Narrative propagation feature dictionary and optional numerical vector
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
class NarrativePropagationConfig:
    """
    Configuration for NarrativePropagationAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"


class NarrativePropagationAnalyzer:
    """
    Detects narrative reinforcement and propagation patterns across text.
    """

    NARRATIVE_KEY_TERMS = {
        "crisis",
        "threat",
        "freedom",
        "corruption",
        "security",
        "control",
        "attack",
        "defend",
        "protect",
        "enemy",
        "power",
        "rights",
    }

    def __init__(self, config: NarrativePropagationConfig | None = None) -> None:
        """
        Initialize NLP pipeline for narrative propagation analysis.

        Args:
            config: Optional configuration object.
        """

        self.config = config or NarrativePropagationConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "NarrativePropagationAnalyzer initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze narrative propagation patterns in text.

        Args:
            text: Input text.

        Returns:
            Dictionary containing narrative propagation features.
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

        sentences: List[str] = [
            sent.text.strip() for sent in doc.sents if sent.text.strip()
        ]

        tokens: List[str] = [
            token.text.lower() for token in doc if token.is_alpha
        ]

        features: Dict[str, float] = {}

        features.update(self._keyword_propagation(tokens))
        features.update(self._sentence_reinforcement(sentences))
        features.update(self._theme_persistence(tokens))
        features.update(self._narrative_focus(tokens))

        logger.debug("Narrative propagation features computed")

        return features

    def _keyword_propagation(self, tokens: List[str]) -> Dict[str, float]:
        """
        Measure repetition of narrative keywords.
        """

        if not tokens:
            return {"narrative_keyword_propagation": 0.0}

        counts = Counter(tokens)

        keyword_hits = sum(
            counts[token] for token in counts if token in self.NARRATIVE_KEY_TERMS
        )

        propagation_ratio = keyword_hits / len(tokens)

        return {"narrative_keyword_propagation": float(propagation_ratio)}

    def _sentence_reinforcement(self, sentences: List[str]) -> Dict[str, float]:
        """
        Detect repeated narrative structures across sentences.
        """

        if not sentences:
            return {"narrative_sentence_reinforcement": 0.0}

        normalized = [sentence.lower() for sentence in sentences]

        counts = Counter(normalized)

        repeated = sum(1 for _, count in counts.items() if count > 1)

        ratio = repeated / len(sentences)

        return {"narrative_sentence_reinforcement": float(ratio)}

    def _theme_persistence(self, tokens: List[str]) -> Dict[str, float]:
        """
        Estimate how persistent narrative themes remain across the text.
        """

        if not tokens:
            return {"narrative_theme_persistence": 0.0}

        counts = Counter(tokens)

        frequent_terms = [term for term, count in counts.items() if count > 2]

        persistence = len(frequent_terms) / len(tokens)

        return {"narrative_theme_persistence": float(persistence)}

    def _narrative_focus(self, tokens: List[str]) -> Dict[str, float]:
        """
        Estimate how concentrated the narrative is around a few key terms.
        """

        if not tokens:
            return {"narrative_focus_score": 0.0}

        counts = Counter(tokens)

        most_common = counts.most_common(5)

        total_hits = sum(count for _, count in most_common)

        focus_score = total_hits / len(tokens)

        return {"narrative_focus_score": float(focus_score)}


def narrative_propagation_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert narrative propagation features into a numerical vector.

    Args:
        features: Dictionary of propagation features.

    Returns:
        NumPy feature vector.
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
            logger.warning(
                "Non-numeric narrative propagation feature skipped: %s", key
            )

    if not numeric_values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(numeric_values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Narrative propagation vector conversion failed")
        raise RuntimeError(
            "Failed to convert propagation features"
        ) from exc