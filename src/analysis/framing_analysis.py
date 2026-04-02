"""
File Name: framing_analysis.py
Module: Narrative Analysis - Media Framing Detection
Description:
    Detects media framing strategies within text for the TruthLens AI system.
    The module analyzes linguistic indicators associated with common framing
    strategies studied in political communication and media analysis research.
    These include responsibility framing, economic framing, moral framing,
    human interest framing, and conflict framing.

    The extracted features help quantify how an issue is framed within a text,
    allowing downstream modules to model narrative bias, ideological messaging,
    and propaganda patterns.

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
    Frame feature dictionary and optional numerical vector
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
class FramingAnalysisConfig:
    """
    Configuration for FramingAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"


class FramingAnalyzer:
    """
    Detects common media framing strategies in political discourse.
    """

    CONFLICT_TERMS = {
        "conflict",
        "fight",
        "battle",
        "war",
        "clash",
        "attack",
        "oppose",
        "confront",
        "dispute",
        "rival",
    }

    ECONOMIC_TERMS = {
        "economy",
        "economic",
        "market",
        "jobs",
        "tax",
        "trade",
        "budget",
        "cost",
        "financial",
        "growth",
    }

    MORAL_TERMS = {
        "moral",
        "ethics",
        "values",
        "justice",
        "right",
        "wrong",
        "duty",
        "principle",
        "virtue",
    }

    HUMAN_INTEREST_TERMS = {
        "family",
        "children",
        "community",
        "people",
        "victim",
        "life",
        "story",
        "emotion",
        "suffering",
        "experience",
    }

    RESPONSIBILITY_TERMS = {
        "responsible",
        "blame",
        "accountable",
        "duty",
        "failure",
        "government",
        "policy",
        "decision",
        "authority",
    }

    def __init__(self, config: FramingAnalysisConfig | None = None) -> None:
        """
        Initialize NLP pipeline for framing analysis.

        Args:
            config: Optional configuration object.
        """

        self.config = config or FramingAnalysisConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "FramingAnalyzer initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze framing strategies in text.

        Args:
            text: Input text.

        Returns:
            Dictionary containing framing scores.
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

        features.update(self._frame_score(tokens, self.CONFLICT_TERMS, "frame_conflict_score"))
        features.update(self._frame_score(tokens, self.ECONOMIC_TERMS, "frame_economic_score"))
        features.update(self._frame_score(tokens, self.MORAL_TERMS, "frame_moral_score"))
        features.update(self._frame_score(tokens, self.HUMAN_INTEREST_TERMS, "frame_human_interest_score"))
        features.update(self._frame_score(tokens, self.RESPONSIBILITY_TERMS, "frame_responsibility_score"))

        logger.debug("Framing features computed")

        return features

    def _frame_score(
        self,
        tokens: List[str],
        lexicon: set,
        feature_name: str,
    ) -> Dict[str, float]:
        """
        Compute framing score based on lexicon frequency.

        Args:
            tokens: Tokenized text.
            lexicon: Lexicon of frame-specific terms.
            feature_name: Output feature name.

        Returns:
            Dictionary with frame score.
        """

        if not tokens:
            return {feature_name: 0.0}

        counts = Counter(tokens)

        frame_hits = sum(counts[token] for token in counts if token in lexicon)

        ratio = frame_hits / max(len(tokens), 1)

        return {feature_name: float(ratio)}


def framing_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert framing feature dictionary into numeric vector.

    Args:
        features: Frame features.

    Returns:
        NumPy vector.
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
            logger.warning("Non-numeric framing feature skipped: %s", key)

    if not numeric_values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(numeric_values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Framing vector conversion failed")
        raise RuntimeError("Failed to convert framing features") from exc