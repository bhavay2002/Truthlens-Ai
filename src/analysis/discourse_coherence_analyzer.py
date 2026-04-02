"""
File Name: discourse_coherence_analyzer.py
Module: Discourse Analysis - Coherence Measurement
Description:
    Measures discourse coherence and logical flow of arguments for the TruthLens
    AI system. The module analyzes sentence-to-sentence semantic similarity,
    narrative continuity across segments, and the presence of discourse
    transition markers.

    These signals help identify coherent argument structures versus chaotic,
    manipulative, or poorly connected discourse patterns often found in
    propaganda, misinformation, or low-quality argumentative text.

Dependencies:
    logging
    typing
    dataclasses
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Dictionary of discourse coherence features and optional numerical vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DiscourseCoherenceConfig:
    """
    Configuration for DiscourseCoherenceAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"


class DiscourseCoherenceAnalyzer:
    """
    Analyzes logical flow and coherence of discourse.
    """

    TRANSITION_MARKERS = {
        "however",
        "therefore",
        "thus",
        "meanwhile",
        "furthermore",
        "moreover",
        "in contrast",
        "on the other hand",
        "consequently",
        "nevertheless",
    }

    def __init__(self, config: DiscourseCoherenceConfig | None = None) -> None:
        """
        Initialize NLP pipeline.
        """

        self.config = config or DiscourseCoherenceConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "DiscourseCoherenceAnalyzer initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze discourse coherence in text.

        Args:
            text: Input text.

        Returns:
            Dictionary containing discourse coherence metrics.
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

        sentences = [sent for sent in doc.sents]

        features: Dict[str, float] = {}

        features.update(self._sentence_coherence(sentences))
        features.update(self._narrative_continuity(doc))
        features.update(self._transition_usage(doc))

        logger.debug("Discourse coherence features computed")

        return features

    def _sentence_coherence(self, sentences: List) -> Dict[str, float]:
        """
        Measure semantic similarity between adjacent sentences.
        """

        if len(sentences) < 2:
            return {"sentence_coherence": 0.0}

        similarities = []

        for i in range(len(sentences) - 1):
            sim = sentences[i].similarity(sentences[i + 1])
            similarities.append(sim)

        score = float(np.mean(similarities)) if similarities else 0.0

        return {"sentence_coherence": score}

    def _narrative_continuity(self, doc: Doc) -> Dict[str, float]:
        """
        Estimate continuity using entity repetition across sentences.
        """

        entities = [ent.text.lower() for ent in doc.ents]

        if not entities:
            return {"narrative_continuity": 0.0}

        unique_entities = set(entities)

        continuity = 1.0 - (len(unique_entities) / max(len(entities), 1))

        return {"narrative_continuity": float(continuity)}

    def _transition_usage(self, doc: Doc) -> Dict[str, float]:
        """
        Measure usage of discourse transition markers.
        """

        tokens = [token.text.lower() for token in doc if token.is_alpha]

        if not tokens:
            return {"discourse_transition_ratio": 0.0}

        count = sum(1 for token in tokens if token in self.TRANSITION_MARKERS)

        ratio = count / max(len(tokens), 1)

        return {"discourse_transition_ratio": float(ratio)}


def discourse_coherence_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert discourse coherence features into numeric vector.
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
            logger.warning("Non-numeric coherence feature skipped: %s", key)

    if not values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Discourse coherence vector conversion failed")
        raise RuntimeError("Failed to convert coherence features") from exc