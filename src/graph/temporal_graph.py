"""
File Name: temporal_graph.py
Module: Graph Analysis - Temporal Narrative Graph
Description:
    Implements temporal narrative analysis utilities for the TruthLens AI
    system. The module models narrative evolution across sentences and
    computes temporal graph signals that capture entity recurrence,
    transition dynamics, topic drift, and narrative volatility. These
    features are useful for detecting misinformation patterns, narrative
    manipulation, and discourse instability in news articles.

Dependencies:
    logging
    typing
    dataclasses
    collections
    re
    numpy

Inputs:
    Raw article text

Outputs:
    Temporal narrative feature dictionary and numerical vector
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TemporalGraphFeatures:
    """
    Structured container for temporal narrative features.
    """

    entity_recurrence: float
    entity_transition_rate: float
    topic_shift_score: float
    narrative_drift: float

    def to_dict(self) -> Dict[str, float]:
        """Convert feature dataclass to dictionary."""
        return {
            "entity_recurrence": self.entity_recurrence,
            "entity_transition_rate": self.entity_transition_rate,
            "topic_shift_score": self.topic_shift_score,
            "narrative_drift": self.narrative_drift,
        }


class TemporalGraphAnalyzer:
    """
    Analyzes narrative evolution across sentences.
    """

    def __init__(self, min_token_length: int = 4) -> None:
        if min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")

        self.min_token_length = min_token_length

        logger.info(
            "TemporalGraphAnalyzer initialized (min_token_length=%d)",
            min_token_length,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_entities(self, sentence: str) -> Set[str]:
        """
        Extract simple token entities (lightweight fallback
        when full NER is not used).
        """

        tokens = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())

        entities = {
            token
            for token in tokens
            if len(token) >= self.min_token_length
        }

        return entities

    def analyze(self, text: str) -> TemporalGraphFeatures:
        """
        Compute temporal narrative features from text.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            return TemporalGraphFeatures(0.0, 0.0, 0.0, 0.0)

        entity_sets: List[Set[str]] = [
            self._extract_entities(sentence)
            for sentence in sentences
        ]

        # -------------------------------------------------
        # Entity recurrence
        # -------------------------------------------------
        entity_counter = Counter()

        for entity_set in entity_sets:
            entity_counter.update(entity_set)

        repeated_entities = [
            count for count in entity_counter.values() if count > 1
        ]

        entity_recurrence = (
            float(len(repeated_entities) / max(len(entity_counter), 1))
        )

        # -------------------------------------------------
        # Entity transition rate
        # -------------------------------------------------
        transitions = 0
        comparisons = 0

        for i in range(len(entity_sets) - 1):

            current_entities = entity_sets[i]
            next_entities = entity_sets[i + 1]

            if not current_entities:
                continue

            overlap = len(current_entities.intersection(next_entities))

            transitions += overlap
            comparisons += len(current_entities)

        entity_transition_rate = (
            float(transitions / max(comparisons, 1))
        )

        # -------------------------------------------------
        # Topic shift score (Jaccard distance)
        # -------------------------------------------------
        shift_scores = []

        for i in range(len(entity_sets) - 1):

            A = entity_sets[i]
            B = entity_sets[i + 1]

            union = A.union(B)

            if not union:
                shift_scores.append(0.0)
                continue

            jaccard_similarity = len(A.intersection(B)) / len(union)

            shift_scores.append(1.0 - jaccard_similarity)

        topic_shift_score = float(np.mean(shift_scores)) if shift_scores else 0.0

        # -------------------------------------------------
        # Narrative drift
        # -------------------------------------------------
        centroid_vector = Counter()

        for entity_set in entity_sets:
            centroid_vector.update(entity_set)

        centroid_set = set(centroid_vector.keys())

        drift_scores = []

        for entity_set in entity_sets:

            union = centroid_set.union(entity_set)

            if not union:
                drift_scores.append(0.0)
                continue

            similarity = len(centroid_set.intersection(entity_set)) / len(union)

            drift_scores.append(1.0 - similarity)

        narrative_drift = float(np.mean(drift_scores)) if drift_scores else 0.0

        features = TemporalGraphFeatures(
            entity_recurrence=float(entity_recurrence),
            entity_transition_rate=float(entity_transition_rate),
            topic_shift_score=float(topic_shift_score),
            narrative_drift=float(narrative_drift),
        )

        logger.debug("Temporal narrative features computed: %s", features)

        return features


def temporal_graph_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert temporal graph features into numerical vector.
    """

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    ordered_keys = (
        "entity_recurrence",
        "entity_transition_rate",
        "topic_shift_score",
        "narrative_drift",
    )

    return np.array(
        [float(features.get(key, 0.0)) for key in ordered_keys],
        dtype=np.float32,
    )