from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# FEATURES
# =========================================================

@dataclass(slots=True)
class TemporalGraphFeatures:

    entity_recurrence: float
    entity_transition_rate: float
    topic_shift_score: float
    narrative_drift: float

    # 🔥 NEW
    temporal_entropy: float
    narrative_volatility: float
    temporal_consistency: float

    def to_dict(self) -> Dict[str, float]:
        # ``@dataclass(slots=True)`` strips ``__dict__``; use the
        # generated ``__slots__`` to build the mapping instead.
        return {f: getattr(self, f) for f in self.__slots__}


# =========================================================
# ANALYZER
# =========================================================

class TemporalGraphAnalyzer:

    def __init__(self, min_token_length: int = 4):
        if min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")

        self.min_token_length = min_token_length
        logger.info("TemporalGraphAnalyzer initialized")

    # =====================================================
    # HELPERS
    # =====================================================

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_entities(self, sentence: str) -> Set[str]:
        tokens = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
        return {t for t in tokens if len(t) >= self.min_token_length}

    # =====================================================
    # MAIN
    # =====================================================

    def analyze(self, text: str) -> TemporalGraphFeatures:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text")

        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            return TemporalGraphFeatures(*([0.0] * 7))

        entity_sets: List[Set[str]] = [
            self._extract_entities(s) for s in sentences
        ]

        # =================================================
        # ENTITY RECURRENCE
        # =================================================
        counter = Counter()
        for s in entity_sets:
            counter.update(s)

        recurrence = sum(1 for c in counter.values() if c > 1)
        entity_recurrence = float(recurrence / (len(counter) + EPS))

        # =================================================
        # TRANSITION RATE (WITH TEMPORAL WEIGHTING 🔥)
        # =================================================
        transitions = []
        weights = []

        for i in range(len(entity_sets) - 1):

            A, B = entity_sets[i], entity_sets[i + 1]

            if not A:
                continue

            overlap = len(A & B) / (len(A) + EPS)

            # 🔥 recency weight
            w = (i + 1) / len(entity_sets)

            transitions.append(overlap * w)
            weights.append(w)

        entity_transition_rate = float(
            np.sum(transitions) / (np.sum(weights) + EPS)
        )

        # =================================================
        # TOPIC SHIFT (JACCARD DISTANCE)
        # =================================================
        shifts = []

        for i in range(len(entity_sets) - 1):

            A, B = entity_sets[i], entity_sets[i + 1]

            union = A | B
            if not union:
                shifts.append(0.0)
                continue

            sim = len(A & B) / len(union)
            shifts.append(1.0 - sim)

        topic_shift_score = float(np.mean(shifts)) if shifts else 0.0

        # =================================================
        # DRIFT (SMOOTHED 🔥)
        # =================================================
        centroid = Counter()
        for s in entity_sets:
            centroid.update(s)

        centroid_set = set(centroid.keys())

        drift_vals = []

        for s in entity_sets:
            union = centroid_set | s
            if not union:
                drift_vals.append(0.0)
                continue

            sim = len(centroid_set & s) / len(union)
            drift_vals.append(1.0 - sim)

        # 🔥 smoothing
        narrative_drift = float(np.mean(drift_vals))

        # =================================================
        # 🔥 NEW METRICS
        # =================================================

        # 1. temporal entropy
        shift_arr = np.array(shifts, dtype=float)
        if shift_arr.size > 0:
            p = shift_arr / (np.sum(shift_arr) + EPS)
            temporal_entropy = float(-np.sum(p * np.log(p + EPS)))
        else:
            temporal_entropy = 0.0

        # 2. volatility (variance of shifts)
        narrative_volatility = float(np.var(shift_arr)) if shift_arr.size else 0.0

        # 3. consistency (inverse volatility)
        #
        # G-E2 fix: variance of a single value is always 0.0, so the
        # naive ``1 - var`` formula reported a 2-sentence document
        # (one transition) as "perfectly consistent" — same value as a
        # 50-sentence document with genuinely zero shift variance.
        # Meanwhile a 1-sentence document took the early-return path
        # and reported ``0.0``. Pick the consistent convention: at
        # least 2 transitions (i.e. ≥3 sentences) are needed before
        # the metric is meaningful — below that we report ``0.0``
        # ("insufficient data") to match the 1-sentence branch.
        if shift_arr.size >= 2:
            temporal_consistency = float(1.0 - narrative_volatility)
        else:
            temporal_consistency = 0.0

        return TemporalGraphFeatures(
            entity_recurrence=entity_recurrence,
            entity_transition_rate=entity_transition_rate,
            topic_shift_score=topic_shift_score,
            narrative_drift=narrative_drift,
            temporal_entropy=temporal_entropy,
            narrative_volatility=narrative_volatility,
            temporal_consistency=temporal_consistency,
        )


# =========================================================
# VECTOR
# =========================================================

def temporal_graph_vector(features: Dict[str, float]) -> np.ndarray:

    keys = (
        "entity_recurrence",
        "entity_transition_rate",
        "topic_shift_score",
        "narrative_drift",
        "temporal_entropy",
        "narrative_volatility",
        "temporal_consistency",
    )

    return np.array(
        [float(features.get(k, 0.0)) for k in keys],
        dtype=np.float32,
    )