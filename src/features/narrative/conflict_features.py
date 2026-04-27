# src/features/conflict_features.py

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import normalized_entropy
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Lexicons
# ---------------------------------------------------------

CONFRONTATION_TERMS = {...}
DISPUTE_TERMS = {...}
ACCUSATION_TERMS = {...}
AGGRESSIVE_LANGUAGE = {...}
POLARIZATION_TERMS = {...}
ESCALATION_TERMS = {...}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class ConflictFeatures(BaseFeature):

    name: str = "conflict_features"
    group: str = "conflict"
    description: str = "Normalized conflict discourse features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        n = len(tokens)

        if n == 0:
            return {}

        counter = Counter(tokens)

        def ratio(lexicon: Set[str]) -> float:
            return sum(counter.get(w, 0) for w in lexicon) / (n + EPS)

        raw = {
            "confrontation": ratio(CONFRONTATION_TERMS),
            "dispute": ratio(DISPUTE_TERMS),
            "accusation": ratio(ACCUSATION_TERMS),
            "aggression": ratio(AGGRESSIVE_LANGUAGE),
            "polarization": ratio(POLARIZATION_TERMS),
            "escalation": ratio(ESCALATION_TERMS),
        }

        # -------------------------
        # NORMALIZED DISTRIBUTION
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = float(values.sum())

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -------------------------
        # INTENSITY (STRONGER)
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY
        # -------------------------

        probs = np.array(list(dist.values()), dtype=np.float32)

        entropy = normalized_entropy(probs)

        # -------------------------
        # DIVERSITY (weighted)
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # RHETORIC (FIXED)
        # -------------------------

        exclam = text.count("!")
        questions = text.count("?")

        rhetoric = (exclam + questions) / (n + EPS)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "conflict_confrontation": self._safe(dist["confrontation"]),
            "conflict_dispute": self._safe(dist["dispute"]),
            "conflict_accusation": self._safe(dist["accusation"]),
            "conflict_aggression": self._safe(dist["aggression"]),
            "conflict_polarization": self._safe(dist["polarization"]),
            "conflict_escalation": self._safe(dist["escalation"]),

            "conflict_intensity": self._safe(intensity),
            "conflict_entropy": self._safe(entropy),
            "conflict_diversity": self._safe(diversity),

            "conflict_rhetoric_score": self._safe(rhetoric),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))