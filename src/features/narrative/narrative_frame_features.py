# src/features/narrative_frame_features.py

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

CONFLICT_FRAME = {...}
ECONOMIC_FRAME = {...}
HUMAN_INTEREST_FRAME = {...}
MORAL_FRAME = {...}
RESPONSIBILITY_FRAME = {...}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class NarrativeFrameFeatures(BaseFeature):

    name: str = "narrative_frame_features"
    group: str = "framing"
    description: str = "Normalized narrative frame features"

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
            "conflict": ratio(CONFLICT_FRAME),
            "economic": ratio(ECONOMIC_FRAME),
            "human": ratio(HUMAN_INTEREST_FRAME),
            "moral": ratio(MORAL_FRAME),
            "responsibility": ratio(RESPONSIBILITY_FRAME),
        }

        # -------------------------
        # NORMALIZED DISTRIBUTION
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = values.sum()

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -------------------------
        # INTENSITY
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY
        # -------------------------

        probs = np.array(list(dist.values()), dtype=np.float32)

        entropy = normalized_entropy(probs)

        # -------------------------
        # DOMINANCE (FIXED)
        # -------------------------

        dominance = float(np.max(probs))

        # -------------------------
        # BALANCE (FIXED)
        # -------------------------

        balance = 1.0 - float(np.std(probs))

        # -------------------------
        # RHETORIC (FIXED)
        # -------------------------

        rhetoric = (text.count("!") + text.count("?")) / (n + EPS)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "frame_conflict": self._safe(dist["conflict"]),
            "frame_economic": self._safe(dist["economic"]),
            "frame_human_interest": self._safe(dist["human"]),
            "frame_moral": self._safe(dist["moral"]),
            "frame_responsibility": self._safe(dist["responsibility"]),

            "frame_intensity": self._safe(intensity),
            "frame_entropy": self._safe(entropy),

            "frame_dominance": self._safe(dominance),
            "frame_balance": self._safe(balance),

            "frame_rhetoric": self._safe(rhetoric),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))