# src/features/discourse_features.py

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# Lexicons
# ---------------------------------------------------------

CAUSAL = {"because","since","therefore","thus","hence","consequently"}
CONTRAST = {"however","but","although","though","nevertheless","yet"}
ADDITIVE = {"also","furthermore","moreover","additionally","besides"}
SEQUENTIAL = {"first","second","then","next","finally"}
EVIDENTIAL = {"according","reported","evidence","study","data","research"}


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class DiscourseFeatures(BaseFeature):

    name: str = "discourse_features"
    group: str = "discourse"
    description: str = "Normalized discourse structure features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = context.tokens or _tokenize(text)
        n = len(tokens)

        if n == 0:
            return {}

        counter = Counter(tokens)

        def ratio(lexicon: Set[str]) -> float:
            return sum(counter.get(w, 0) for w in lexicon) / (n + EPS)

        raw = {
            "causal": ratio(CAUSAL),
            "contrast": ratio(CONTRAST),
            "additive": ratio(ADDITIVE),
            "sequential": ratio(SEQUENTIAL),
            "evidential": ratio(EVIDENTIAL),
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

        probs = np.array(list(dist.values()), dtype=np.float32)

        # -------------------------
        # INTENSITY
        # -------------------------

        intensity = float(np.linalg.norm(values))

        # -------------------------
        # ENTROPY (CRITICAL)
        # -------------------------

        if probs.sum() > 0:
            entropy_raw = -np.sum(probs * np.log(probs + EPS))
            entropy = entropy_raw / (np.log(len(probs)) + EPS)
        else:
            entropy = 0.0

        # -------------------------
        # BALANCE
        # -------------------------

        balance = 1.0 - float(np.std(probs))

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "disc_causal": self._safe(dist["causal"]),
            "disc_contrast": self._safe(dist["contrast"]),
            "disc_additive": self._safe(dist["additive"]),
            "disc_sequential": self._safe(dist["sequential"]),
            "disc_evidential": self._safe(dist["evidential"]),

            "disc_intensity": self._safe(intensity),
            "disc_entropy": self._safe(entropy),
            "disc_balance": self._safe(balance),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))