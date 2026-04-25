# src/features/ideological_features.py (RESEARCH-GRADE FINAL)

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

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    return sum(counter.get(w, 0) for w in lexicon) / (total + EPS)


# ---------------------------------------------------------
# Lexicons (same)
# ---------------------------------------------------------

LEFT_LEXICON = {...}
RIGHT_LEXICON = {...}
POLARIZING_TERMS = {...}
GROUP_REFERENCES = {...}
COMPILED_IDEOLOGY_PHRASES = [...]


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class IdeologicalFeatures(BaseFeature):

    name: str = "ideological_features"
    group: str = "ideology"

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        text_lower = text.lower()
        tokens = context.tokens or _tokenize(text_lower)

        if not tokens:
            return {}

        counter = Counter(tokens)
        n = len(tokens)

        # -------------------------
        # RAW RATIOS
        # -------------------------

        raw = {
            "left": _ratio(counter, LEFT_LEXICON, n),
            "right": _ratio(counter, RIGHT_LEXICON, n),
        }

        polarization = _ratio(counter, POLARIZING_TERMS, n)
        group_ref = _ratio(counter, GROUP_REFERENCES, n)

        # -------------------------
        # NORMALIZED IDEOLOGY (CRITICAL)
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = values.sum()

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -------------------------
        # BALANCE (FIXED)
        # -------------------------

        balance = 1.0 - abs(dist["left"] - dist["right"])

        # -------------------------
        # ENTROPY (FIXED)
        # -------------------------

        probs = np.array(list(dist.values()), dtype=np.float32)

        if probs.sum() < EPS:
            entropy = 0.0
        else:
            entropy_raw = -np.sum(probs * np.log(probs + EPS))
            entropy = entropy_raw / (np.log(len(probs)) + EPS)

        # -------------------------
        # PHRASE INTENSITY
        # -------------------------

        phrase_hits = sum(
            len(p.findall(text_lower)) for p in COMPILED_IDEOLOGY_PHRASES
        )
        phrase_score = phrase_hits / (n + EPS)

        # -------------------------
        # GLOBAL SIGNALS
        # -------------------------

        intensity = float(np.mean(list(raw.values())))

        signal_strength = (
            (raw["left"] + raw["right"]) * 0.6 +
            polarization * 0.4
        )

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "ideology_left": self._safe(dist["left"]),
            "ideology_right": self._safe(dist["right"]),

            "ideology_balance": self._safe(balance),
            "ideology_entropy": self._safe(entropy),

            "ideology_polarization": self._safe(polarization),
            "ideology_group_reference": self._safe(group_ref),

            "ideology_phrase_score": self._safe(phrase_score),

            "ideology_intensity": self._safe(intensity),
            "ideology_signal_strength": self._safe(signal_strength),
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))