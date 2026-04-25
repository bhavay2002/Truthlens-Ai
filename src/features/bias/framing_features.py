# src/features/framing_features.py 

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
# Lexicons (same as yours)
# ---------------------------------------------------------

ECONOMIC_FRAME = {...}
MORAL_FRAME = {...}
SECURITY_FRAME = {...}
HUMAN_INTEREST_FRAME = {...}
CONFLICT_FRAME = {...}

COMPILED_FRAME_PHRASES = [...]


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class FramingFeatures(BaseFeature):

    name: str = "framing_features"
    group: str = "framing"

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
        # RAW FRAME RATIOS
        # -------------------------

        raw = {
            "economic": _ratio(counter, ECONOMIC_FRAME, n),
            "moral": _ratio(counter, MORAL_FRAME, n),
            "security": _ratio(counter, SECURITY_FRAME, n),
            "human": _ratio(counter, HUMAN_INTEREST_FRAME, n),
            "conflict": _ratio(counter, CONFLICT_FRAME, n),
        }

        # -------------------------
        # NORMALIZED DISTRIBUTION (CRITICAL)
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = float(values.sum())

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

        # -------------------------
        # PHRASE INTENSITY (FIXED)
        # -------------------------

        phrase_hits = sum(len(p.findall(text_lower)) for p in COMPILED_FRAME_PHRASES)
        phrase_score = phrase_hits / (n + EPS)

        # -------------------------
        # STRUCTURAL SIGNALS
        # -------------------------

        quote_count = text.count('"')
        quote_density = quote_count / (n + EPS)

        # -------------------------
        # GLOBAL METRICS
        # -------------------------

        intensity = float(np.mean(list(raw.values())))

        probs = np.array(list(dist.values()), dtype=np.float32)

        if probs.sum() < EPS:
            entropy = 0.0
        else:
            entropy_raw = -np.sum(probs * np.log(probs + EPS))
            entropy = entropy_raw / (np.log(len(probs)) + EPS)

        diversity = sum(v > 0 for v in raw.values()) / len(raw)

        dominance = max(dist.values()) if dist else 0.0

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "frame_economic": self._safe(dist["economic"]),
            "frame_moral": self._safe(dist["moral"]),
            "frame_security": self._safe(dist["security"]),
            "frame_human": self._safe(dist["human"]),
            "frame_conflict": self._safe(dist["conflict"]),

            "frame_phrase_score": self._safe(phrase_score),
            "frame_quote_density": self._safe(quote_density),

            "frame_intensity": self._safe(intensity),
            "frame_diversity": self._safe(diversity),
            "frame_entropy": self._safe(entropy),
            "frame_dominance": self._safe(dominance),
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))