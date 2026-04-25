# src/features/propaganda_lexicon_features.py

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
# Lexicons (reuse yours)
# ---------------------------------------------------------

NAME_CALLING = {...}
FEAR_APPEAL = {...}
EXAGGERATION = {...}
BANDWAGON = {...}
SLOGANS = {...}

BANDWAGON_PHRASES = [...]
SLOGAN_PHRASES = [...]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _count(counter: Counter, lexicon: Set[str]) -> int:
    return sum(counter.get(w, 0) for w in lexicon)


def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    return _count(counter, lexicon) / (total + EPS)


def _phrase_hits(text: str, patterns: List[str]) -> int:
    return sum(bool(re.search(p, text)) for p in patterns)


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class PropagandaLexiconFeatures(BaseFeature):

    name: str = "propaganda_lexicon_features"
    group: str = "propaganda"
    description: str = "Normalized propaganda lexicon + phrase features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip().lower()
        if not text:
            return {}

        tokens = context.tokens or _tokenize(text)
        n = len(tokens)

        if n == 0:
            return {}

        counter = Counter(tokens)

        raw = {
            "name_calling": _ratio(counter, NAME_CALLING, n),
            "fear": _ratio(counter, FEAR_APPEAL, n),
            "exaggeration": _ratio(counter, EXAGGERATION, n),
            "bandwagon": _ratio(counter, BANDWAGON, n),
            "slogan": _ratio(counter, SLOGANS, n),
        }

        # -------------------------
        # PHRASE SIGNALS (INTEGRATED)
        # -------------------------

        phrase_bandwagon = _phrase_hits(text, BANDWAGON_PHRASES)
        phrase_slogan = _phrase_hits(text, SLOGAN_PHRASES)

        raw["bandwagon"] += phrase_bandwagon * 0.1
        raw["slogan"] += phrase_slogan * 0.1

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
        # DIVERSITY
        # -------------------------

        diversity = float(np.count_nonzero(values) / len(values))

        # -------------------------
        # RHETORIC
        # -------------------------

        rhetoric = (text.count("!") + text.count("?")) / (n + EPS)

        # -------------------------
        # CAPS EMPHASIS
        # -------------------------

        caps_tokens = sum(
            1 for w in text.split() if w.isupper() and len(w) > 2
        )
        caps_ratio = caps_tokens / (n + EPS)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "prop_lex_name_calling": self._safe(dist["name_calling"]),
            "prop_lex_fear": self._safe(dist["fear"]),
            "prop_lex_exaggeration": self._safe(dist["exaggeration"]),
            "prop_lex_bandwagon": self._safe(dist["bandwagon"]),
            "prop_lex_slogan": self._safe(dist["slogan"]),

            "prop_lex_phrase_bandwagon": float(phrase_bandwagon),
            "prop_lex_phrase_slogan": float(phrase_slogan),

            "prop_lex_intensity": self._safe(intensity),
            "prop_lex_entropy": self._safe(entropy),
            "prop_lex_diversity": self._safe(diversity),

            "prop_lex_rhetoric": self._safe(rhetoric),
            "prop_lex_caps_ratio": self._safe(caps_ratio),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))