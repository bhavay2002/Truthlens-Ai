# src/features/bias_lexicon_features.py 

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.lexicon_matcher import (
    WeightedLexiconMatcher,
    compute_negation_mask,
    to_token_array,
)

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# TOKENIZATION
# =========================================================

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


# =========================================================
# NEGATION
# =========================================================

NEGATIONS = {"not", "no", "never", "n't"}


def _neg_factor(tokens: List[str], idx: int, window: int = 3) -> float:
    start = max(0, idx - window)
    return 0.3 if any(t in NEGATIONS for t in tokens[start:idx]) else 1.0


def _weighted_count(tokens: List[str], lexicon: Set[str]) -> float:
    score = 0.0
    for i, t in enumerate(tokens):
        if t in lexicon:
            score += _neg_factor(tokens, i)
    return score


# =========================================================
# LEXICONS (same as yours)
# =========================================================

EVALUATIVE_WORDS = {...}
ASSERTIVE_WORDS = {...}
HEDGING_WORDS = {...}
INTENSIFIERS = {...}

COMPILED_BIAS_PHRASES = [...]


# =========================================================
# VECTORIZED MATCHERS (built once at import)
# =========================================================

_BIAS_LEX_MATCHERS: Dict[str, WeightedLexiconMatcher] = {
    "eval":   WeightedLexiconMatcher(EVALUATIVE_WORDS, "eval"),
    "assert": WeightedLexiconMatcher(ASSERTIVE_WORDS,  "assert"),
    "hedge":  WeightedLexiconMatcher(HEDGING_WORDS,    "hedge"),
    "intens": WeightedLexiconMatcher(INTENSIFIERS,     "intens"),
}


# =========================================================
# FEATURE
# =========================================================

@dataclass
@register_feature
class BiasLexiconFeatures(BaseFeature):

    name: str = "bias_lexicon_features"
    group: str = "bias"

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        text_lower = text.lower()
        tokens = context.tokens or _tokenize(text_lower)

        n = len(tokens)
        if n == 0:
            return {}

        # -------------------------
        # RAW COUNTS (vectorized)
        # -------------------------

        tokens_arr = to_token_array(tokens)
        neg_mask = compute_negation_mask(tokens_arr, NEGATIONS, window=3)

        raw = {
            key: matcher.negation_aware_sum(tokens_arr, neg_mask)
            for key, matcher in _BIAS_LEX_MATCHERS.items()
        }

        total_bias = sum(raw.values())

        # -------------------------
        # RATIOS
        # -------------------------

        ratios = {k: v / (n + EPS) for k, v in raw.items()}

        # -------------------------
        # NORMALIZED DISTRIBUTION (CRITICAL)
        # -------------------------

        values = np.array(list(raw.values()), dtype=np.float32)
        total = values.sum()

        if total < EPS:
            dist = {k: 0.0 for k in raw}
        else:
            norm = values / (total + EPS)
            dist = dict(zip(raw.keys(), norm.astype(float)))

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
        # PHRASES (COUNTED)
        # -------------------------

        phrase_hits = sum(
            len(p.findall(text_lower)) for p in COMPILED_BIAS_PHRASES
        )
        phrase_score = phrase_hits / (n + EPS)

        # -------------------------
        # STRUCTURAL (FIXED)
        # -------------------------

        exclam = text.count("!")
        exclam_density = exclam / (n + EPS)

        caps_ratio = sum(
            1 for w in text.split() if w.isupper() and len(w) > 2
        ) / (n + EPS)

        # -------------------------
        # HIGH-LEVEL SIGNALS
        # -------------------------

        subjectivity = ratios["eval"] + ratios["intens"]

        # bounded certainty
        certainty = ratios["assert"] / (
            ratios["assert"] + ratios["hedge"] + EPS
        )

        polarity_balance = abs(ratios["assert"] - ratios["hedge"])

        density = total_bias / (n + EPS)

        intensity = float(np.mean(list(ratios.values())))

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "bias_eval_ratio": self._safe(ratios["eval"]),
            "bias_assertive_ratio": self._safe(ratios["assert"]),
            "bias_hedging_ratio": self._safe(ratios["hedge"]),
            "bias_intensifier_ratio": self._safe(ratios["intens"]),

            "bias_phrase_score": self._safe(phrase_score),
            "bias_exclamation_density": self._safe(exclam_density),
            "bias_caps_ratio": self._safe(caps_ratio),

            "bias_density": self._safe(density),
            "bias_intensity": self._safe(intensity),

            "bias_subjectivity": self._safe(subjectivity),
            "bias_certainty": self._safe(certainty),
            "bias_polarity_balance": self._safe(polarity_balance),

            "bias_entropy": self._safe(entropy),
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    # -----------------------------------------------------

    def extract_batch(self, contexts):
        return [self.extract(ctx) for ctx in contexts]