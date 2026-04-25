# src/features/token_features.py

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class TokenFeatures(BaseFeature):

    name: str = "token_features"
    group: str = "token"
    description: str = "Advanced token distribution features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = context.tokens or _simple_tokenize(text)
        n = len(tokens)

        if n == 0:
            return self._empty()

        tokens_arr = np.array(tokens, dtype=str)
        unique, counts = np.unique(tokens_arr, return_counts=True)

        vocab = len(unique)

        # -------------------------
        # BASIC NORMALIZED
        # -------------------------

        length_norm = np.log1p(n) / 10.0
        vocab_norm = np.log1p(vocab) / 10.0

        # -------------------------
        # FREQUENCY DISTRIBUTION
        # -------------------------

        probs = counts / (n + EPS)

        # entropy
        entropy_raw = -np.sum(probs * np.log(probs + EPS))
        entropy = entropy_raw / np.log(len(probs) + EPS)

        # -------------------------
        # CONCENTRATION (TOP-K)
        # -------------------------

        topk = np.sort(probs)[-5:] if len(probs) >= 5 else probs
        topk_mass = float(np.sum(topk))

        # -------------------------
        # REPETITION STRENGTH
        # -------------------------

        repetition = float(np.sum(probs ** 2))

        # -------------------------
        # INEQUALITY (GINI-LIKE)
        # -------------------------

        sorted_probs = np.sort(probs)
        gini = float(1.0 - 2.0 * np.sum((len(probs) - np.arange(len(probs))) * sorted_probs) / (len(probs) + EPS))

        # -------------------------
        # TOKEN LENGTH STATS
        # -------------------------

        lengths = np.char.str_len(tokens_arr)

        avg_len = float(np.mean(lengths) / 20.0)
        std_len = float(np.std(lengths) / 10.0)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "tok_length_norm": self._safe(length_norm),
            "tok_vocab_norm": self._safe(vocab_norm),

            "tok_entropy": self._safe(entropy),
            "tok_topk_mass": self._safe(topk_mass),

            "tok_repetition_strength": self._safe(repetition),
            "tok_gini": self._safe(gini),

            "tok_avg_length": self._safe(avg_len),
            "tok_std_length": self._safe(std_len),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "tok_length_norm": 0.0,
            "tok_vocab_norm": 0.0,
            "tok_entropy": 0.0,
            "tok_topk_mass": 0.0,
            "tok_repetition_strength": 0.0,
            "tok_gini": 0.0,
            "tok_avg_length": 0.0,
            "tok_std_length": 0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))