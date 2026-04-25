# src/features/lexical_features.py

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

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class LexicalFeatures(BaseFeature):

    name: str = "lexical_features"
    group: str = "lexical"
    description: str = "Advanced lexical richness and diversity features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = context.tokens or _tokenize(text)
        n = len(tokens)

        if n == 0:
            return self._empty()

        tokens_arr = np.array(tokens, dtype=str)
        unique, counts = np.unique(tokens_arr, return_counts=True)

        vocab_size = len(unique)

        # -------------------------
        # BASIC RATIOS
        # -------------------------

        ttr = vocab_size / (n + EPS)

        # Corrected TTR (better for length variation)
        cttr = vocab_size / np.sqrt(2 * n + EPS)

        hapax_1 = np.sum(counts == 1)
        hapax_2 = np.sum(counts == 2)

        hapax_ratio = hapax_1 / (n + EPS)
        dislegomena_ratio = hapax_2 / (n + EPS)

        # -------------------------
        # ENTROPY (CRITICAL)
        # -------------------------

        probs = counts / (n + EPS)

        entropy_raw = -np.sum(probs * np.log(probs + EPS))
        entropy = entropy_raw / (np.log(len(probs) + EPS))

        # -------------------------
        # SIMPSON DIVERSITY
        # -------------------------

        simpson = 1.0 - np.sum(probs ** 2)

        # -------------------------
        # YULE'S K (ADVANCED)
        # -------------------------

        freq_sq_sum = np.sum(counts ** 2)
        yule_k = (freq_sq_sum - n) / (n ** 2 + EPS)

        # -------------------------
        # WORD LENGTH STATS
        # -------------------------

        lengths = np.char.str_len(tokens_arr)

        avg_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "lex_vocab_ttr": self._safe(ttr),
            "lex_vocab_cttr": self._safe(cttr),

            "lex_hapax_ratio": self._safe(hapax_ratio),
            "lex_dislegomena_ratio": self._safe(dislegomena_ratio),

            "lex_entropy": self._safe(entropy),
            "lex_simpson_diversity": self._safe(simpson),

            "lex_yule_k": self._safe(yule_k),

            "lex_avg_word_length": self._safe(avg_len / 20.0),  # normalized
            "lex_std_word_length": self._safe(std_len / 10.0),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "lex_vocab_ttr": 0.0,
            "lex_vocab_cttr": 0.0,
            "lex_hapax_ratio": 0.0,
            "lex_dislegomena_ratio": 0.0,
            "lex_entropy": 0.0,
            "lex_simpson_diversity": 0.0,
            "lex_yule_k": 0.0,
            "lex_avg_word_length": 0.0,
            "lex_std_word_length": 0.0,
        }

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))