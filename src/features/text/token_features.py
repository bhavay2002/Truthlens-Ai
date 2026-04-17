"""
File Name: token_features.py
Module: Text Feature Engineering - Token Features
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)


def _simple_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
@register_feature
class TokenFeatures(BaseFeature):
    name: str = "token_features"
    description: str = "Basic token-level lexical statistics"

    def _compute_features(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {
                "token_count": 0.0,
                "unique_token_count": 0.0,
                "type_token_ratio": 0.0,
                "avg_token_length": 0.0,
                "max_token_length": 0.0,
                "repetition_ratio": 0.0,
            }

        tokens_arr = np.asarray(tokens, dtype=str)
        token_count = tokens_arr.size
        unique_tokens, counts = np.unique(tokens_arr, return_counts=True)
        token_lengths = np.char.str_len(tokens_arr)

        unique_token_count = unique_tokens.size
        repeated_tokens = counts[counts > 1].sum()

        return {
            "token_count": float(token_count),
            "unique_token_count": float(unique_token_count),
            "type_token_ratio": float(unique_token_count / token_count),
            "avg_token_length": float(token_lengths.mean()),
            "max_token_length": float(token_lengths.max()),
            "repetition_ratio": float(repeated_tokens / token_count),
        }

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")
        if not context.text.strip():
            return {}

        tokens = context.tokens or _simple_tokenize(context.text)
        if not tokens:
            logger.warning("No tokens extracted from text")
            return self._compute_features([])

        features = self._compute_features(tokens)

        logger.debug(
            "Token features extracted | tokens=%d unique=%d",
            int(features["token_count"]),
            int(features["unique_token_count"]),
        )
        return features

    def extract_batch(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        return [self.extract(context) for context in contexts]