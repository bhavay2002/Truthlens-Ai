"""
File Name: token_features.py
Module: Text Feature Engineering - Token Features
Description:
    Implements token-level statistical features used for NLP analysis
    within the TruthLens feature engineering system. These features
    capture fundamental lexical characteristics of input text including
    token counts, vocabulary richness, repetition statistics, and
    length-based properties.

    The module integrates with the TruthLens feature abstraction layer
    and FeatureRegistry to allow automatic discovery and execution
    within the feature pipeline.

Dependencies:
    re
    collections
    dataclasses
    typing
    logging

Inputs:
    FeatureContext (contains text and optional tokens)

Outputs:
    Dict[str, float] containing token-level features
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
    """
    Basic tokenizer used as fallback when tokens are not
    provided in FeatureContext.

    Parameters
    ----------
    text : str

    Returns
    -------
    List[str]
    """

    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    tokens = re.findall(r"\b\w+\b", text.lower())

    return tokens


@dataclass
@register_feature
class TokenFeatures(BaseFeature):
    """
    Extracts token-level statistical features.

    Example features:
    - token_count
    - unique_token_count
    - type_token_ratio
    - avg_token_length
    - max_token_length
    - repetition_ratio
    """

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
            "avg_token_length": float(token_lengths.mean() if token_count else 0.0),
            "max_token_length": float(token_lengths.max() if token_count else 0.0),
            "repetition_ratio": float(repeated_tokens / token_count),
        }

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract token-level features.

        Parameters
        ----------
        context : FeatureContext

        Returns
        -------
        Dict[str, float]
        """

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _simple_tokenize(context.text)

        if not tokens:
            logger.warning("No tokens extracted from text")
            return self._compute_features([])

        features = self._compute_features(tokens)

        logger.debug(
            "Token features extracted | tokens=%d unique=%d",
            token_count,
            unique_token_count,
        )

        return features

    def extract_batch(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        """Extract token features for a batch of contexts."""

        return [self.extract(context) for context in contexts]