"""
File Name: lexical_features.py
Module: Text Feature Engineering - Lexical Features
Description:
    Implements lexical richness and vocabulary diversity features for the
    TruthLens feature engineering system. These features quantify the
    complexity, diversity, and structural properties of vocabulary used in
    the input text.

    The module integrates with the TruthLens BaseFeature abstraction and
    FeatureRegistry for automatic discovery and execution in the feature
    pipeline.

Dependencies:
    re
    collections
    dataclasses
    typing
    logging

Inputs:
    FeatureContext (text and optional tokens)

Outputs:
    Dict[str, float] containing lexical richness features
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


def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer fallback if tokens are not provided
    in FeatureContext.
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")

    return re.findall(r"\b\w+\b", text.lower())


@dataclass
@register_feature
class LexicalFeatures(BaseFeature):
    """
    Computes lexical diversity and richness statistics.

    Example features:
    - vocabulary_size
    - hapax_legomena_ratio
    - hapax_dislegomena_ratio
    - lexical_density
    - average_word_length
    """

    name: str = "lexical_features"
    description: str = "Lexical richness and vocabulary diversity metrics"

    def _compute_features(self, tokens: List[str]) -> Dict[str, float]:
        if not tokens:
            return {
                "vocabulary_size": 0.0,
                "hapax_legomena_ratio": 0.0,
                "hapax_dislegomena_ratio": 0.0,
                "lexical_density": 0.0,
                "average_word_length": 0.0,
            }

        tokens_arr = np.asarray(tokens, dtype=str)
        token_count = tokens_arr.size
        unique_tokens, counts = np.unique(tokens_arr, return_counts=True)
        word_lengths = np.char.str_len(tokens_arr)

        vocabulary_size = unique_tokens.size
        hapax_legomena = np.count_nonzero(counts == 1)
        hapax_dislegomena = np.count_nonzero(counts == 2)

        return {
            "vocabulary_size": float(vocabulary_size),
            "hapax_legomena_ratio": float(hapax_legomena / token_count),
            "hapax_dislegomena_ratio": float(hapax_dislegomena / token_count),
            "lexical_density": float(vocabulary_size / token_count),
            "average_word_length": float(word_lengths.mean() if token_count else 0.0),
        }

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract lexical features from input text.

        Parameters
        ----------
        context : FeatureContext

        Returns
        -------
        Dict[str, float]
        """

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens extracted from text")
            return self._compute_features([])

        features = self._compute_features(tokens)

        logger.debug(
            "Lexical features extracted | tokens=%d vocab=%d",
            token_count,
            vocabulary_size,
        )

        return features

    def extract_batch(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        """Extract lexical features for a batch of contexts."""

        return [self.extract(context) for context in contexts]