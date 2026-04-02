"""
File Name: bias_features.py
Module: Feature Engineering - Bias Features
Description:
    Extracts linguistic indicators associated with biased or subjective
    language in text. The module computes normalized statistics based on
    bias-related lexicons, sentiment polarity imbalance, and subjective
    phrasing patterns.

    These features help identify framing bias, loaded language, and
    subjective reporting patterns commonly observed in political or
    opinionated texts.

    The implementation integrates with the TruthLens feature system
    using the BaseFeature abstraction and FeatureRegistry.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing bias-related linguistic signals
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Bias Lexicons
# ---------------------------------------------------------------------

LOADED_LANGUAGE = {
    "radical", "extreme", "outrageous", "disaster", "corrupt",
    "shocking", "disturbing", "devastating", "dangerous"
}

SUBJECTIVE_WORDS = {
    "clearly", "obviously", "undoubtedly", "certainly",
    "unfortunately", "fortunately", "remarkably"
}

UNCERTAINTY_WORDS = {
    "allegedly", "reportedly", "apparently", "possibly",
    "suggests", "may", "might", "could"
}

POLARIZING_WORDS = {
    "enemy", "threat", "attack", "destroy", "fight",
    "war", "crisis", "collapse"
}


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------

@dataclass
@register_feature
class BiasFeatures(BaseFeature):
    """
    Extracts bias-related linguistic signals.

    Output features include:
        bias_loaded_language_ratio
        bias_subjective_ratio
        bias_uncertainty_ratio
        bias_polarization_ratio
        bias_intensity
    """

    name: str = "bias_features"
    description: str = "Bias and subjective language indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for bias feature extraction")
            return {}

        token_counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(word_set: set) -> float:
            count = sum(token_counter.get(w, 0) for w in word_set)
            return count / total_tokens

        loaded_ratio = ratio(LOADED_LANGUAGE)
        subjective_ratio = ratio(SUBJECTIVE_WORDS)
        uncertainty_ratio = ratio(UNCERTAINTY_WORDS)
        polarization_ratio = ratio(POLARIZING_WORDS)

        bias_intensity = (
            loaded_ratio +
            subjective_ratio +
            polarization_ratio
        ) / 3.0

        features: Dict[str, float] = {
            "bias_loaded_language_ratio": float(loaded_ratio),
            "bias_subjective_ratio": float(subjective_ratio),
            "bias_uncertainty_ratio": float(uncertainty_ratio),
            "bias_polarization_ratio": float(polarization_ratio),
            "bias_intensity": float(bias_intensity),
        }

        logger.debug(
            "Bias features extracted | intensity=%.4f",
            bias_intensity,
        )

        return features