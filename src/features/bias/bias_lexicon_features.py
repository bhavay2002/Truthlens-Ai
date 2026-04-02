"""
File Name: bias_lexicon_features.py
Module: Feature Engineering - Bias Lexicon Features
Description:
    Extracts bias indicators using curated lexical resources commonly
    associated with subjective language, ideological framing, and
    evaluative reporting. The module measures the density and diversity
    of bias-related terms and produces normalized ratios useful for
    downstream bias detection models.

    This implementation is lightweight, deterministic, and integrates
    with the TruthLens feature framework through BaseFeature and
    FeatureRegistry.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing text and optional tokens

Outputs:
    Dict[str, float] representing bias lexicon statistics
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------
# Bias Lexicons
# ---------------------------------------------------------------------

EVALUATIVE_WORDS: Set[str] = {
    "outrageous", "shocking", "terrible", "amazing", "unacceptable",
    "ridiculous", "disastrous", "remarkable", "incredible"
}

ASSERTIVE_WORDS: Set[str] = {
    "clearly", "obviously", "undoubtedly", "certainly", "definitely"
}

HEDGING_WORDS: Set[str] = {
    "allegedly", "reportedly", "apparently", "possibly",
    "may", "might", "could", "suggests"
}

INTENSIFIERS: Set[str] = {
    "very", "extremely", "highly", "deeply", "strongly", "completely"
}


@dataclass
@register_feature
class BiasLexiconFeatures(BaseFeature):
    """
    Extracts bias indicators from lexical cues.

    Output Features
    ---------------
    bias_eval_ratio
    bias_assertive_ratio
    bias_hedging_ratio
    bias_intensifier_ratio
    bias_lexicon_density
    bias_lexicon_diversity
    """

    name: str = "bias_lexicon_features"
    description: str = "Lexicon-based bias indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for bias lexicon extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        eval_ratio = ratio(EVALUATIVE_WORDS)
        assert_ratio = ratio(ASSERTIVE_WORDS)
        hedge_ratio = ratio(HEDGING_WORDS)
        intens_ratio = ratio(INTENSIFIERS)

        lexicon_counts = [
            sum(counter.get(w, 0) for w in EVALUATIVE_WORDS),
            sum(counter.get(w, 0) for w in ASSERTIVE_WORDS),
            sum(counter.get(w, 0) for w in HEDGING_WORDS),
            sum(counter.get(w, 0) for w in INTENSIFIERS),
        ]

        total_bias_tokens = sum(lexicon_counts)
        density = total_bias_tokens / total_tokens

        diversity = sum(1 for c in lexicon_counts if c > 0) / len(lexicon_counts)

        features: Dict[str, float] = {
            "bias_eval_ratio": float(eval_ratio),
            "bias_assertive_ratio": float(assert_ratio),
            "bias_hedging_ratio": float(hedge_ratio),
            "bias_intensifier_ratio": float(intens_ratio),
            "bias_lexicon_density": float(density),
            "bias_lexicon_diversity": float(diversity),
        }

        logger.debug(
            "Bias lexicon features extracted | density=%.4f diversity=%.4f",
            density,
            diversity,
        )

        return features