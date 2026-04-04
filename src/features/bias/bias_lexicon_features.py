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


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    """Robust tokenizer for lexical bias detection."""
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def _count(counter: Counter, lexicon: Set[str]) -> int:
    return sum(counter.get(w, 0) for w in lexicon)


def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    if total == 0:
        return 0.0
    return _count(counter, lexicon) / total


# ---------------------------------------------------------
# Bias Lexicons
# ---------------------------------------------------------
# ---------------------------------------------------------
# evaluative framing (strong opinionated descriptors)
# ---------------------------------------------------------

EVALUATIVE_WORDS: Set[str] = {
    "outrageous","shocking","terrible","awful","horrible",
    "amazing","incredible","remarkable","extraordinary",
    "unacceptable","ridiculous","absurd","pathetic",
    "disastrous","devastating","catastrophic",
    "excellent","outstanding","poor","weak",
    "strong","successful","failed","ineffective",
    "dangerous","reckless","irresponsible",
    "alarming","disturbing"
}


# ---------------------------------------------------------
# strong certainty / assertive framing
# ---------------------------------------------------------

ASSERTIVE_WORDS: Set[str] = {
    "clearly","obviously","undoubtedly","certainly",
    "definitely","surely","evidently","plainly",
    "unquestionably","indisputably","undeniably",
}


# ---------------------------------------------------------
# hedging / epistemic uncertainty
# ---------------------------------------------------------

HEDGING_WORDS: Set[str] = {
    "allegedly","reportedly","apparently",
    "possibly","potentially",
    "may","might","could",
    "perhaps","likely","unlikely",
    "suggests","indicates","seems",
    "presumably","arguably",
    "roughly","approximately"
}


# ---------------------------------------------------------
# emotional amplification
# ---------------------------------------------------------

INTENSIFIERS: Set[str] = {
    "very","extremely","highly","deeply",
    "strongly","completely","totally",
    "remarkably","particularly",
    "incredibly","exceptionally",
    "significantly","dramatically",
    "massively","tremendously"
}

BIAS_PHRASES = [

    # certainty framing
    r"it\s+is\s+clear\s+that",
    r"there\s+is\s+no\s+doubt",
    r"it\s+is\s+obvious",
    r"it\s+is\s+evident",

    # rhetorical framing
    r"the\s+truth\s+is",
    r"the\s+fact\s+is",
    r"the\s+reality\s+is",

    # ideological persuasion
    r"everyone\s+knows",
    r"no\s+one\s+can\s+deny",
    r"it\s+is\s+undeniable",

    # opinion framing
    r"in\s+reality",
    r"in\s+truth",
]


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class BiasLexiconFeatures(BaseFeature):

    """
    Extract bias indicators from lexical cues.

    Output Features
    ---------------
    bias_eval_ratio
    bias_assertive_ratio
    bias_hedging_ratio
    bias_intensifier_ratio
    bias_phrase_count
    bias_exclamation_density
    bias_caps_ratio
    bias_lexicon_density
    bias_lexicon_diversity
    """

    name: str = "bias_lexicon_features"
    description: str = "Lexicon-based bias indicators"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        text = context.text
        tokens = context.tokens or _tokenize(text)

        if not tokens:
            logger.warning("No tokens available for bias lexicon extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        eval_ratio = _ratio(counter, EVALUATIVE_WORDS, total_tokens)
        assert_ratio = _ratio(counter, ASSERTIVE_WORDS, total_tokens)
        hedge_ratio = _ratio(counter, HEDGING_WORDS, total_tokens)
        intens_ratio = _ratio(counter, INTENSIFIERS, total_tokens)

        counts = [
            _count(counter, EVALUATIVE_WORDS),
            _count(counter, ASSERTIVE_WORDS),
            _count(counter, HEDGING_WORDS),
            _count(counter, INTENSIFIERS),
        ]

        total_bias_tokens = sum(counts)

        density = total_bias_tokens / total_tokens
        diversity = sum(1 for c in counts if c > 0) / len(counts)

        # -------------------------------------------------
        # phrase bias detection
        # -------------------------------------------------

        phrase_count = sum(
            bool(re.search(p, text.lower()))
            for p in BIAS_PHRASES
        )

        # -------------------------------------------------
        # structural bias indicators
        # -------------------------------------------------

        exclamation_density = text.count("!") / max(len(text), 1)

        caps_words = sum(
            1 for w in text.split() if w.isupper() and len(w) > 2
        )

        caps_ratio = caps_words / total_tokens

        features: Dict[str, float] = {

            "bias_eval_ratio": eval_ratio,
            "bias_assertive_ratio": assert_ratio,
            "bias_hedging_ratio": hedge_ratio,
            "bias_intensifier_ratio": intens_ratio,

            "bias_phrase_count": float(phrase_count),

            "bias_exclamation_density": exclamation_density,
            "bias_caps_ratio": caps_ratio,

            "bias_lexicon_density": density,
            "bias_lexicon_diversity": diversity,
        }

        logger.debug(
            "Bias lexicon features extracted | density=%.4f diversity=%.4f",
            density,
            diversity,
        )

        return features