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
# Bias Lexicons
# ---------------------------------------------------------

# ---------------------------------------------------------
# Loaded evaluative language
# ---------------------------------------------------------

LOADED_LANGUAGE: Set[str] = {
    "radical", "extreme", "extremist",
    "outrageous", "absurd", "ridiculous",
    "disaster", "catastrophe",
    "corrupt", "corruption",
    "shocking", "disturbing",
    "devastating", "dangerous",
    "reckless", "irresponsible",
    "disgraceful", "scandalous",
    "terrible", "horrible",
    "awful", "pathetic",
    "shameful", "alarming"
}


# ---------------------------------------------------------
# Subjective / opinionated phrasing
# ---------------------------------------------------------

SUBJECTIVE_WORDS: Set[str] = {
    "clearly", "obviously", "undoubtedly",
    "certainly", "surely",
    "apparently", "evidently",
    "unfortunately", "fortunately",
    "remarkably", "interestingly",
    "surprisingly", "notably",
    "frankly", "honestly",
    "ironically"
}


# ---------------------------------------------------------
# Uncertainty / hedging
# ---------------------------------------------------------

UNCERTAINTY_WORDS: Set[str] = {
    "allegedly", "reportedly", "apparently",
    "possibly", "potentially",
    "suggests", "indicates",
    "may", "might", "could",
    "perhaps", "likely",
    "unlikely", "presumably",
    "seemingly"
}


# ---------------------------------------------------------
# Polarization framing
# ---------------------------------------------------------

POLARIZING_WORDS: Set[str] = {
    "enemy", "enemies",
    "threat", "threats",
    "attack", "attacks",
    "destroy", "destruction",
    "fight", "battle",
    "war", "conflict",
    "crisis", "collapse",
    "chaos", "division",
    "clash", "confrontation"
}


# ---------------------------------------------------------
# Evaluative adjectives
# ---------------------------------------------------------

EVALUATIVE_WORDS: Set[str] = {
    "good", "bad",
    "terrible", "awful",
    "excellent", "outstanding",
    "strong", "weak",
    "successful", "failed",
    "effective", "ineffective",
    "remarkable", "poor",
    "significant", "insignificant"
}


# Phrase-based bias patterns
BIAS_PHRASES = [

    # certainty framing
    r"clearly\s+shows",
    r"it\s+is\s+obvious",
    r"there\s+is\s+no\s+doubt",
    r"it\s+is\s+clear\s+that",

    # strong claims
    r"the\s+truth\s+is",
    r"the\s+reality\s+is",
    r"what\s+this\s+really\s+means",

    # rhetorical framing
    r"the\s+fact\s+is",
    r"everyone\s+knows",
    r"it\s+is\s+undeniable",

    # opinion framing
    r"in\s+reality",
    r"in\s+truth",
]


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
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class BiasFeatures(BaseFeature):

    """
    Extract bias-related linguistic signals.

    Output Features
    ---------------
    bias_loaded_language_ratio
    bias_subjective_ratio
    bias_uncertainty_ratio
    bias_polarization_ratio
    bias_evaluative_ratio
    bias_phrase_count
    bias_exclamation_density
    bias_caps_ratio
    bias_intensity
    bias_diversity
    """

    name: str = "bias_features"
    description: str = "Bias and subjective language indicators"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        text = context.text

        tokens = context.tokens or _tokenize(text)

        if not tokens:
            logger.warning("No tokens available for bias feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        loaded_ratio = _ratio(counter, LOADED_LANGUAGE, total_tokens)
        subjective_ratio = _ratio(counter, SUBJECTIVE_WORDS, total_tokens)
        uncertainty_ratio = _ratio(counter, UNCERTAINTY_WORDS, total_tokens)
        polarization_ratio = _ratio(counter, POLARIZING_WORDS, total_tokens)
        evaluative_ratio = _ratio(counter, EVALUATIVE_WORDS, total_tokens)

        counts = [
            _count(counter, LOADED_LANGUAGE),
            _count(counter, SUBJECTIVE_WORDS),
            _count(counter, UNCERTAINTY_WORDS),
            _count(counter, POLARIZING_WORDS),
            _count(counter, EVALUATIVE_WORDS),
        ]

        # -------------------------------------------------
        # Phrase bias detection
        # -------------------------------------------------

        phrase_count = sum(bool(re.search(p, text.lower())) for p in BIAS_PHRASES)

        # -------------------------------------------------
        # Structural rhetoric signals
        # -------------------------------------------------

        exclamation_density = text.count("!") / max(len(text), 1)

        caps_tokens = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
        caps_ratio = caps_tokens / total_tokens

        # -------------------------------------------------
        # Aggregate metrics
        # -------------------------------------------------

        intensity = (
            loaded_ratio +
            subjective_ratio +
            polarization_ratio +
            evaluative_ratio
        ) / 4.0

        diversity = sum(1 for c in counts if c > 0) / len(counts)

        features: Dict[str, float] = {

            "bias_loaded_language_ratio": loaded_ratio,
            "bias_subjective_ratio": subjective_ratio,
            "bias_uncertainty_ratio": uncertainty_ratio,
            "bias_polarization_ratio": polarization_ratio,
            "bias_evaluative_ratio": evaluative_ratio,

            "bias_phrase_count": float(phrase_count),

            "bias_exclamation_density": exclamation_density,
            "bias_caps_ratio": caps_ratio,

            "bias_intensity": intensity,
            "bias_diversity": diversity,
        }

        logger.debug(
            "Bias features extracted | intensity=%.4f diversity=%.4f",
            intensity,
            diversity,
        )

        return features