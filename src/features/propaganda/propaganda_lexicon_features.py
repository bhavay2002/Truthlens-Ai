"""
File Name: propaganda_lexicon_features.py
Module: Feature Engineering - Propaganda Lexicon Features
Description:
    Extracts propaganda-related linguistic signals using curated lexicons.
    The module measures the density and diversity of propaganda techniques
    present in the text such as name-calling, fear appeals, exaggeration,
    bandwagon language, and slogan-like phrasing.

    These features provide interpretable indicators useful for propaganda
    detection models and explainability components within the TruthLens
    system.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing propaganda lexicon statistics
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
    """Robust tokenizer for lexical propaganda detection."""
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

def _count(counter: Counter, lexicon: Set[str]) -> int:
    return sum(counter.get(w, 0) for w in lexicon)


def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    if total == 0:
        return 0.0
    return _count(counter, lexicon) / total


# ---------------------------------------------------------
# Propaganda Lexicons
# ---------------------------------------------------------

# ---------------------------------------------------------
# Name Calling / Labeling Opponents
# ---------------------------------------------------------

NAME_CALLING: Set[str] = {
    "traitor", "traitors",
    "enemy", "enemies",
    "corrupt", "corruption",
    "liar", "liars", "lying",
    "radical", "radicals",
    "extremist", "extremists",
    "fraud", "fraudulent",
    "criminal", "criminals",
    "crooked", "fake",
    "dishonest", "evil",
    "scandal", "scandalous",
    "propagandist",
    "hypocrite", "hypocrisy",
    "thug", "terrorist",
    "dictator", "tyrant",
    "puppet", "stooge",
    "conspirator"
}


# ---------------------------------------------------------
# Fear Appeal Framing
# ---------------------------------------------------------

FEAR_APPEAL: Set[str] = {
    "danger", "dangerous",
    "threat", "threats", "threatening",
    "terror", "terrorist", "terrorism",
    "crisis", "emergency",
    "attack", "attacks", "attacking",
    "invasion", "invading",
    "collapse", "breakdown",
    "disaster", "catastrophe", "catastrophic",
    "chaos", "panic",
    "violence", "violent",
    "war", "conflict",
    "instability"
}


# ---------------------------------------------------------
# Exaggeration / Absolutist Language
# ---------------------------------------------------------

EXAGGERATION: Set[str] = {
    "always", "never",
    "everyone", "everybody",
    "nobody", "noone",
    "all", "none",
    "everything", "nothing",
    "completely", "absolutely",
    "totally", "entirely",
    "undeniable", "certainly",
    "guaranteed", "inevitable",
    "perfectly"
}


# ---------------------------------------------------------
# Bandwagon Language
# ---------------------------------------------------------

BANDWAGON: Set[str] = {
    "everyone", "everybody",
    "majority", "millions",
    "many", "most",
    "all", "nation",
    "people", "citizens",
    "community", "society",
    "public", "population"
}


# ---------------------------------------------------------
# Ideological Slogans / Values
# ---------------------------------------------------------

SLOGANS: Set[str] = {
    "freedom", "democracy", "justice",
    "patriotism", "honor", "truth",
    "liberty", "unity", "rights",
    "equality", "fairness",
    "values", "principles",
    "prosperity", "progress",
    "security", "strength"
}

# Phrase-based patterns (important for propaganda slogans)

BANDWAGON_PHRASES = [
    r"everyone\s+knows",
    r"most\s+people",
    r"the\s+people",
    r"the\s+nation",
]

SLOGAN_PHRASES = [
    r"fight\s+for\s+freedom",
    r"defend\s+democracy",
    r"stand\s+for\s+justice",
]


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class PropagandaLexiconFeatures(BaseFeature):

    """
    Extract propaganda lexicon features.

    Output Features
    ---------------

    propaganda_name_calling_ratio
    propaganda_fear_ratio
    propaganda_exaggeration_ratio
    propaganda_bandwagon_ratio
    propaganda_slogan_ratio

    propaganda_phrase_bandwagon
    propaganda_phrase_slogan

    propaganda_exclamation_density
    propaganda_caps_ratio

    propaganda_lexicon_density
    propaganda_lexicon_diversity
    """

    name: str = "propaganda_lexicon_features"
    description: str = "Lexicon-based propaganda detection features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        text = context.text
        tokens = context.tokens or _tokenize(text)

        if not tokens:
            logger.warning("No tokens available for propaganda lexicon extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        # -------------------------------------------------
        # Lexicon ratios
        # -------------------------------------------------

        name_ratio = _ratio(counter, NAME_CALLING, total_tokens)
        fear_ratio = _ratio(counter, FEAR_APPEAL, total_tokens)
        exaggeration_ratio = _ratio(counter, EXAGGERATION, total_tokens)
        bandwagon_ratio = _ratio(counter, BANDWAGON, total_tokens)
        slogan_ratio = _ratio(counter, SLOGANS, total_tokens)

        counts = [
            _count(counter, NAME_CALLING),
            _count(counter, FEAR_APPEAL),
            _count(counter, EXAGGERATION),
            _count(counter, BANDWAGON),
            _count(counter, SLOGANS),
        ]

        total_prop_tokens = sum(counts)

        density = total_prop_tokens / total_tokens
        diversity = sum(1 for c in counts if c > 0) / len(counts)

        # -------------------------------------------------
        # Phrase patterns
        # -------------------------------------------------

        phrase_bandwagon = sum(bool(re.search(p, text.lower())) for p in BANDWAGON_PHRASES)
        phrase_slogan = sum(bool(re.search(p, text.lower())) for p in SLOGAN_PHRASES)

        # -------------------------------------------------
        # Structural rhetoric signals
        # -------------------------------------------------

        exclamation_density = text.count("!") / max(len(text), 1)

        caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
        caps_ratio = caps_words / total_tokens

        features: Dict[str, float] = {

            "propaganda_name_calling_ratio": name_ratio,
            "propaganda_fear_ratio": fear_ratio,
            "propaganda_exaggeration_ratio": exaggeration_ratio,
            "propaganda_bandwagon_ratio": bandwagon_ratio,
            "propaganda_slogan_ratio": slogan_ratio,

            "propaganda_phrase_bandwagon": float(phrase_bandwagon),
            "propaganda_phrase_slogan": float(phrase_slogan),

            "propaganda_exclamation_density": exclamation_density,
            "propaganda_caps_ratio": caps_ratio,

            "propaganda_lexicon_density": density,
            "propaganda_lexicon_diversity": diversity,
        }

        logger.debug(
            "Propaganda lexicon features extracted | density=%.4f diversity=%.4f",
            density,
            diversity,
        )

        return features