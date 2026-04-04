"""
File Name: propaganda_features.py
Module: Feature Engineering - Propaganda Features
Description:
    Extracts linguistic signals associated with propaganda techniques
    commonly found in political communication and misinformation. The
    module uses curated lexicons and heuristic rules to detect patterns
    such as name-calling, fear appeals, exaggeration, and emotional
    manipulation.

    These features help quantify propaganda indicators and are designed
    to integrate with the TruthLens feature extraction framework through
    BaseFeature and FeatureRegistry.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing text and optional tokens

Outputs:
    Dict[str, float] representing propaganda-related signals
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
# Propaganda Lexicons
# ---------------------------------------------------------

# ---------------------------------------------------------
# Name Calling / Labeling Opponents
# ---------------------------------------------------------

NAME_CALLING: Set[str] = {
    "traitor", "enemy", "corrupt", "liar", "lies",
    "radical", "extremist", "fraud", "criminal",
    "evil", "disgrace", "scandal", "dishonest",
    "crooked", "fake", "propagandist",
    "hypocrite", "thug", "terrorist",
    "dictator", "tyrant", "puppet",
    "stooge", "conspirator"
}


# ---------------------------------------------------------
# Fear Appeal Framing
# ---------------------------------------------------------

FEAR_APPEAL: Set[str] = {
    "danger", "dangerous", "threat", "threatening",
    "crisis", "terror", "terrorist",
    "collapse", "destroy", "destruction",
    "attack", "attacks", "invasion",
    "risk", "catastrophe", "catastrophic",
    "disaster", "chaos", "panic",
    "violence", "war", "conflict",
    "instability", "emergency"
}


# ---------------------------------------------------------
# Exaggeration / Absolutist Language
# ---------------------------------------------------------

EXAGGERATION: Set[str] = {
    "always", "never", "everyone", "everybody",
    "nobody", "noone", "all", "none",
    "completely", "absolutely", "totally",
    "entirely", "withoutdoubt",
    "undeniable", "guaranteed",
    "inevitable", "certainly",
    "perfectly", "entirely"
}


# ---------------------------------------------------------
# Glittering Generalities
# ---------------------------------------------------------

GLITTERING_GENERALITIES: Set[str] = {
    "freedom", "democracy", "justice",
    "patriotism", "honor", "truth",
    "liberty", "rights", "unity",
    "integrity", "prosperity",
    "progress", "peace",
    "security", "strength",
    "values", "principles",
    "fairness", "equality"
}


# ---------------------------------------------------------
# Us vs Them Polarization
# ---------------------------------------------------------

US_VS_THEM: Set[str] = {
    "they", "them", "their",
    "those", "others",
    "outsiders", "foreigners",
    "immigrants", "migrants",
    "elites", "globalists",
    "establishment",
    "bureaucrats", "corporations",
    "politicians"
}


# ---------------------------------------------------------
# Authority Appeal
# ---------------------------------------------------------

AUTHORITY_APPEAL: Set[str] = {
    "experts", "scientists", "researchers",
    "leaders", "officials",
    "authorities", "government",
    "intelligence", "agencies",
    "analysts", "investigators",
    "reports", "studies",
    "evidence", "data"
}


# ---------------------------------------------------------
# Emotional Intensifiers
# ---------------------------------------------------------

INTENSIFIERS: Set[str] = {
    "very", "extremely", "incredibly",
    "deeply", "seriously",
    "highly", "strongly",
    "massively", "tremendously",
    "remarkably", "exceptionally",
    "particularly", "greatly"
}

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    """Compute normalized frequency of lexicon terms."""
    if total == 0:
        return 0.0

    count = sum(counter.get(w, 0) for w in lexicon)
    return count / total


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class PropagandaFeatures(BaseFeature):

    """
    Extract lexical propaganda indicators.

    Output Features
    ---------------

    propaganda_name_calling_ratio
    propaganda_fear_ratio
    propaganda_exaggeration_ratio
    propaganda_glitter_ratio
    propaganda_us_vs_them_ratio
    propaganda_authority_ratio
    propaganda_intensifier_ratio

    propaganda_exclamation_density
    propaganda_caps_ratio

    propaganda_intensity
    propaganda_diversity
    """

    name: str = "propaganda_features"
    description: str = "Lexicon-based propaganda detection features"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        text = context.text

        tokens = context.tokens or _tokenize(text)

        if not tokens:
            logger.warning("No tokens available for propaganda feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        name_ratio = _ratio(counter, NAME_CALLING, total_tokens)
        fear_ratio = _ratio(counter, FEAR_APPEAL, total_tokens)
        exaggeration_ratio = _ratio(counter, EXAGGERATION, total_tokens)
        glitter_ratio = _ratio(counter, GLITTERING_GENERALITIES, total_tokens)
        us_vs_them_ratio = _ratio(counter, US_VS_THEM, total_tokens)
        authority_ratio = _ratio(counter, AUTHORITY_APPEAL, total_tokens)
        intensifier_ratio = _ratio(counter, INTENSIFIERS, total_tokens)

        # -------------------------------------------------
        # Structural propaganda signals
        # -------------------------------------------------

        exclamation_density = text.count("!") / max(len(text), 1)

        caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
        caps_ratio = caps_words / total_tokens

        values = [
            name_ratio,
            fear_ratio,
            exaggeration_ratio,
            glitter_ratio,
            us_vs_them_ratio,
            authority_ratio,
            intensifier_ratio,
        ]

        intensity = sum(values) / len(values)
        diversity = sum(1 for v in values if v > 0) / len(values)

        features: Dict[str, float] = {

            "propaganda_name_calling_ratio": name_ratio,
            "propaganda_fear_ratio": fear_ratio,
            "propaganda_exaggeration_ratio": exaggeration_ratio,
            "propaganda_glitter_ratio": glitter_ratio,
            "propaganda_us_vs_them_ratio": us_vs_them_ratio,
            "propaganda_authority_ratio": authority_ratio,
            "propaganda_intensifier_ratio": intensifier_ratio,

            "propaganda_exclamation_density": exclamation_density,
            "propaganda_caps_ratio": caps_ratio,

            "propaganda_intensity": intensity,
            "propaganda_diversity": diversity,
        }

        logger.debug(
            "Propaganda features extracted | intensity=%.4f diversity=%.4f",
            intensity,
            diversity,
        )

        return features