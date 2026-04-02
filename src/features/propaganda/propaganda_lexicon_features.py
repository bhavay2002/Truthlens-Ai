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


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------
# Propaganda Lexicons
# ---------------------------------------------------------------------

NAME_CALLING: Set[str] = {
    "traitor", "enemy", "corrupt", "liar", "radical",
    "extremist", "fraud", "criminal"
}

FEAR_APPEAL: Set[str] = {
    "danger", "threat", "terror", "crisis",
    "attack", "invasion", "collapse"
}

EXAGGERATION: Set[str] = {
    "always", "never", "everyone", "nobody",
    "completely", "absolutely", "totally"
}

BANDWAGON: Set[str] = {
    "everyone", "majority", "all people",
    "the nation", "the people"
}

SLOGANS: Set[str] = {
    "freedom", "democracy", "justice",
    "patriotism", "honor", "truth"
}


@dataclass
@register_feature
class PropagandaLexiconFeatures(BaseFeature):
    """
    Extracts propaganda signals based on lexicon matching.

    Output Features
    ---------------
    propaganda_name_calling_ratio
    propaganda_fear_ratio
    propaganda_exaggeration_ratio
    propaganda_bandwagon_ratio
    propaganda_slogan_ratio
    propaganda_lexicon_density
    propaganda_lexicon_diversity
    """

    name: str = "propaganda_lexicon_features"
    description: str = "Lexicon-based propaganda detection features"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for propaganda lexicon extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        name_ratio = ratio(NAME_CALLING)
        fear_ratio = ratio(FEAR_APPEAL)
        exaggeration_ratio = ratio(EXAGGERATION)
        bandwagon_ratio = ratio(BANDWAGON)
        slogan_ratio = ratio(SLOGANS)

        counts = [
            sum(counter.get(w, 0) for w in NAME_CALLING),
            sum(counter.get(w, 0) for w in FEAR_APPEAL),
            sum(counter.get(w, 0) for w in EXAGGERATION),
            sum(counter.get(w, 0) for w in BANDWAGON),
            sum(counter.get(w, 0) for w in SLOGANS),
        ]

        total_prop_tokens = sum(counts)
        density = total_prop_tokens / total_tokens
        diversity = sum(1 for c in counts if c > 0) / len(counts)

        features: Dict[str, float] = {
            "propaganda_name_calling_ratio": float(name_ratio),
            "propaganda_fear_ratio": float(fear_ratio),
            "propaganda_exaggeration_ratio": float(exaggeration_ratio),
            "propaganda_bandwagon_ratio": float(bandwagon_ratio),
            "propaganda_slogan_ratio": float(slogan_ratio),
            "propaganda_lexicon_density": float(density),
            "propaganda_lexicon_diversity": float(diversity),
        }

        logger.debug(
            "Propaganda lexicon features extracted | density=%.4f diversity=%.4f",
            density,
            diversity,
        )

        return features