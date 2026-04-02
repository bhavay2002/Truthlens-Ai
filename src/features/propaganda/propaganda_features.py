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
    "danger", "threat", "crisis", "terror",
    "collapse", "destroy", "attack", "invasion"
}

EXAGGERATION: Set[str] = {
    "always", "never", "everyone", "nobody",
    "completely", "absolutely", "totally"
}

GLITTERING_GENERALITIES: Set[str] = {
    "freedom", "democracy", "justice",
    "patriotism", "honor", "truth"
}

US_VS_THEM: Set[str] = {
    "they", "them", "their", "those",
    "outsiders", "others"
}


@dataclass
@register_feature
class PropagandaFeatures(BaseFeature):
    """
    Extracts propaganda indicators from lexical cues.

    Output Features
    ---------------
    propaganda_name_calling_ratio
    propaganda_fear_ratio
    propaganda_exaggeration_ratio
    propaganda_glitter_ratio
    propaganda_us_vs_them_ratio
    propaganda_intensity
    propaganda_diversity
    """

    name: str = "propaganda_features"
    description: str = "Lexicon-based propaganda indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for propaganda feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        name_ratio = ratio(NAME_CALLING)
        fear_ratio = ratio(FEAR_APPEAL)
        exaggeration_ratio = ratio(EXAGGERATION)
        glitter_ratio = ratio(GLITTERING_GENERALITIES)
        us_vs_them_ratio = ratio(US_VS_THEM)

        values = [
            name_ratio,
            fear_ratio,
            exaggeration_ratio,
            glitter_ratio,
            us_vs_them_ratio,
        ]

        intensity = sum(values) / len(values)
        diversity = sum(1 for v in values if v > 0) / len(values)

        features: Dict[str, float] = {
            "propaganda_name_calling_ratio": float(name_ratio),
            "propaganda_fear_ratio": float(fear_ratio),
            "propaganda_exaggeration_ratio": float(exaggeration_ratio),
            "propaganda_glitter_ratio": float(glitter_ratio),
            "propaganda_us_vs_them_ratio": float(us_vs_them_ratio),
            "propaganda_intensity": float(intensity),
            "propaganda_diversity": float(diversity),
        }

        logger.debug(
            "Propaganda features extracted | intensity=%.4f diversity=%.4f",
            intensity,
            diversity,
        )

        return features