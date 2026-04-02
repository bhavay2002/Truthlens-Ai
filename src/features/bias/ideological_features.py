"""
File Name: ideological_features.py
Module: Feature Engineering - Ideological Features
Description:
    Extracts ideology-related linguistic signals from text by matching tokens
    against curated ideological lexicons and computing distributional metrics.
    The module produces normalized ratios for left/right ideological cues,
    polarization indicators, and balance metrics. Designed to be lightweight,
    deterministic, and compatible with the TruthLens feature pipeline.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing ideology and polarization indicators
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
# Ideological Lexicons (lightweight heuristics; extendable via config)
# ---------------------------------------------------------------------

LEFT_LEXICON: Set[str] = {
    "equality", "progressive", "climate", "regulation", "welfare",
    "diversity", "inclusion", "redistribution", "public", "universal",
    "labor", "union", "social", "equity", "justice"
}

RIGHT_LEXICON: Set[str] = {
    "freedom", "liberty", "market", "tax", "security",
    "patriot", "tradition", "border", "sovereignty",
    "private", "deregulation", "military", "law", "order"
}

POLARIZING_TERMS: Set[str] = {
    "elite", "establishment", "radical", "extremist",
    "corrupt", "enemy", "attack", "threat"
}

GROUP_REFERENCES: Set[str] = {
    "they", "them", "those", "these", "people", "group"
}


@dataclass
@register_feature
class IdeologicalFeatures(BaseFeature):
    """
    Extract ideology and polarization indicators.

    Output Features
    ---------------
    ideology_left_ratio
    ideology_right_ratio
    ideology_balance
    ideology_polarization_ratio
    ideology_group_reference_ratio
    ideology_signal_strength
    """

    name: str = "ideological_features"
    description: str = "Ideological framing and polarization signals"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for ideological feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        left_ratio = ratio(LEFT_LEXICON)
        right_ratio = ratio(RIGHT_LEXICON)
        polarization_ratio = ratio(POLARIZING_TERMS)
        group_ref_ratio = ratio(GROUP_REFERENCES)

        # Balance: closeness of left/right ratios (lower difference = more balanced)
        balance = 1.0 - abs(left_ratio - right_ratio)

        signal_strength = (left_ratio + right_ratio + polarization_ratio) / 3.0

        features: Dict[str, float] = {
            "ideology_left_ratio": float(left_ratio),
            "ideology_right_ratio": float(right_ratio),
            "ideology_balance": float(balance),
            "ideology_polarization_ratio": float(polarization_ratio),
            "ideology_group_reference_ratio": float(group_ref_ratio),
            "ideology_signal_strength": float(signal_strength),
        }

        logger.debug(
            "Ideological features extracted | left=%.4f right=%.4f balance=%.4f",
            left_ratio,
            right_ratio,
            balance,
        )

        return features