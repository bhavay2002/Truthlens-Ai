"""
File Name: conflict_features.py
Module: Feature Engineering - Conflict Features
Description:
    Extracts linguistic indicators of conflict, confrontation, and adversarial
    discourse within text. These features help identify narrative escalation,
    argumentative tone, and polarized framing often present in political,
    journalistic, or propagandistic content.

    The module uses curated lexicons and structural heuristics to quantify
    the presence of conflict-related language such as attacks, disputes,
    accusations, and aggressive rhetoric. These signals contribute to
    downstream models analyzing narrative dynamics and media framing.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing conflict-related discourse indicators
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
    """Basic tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------
# Conflict Lexicons
# ---------------------------------------------------------------------

CONFRONTATION_TERMS: Set[str] = {
    "fight",
    "battle",
    "clash",
    "attack",
    "war",
    "confront",
    "struggle",
}

DISPUTE_TERMS: Set[str] = {
    "dispute",
    "argument",
    "debate",
    "disagreement",
    "controversy",
    "criticized",
}

ACCUSATION_TERMS: Set[str] = {
    "accuse",
    "blame",
    "fault",
    "responsible",
    "allege",
    "charged",
}

AGGRESSIVE_LANGUAGE: Set[str] = {
    "destroy",
    "defeat",
    "threat",
    "enemy",
    "hostile",
    "attack",
}


@dataclass
@register_feature
class ConflictFeatures(BaseFeature):
    """
    Extracts indicators of conflict-oriented discourse.

    Output Features
    ---------------
    conflict_confrontation_ratio
    conflict_dispute_ratio
    conflict_accusation_ratio
    conflict_aggression_ratio
    conflict_intensity
    conflict_diversity
    """

    name: str = "conflict_features"
    description: str = "Conflict and confrontation discourse indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """Extract conflict-related features."""
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for conflict feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        confrontation_ratio = ratio(CONFRONTATION_TERMS)
        dispute_ratio = ratio(DISPUTE_TERMS)
        accusation_ratio = ratio(ACCUSATION_TERMS)
        aggression_ratio = ratio(AGGRESSIVE_LANGUAGE)

        values = [
            confrontation_ratio,
            dispute_ratio,
            accusation_ratio,
            aggression_ratio,
        ]

        intensity = sum(values) / len(values)
        diversity = sum(1 for v in values if v > 0) / len(values)

        features: Dict[str, float] = {
            "conflict_confrontation_ratio": float(confrontation_ratio),
            "conflict_dispute_ratio": float(dispute_ratio),
            "conflict_accusation_ratio": float(accusation_ratio),
            "conflict_aggression_ratio": float(aggression_ratio),
            "conflict_intensity": float(intensity),
            "conflict_diversity": float(diversity),
        }

        logger.debug(
            "Conflict features extracted | intensity=%.4f diversity=%.4f",
            intensity,
            diversity,
        )

        return features