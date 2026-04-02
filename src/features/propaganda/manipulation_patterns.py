"""
File Name: manipulation_patterns.py
Module: Feature Engineering - Propaganda / Manipulation Patterns
Description:
    Detects linguistic manipulation patterns commonly used in propaganda,
    persuasive messaging, and misinformation. The module identifies patterns
    such as emotional manipulation, urgency framing, blame attribution,
    scapegoating, rhetorical exaggeration, and false dilemmas.

    These signals help characterize how language attempts to influence
    perception or decision-making rather than simply convey information.

    The implementation is lightweight and deterministic, relying on curated
    pattern lexicons and structural heuristics. It integrates with the
    TruthLens feature framework via BaseFeature and FeatureRegistry.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing text and optional tokens

Outputs:
    Dict[str, float] representing manipulation pattern indicators
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
    """Basic tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------
# Manipulation Pattern Lexicons
# ---------------------------------------------------------------------

URGENCY_TERMS: Set[str] = {
    "urgent", "immediately", "now", "crisis", "emergency",
    "act", "instant", "before it's too late"
}

BLAME_TERMS: Set[str] = {
    "blame", "fault", "responsible", "caused", "betrayed",
    "failed", "destroyed"
}

SCAPEGOATING_TERMS: Set[str] = {
    "they", "them", "their", "outsiders", "immigrants",
    "elites", "media", "establishment"
}

ABSOLUTE_TERMS: Set[str] = {
    "always", "never", "everyone", "nobody", "all", "none"
}

FEAR_MANIPULATION_TERMS: Set[str] = {
    "threat", "danger", "attack", "collapse",
    "disaster", "catastrophe"
}


@dataclass
@register_feature
class ManipulationPatterns(BaseFeature):
    """
    Detects rhetorical manipulation patterns.

    Output Features
    ---------------
    manipulation_urgency_ratio
    manipulation_blame_ratio
    manipulation_scapegoat_ratio
    manipulation_absolute_ratio
    manipulation_fear_ratio
    manipulation_intensity
    manipulation_diversity
    """

    name: str = "manipulation_patterns"
    description: str = "Rhetorical manipulation and persuasion indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """Extract manipulation-related features."""
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for manipulation feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        urgency_ratio = ratio(URGENCY_TERMS)
        blame_ratio = ratio(BLAME_TERMS)
        scapegoat_ratio = ratio(SCAPEGOATING_TERMS)
        absolute_ratio = ratio(ABSOLUTE_TERMS)
        fear_ratio = ratio(FEAR_MANIPULATION_TERMS)

        values = [
            urgency_ratio,
            blame_ratio,
            scapegoat_ratio,
            absolute_ratio,
            fear_ratio,
        ]

        intensity = sum(values) / len(values)
        diversity = sum(1 for v in values if v > 0) / len(values)

        features: Dict[str, float] = {
            "manipulation_urgency_ratio": float(urgency_ratio),
            "manipulation_blame_ratio": float(blame_ratio),
            "manipulation_scapegoat_ratio": float(scapegoat_ratio),
            "manipulation_absolute_ratio": float(absolute_ratio),
            "manipulation_fear_ratio": float(fear_ratio),
            "manipulation_intensity": float(intensity),
            "manipulation_diversity": float(diversity),
        }

        logger.debug(
            "Manipulation patterns extracted | intensity=%.4f diversity=%.4f",
            intensity,
            diversity,
        )

        return features