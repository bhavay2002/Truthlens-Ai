"""
File Name: narrative_features.py
Module: Feature Engineering - Narrative Features
Description:
    Extracts narrative structure indicators from text. The module detects
    narrative roles, conflict framing, temporal storytelling markers,
    and narrative progression signals commonly found in news reporting
    and political narratives.

    These features help characterize storytelling structure used to
    influence interpretation, such as hero/villain framing, crisis
    escalation, and resolution narratives.

    The implementation is lightweight and lexicon-based with simple
    structural heuristics, allowing deterministic feature extraction
    without requiring heavy NLP models.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing narrative structure indicators
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
# Narrative Lexicons
# ---------------------------------------------------------------------

HERO_TERMS: Set[str] = {
    "hero", "leader", "defender", "champion",
    "protect", "save", "rescue"
}

VILLAIN_TERMS: Set[str] = {
    "villain", "enemy", "corrupt", "attacker",
    "threat", "destroy", "betray"
}

VICTIM_TERMS: Set[str] = {
    "victim", "suffer", "harm", "damage",
    "loss", "injured", "affected"
}

CONFLICT_TERMS: Set[str] = {
    "conflict", "battle", "fight", "clash",
    "dispute", "attack", "war"
}

RESOLUTION_TERMS: Set[str] = {
    "resolve", "agreement", "peace", "solution",
    "settlement", "deal"
}

CRISIS_TERMS: Set[str] = {
    "crisis", "emergency", "disaster",
    "collapse", "panic"
}


@dataclass
@register_feature
class NarrativeFeatures(BaseFeature):
    """
    Extract narrative structure indicators.

    Output Features
    ---------------
    narrative_hero_ratio
    narrative_villain_ratio
    narrative_victim_ratio
    narrative_conflict_ratio
    narrative_resolution_ratio
    narrative_crisis_ratio
    narrative_role_diversity
    narrative_conflict_intensity
    """

    name: str = "narrative_features"
    description: str = "Narrative structure and role framing indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """Extract narrative-related features."""

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for narrative feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        hero_ratio = ratio(HERO_TERMS)
        villain_ratio = ratio(VILLAIN_TERMS)
        victim_ratio = ratio(VICTIM_TERMS)
        conflict_ratio = ratio(CONFLICT_TERMS)
        resolution_ratio = ratio(RESOLUTION_TERMS)
        crisis_ratio = ratio(CRISIS_TERMS)

        role_values = [hero_ratio, villain_ratio, victim_ratio]

        role_diversity = sum(1 for v in role_values if v > 0) / len(role_values)

        conflict_intensity = (conflict_ratio + crisis_ratio) / 2.0

        features: Dict[str, float] = {
            "narrative_hero_ratio": float(hero_ratio),
            "narrative_villain_ratio": float(villain_ratio),
            "narrative_victim_ratio": float(victim_ratio),
            "narrative_conflict_ratio": float(conflict_ratio),
            "narrative_resolution_ratio": float(resolution_ratio),
            "narrative_crisis_ratio": float(crisis_ratio),
            "narrative_role_diversity": float(role_diversity),
            "narrative_conflict_intensity": float(conflict_intensity),
        }

        logger.debug(
            "Narrative features extracted | conflict=%.4f roles=%.4f",
            conflict_intensity,
            role_diversity,
        )

        return features