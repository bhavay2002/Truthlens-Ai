"""
File Name: narrative_role_features.py
Module: Feature Engineering - Narrative Role Features
Description:
    Extracts narrative role indicators from text. The module attempts to
    identify linguistic signals that assign narrative roles such as
    hero, villain, and victim within storytelling or political discourse.

    The implementation combines lightweight lexical cues with optional
    named entity detection (via spaCy when available) to estimate how
    narrative roles are distributed and whether the text frames actors
    positively or negatively.

    These features are useful for narrative analysis, bias detection,
    propaganda identification, and discourse modeling.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections
    spacy (optional)

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing narrative role indicators
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

try:
    import spacy

    _NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:  # noqa: BLE001
    _NLP = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. NarrativeRoleFeatures using lexical fallback.")


def _tokenize(text: str) -> List[str]:
    """Basic tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------
# Narrative Role Lexicons
# ---------------------------------------------------------------------

HERO_TERMS: Set[str] = {
    "hero",
    "leader",
    "defender",
    "protector",
    "rescued",
    "saved",
    "champion",
}

VILLAIN_TERMS: Set[str] = {
    "villain",
    "enemy",
    "corrupt",
    "attacker",
    "threat",
    "criminal",
    "traitor",
}

VICTIM_TERMS: Set[str] = {
    "victim",
    "suffer",
    "injured",
    "attacked",
    "abused",
    "affected",
    "harmed",
}


@dataclass
@register_feature
class NarrativeRoleFeatures(BaseFeature):
    """
    Extract narrative role indicators.

    Output Features
    ---------------
    narrative_role_hero_ratio
    narrative_role_villain_ratio
    narrative_role_victim_ratio
    narrative_role_balance
    narrative_role_diversity
    narrative_entity_density
    """

    name: str = "narrative_role_features"
    description: str = "Hero / villain / victim narrative role signals"

    def _entity_density(self, text: str) -> float:
        """Estimate entity density using spaCy if available."""
        if not SPACY_AVAILABLE:
            return 0.0

        doc = _NLP(text)
        entity_count = len(doc.ents)
        token_count = max(len(doc), 1)

        return entity_count / token_count

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """Extract narrative role features."""
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for narrative role extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        hero_ratio = ratio(HERO_TERMS)
        villain_ratio = ratio(VILLAIN_TERMS)
        victim_ratio = ratio(VICTIM_TERMS)

        roles = [hero_ratio, villain_ratio, victim_ratio]

        role_diversity = sum(1 for r in roles if r > 0) / len(roles)

        role_balance = 1.0 - max(roles)

        entity_density = self._entity_density(context.text)

        features: Dict[str, float] = {
            "narrative_role_hero_ratio": float(hero_ratio),
            "narrative_role_villain_ratio": float(villain_ratio),
            "narrative_role_victim_ratio": float(victim_ratio),
            "narrative_role_balance": float(role_balance),
            "narrative_role_diversity": float(role_diversity),
            "narrative_entity_density": float(entity_density),
        }

        logger.debug(
            "Narrative role features extracted | hero=%.4f villain=%.4f victim=%.4f",
            hero_ratio,
            villain_ratio,
            victim_ratio,
        )

        return features