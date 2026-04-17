"""
File Name: narrative_role_features.py
Module: Feature Engineering - Narrative Role Features
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


HERO_TERMS: Set[str] = {
    "hero", "leader", "defender", "protector", "champion",
    "rescued", "saved", "helped", "supported", "defended",
    "guardian", "advocate", "ally", "reformer",
}
VILLAIN_TERMS: Set[str] = {
    "villain", "enemy", "corrupt", "attacker", "threat",
    "criminal", "traitor", "abuser", "oppressor",
    "manipulator", "aggressor", "tyrant",
}
VICTIM_TERMS: Set[str] = {
    "victim", "suffer", "injured", "attacked",
    "abused", "affected", "harmed", "targeted",
    "displaced", "oppressed", "hurt",
}
POLARIZATION_TERMS: Set[str] = {
    "us", "them", "enemy", "opponent", "outsiders",
    "elite", "establishment", "radicals", "extremists",
}


@dataclass
@register_feature
class NarrativeRoleFeatures(BaseFeature):
    name: str = "narrative_role_features"
    description: str = "Hero / villain / victim narrative role signals"

    _nlp: Any = field(default=None, init=False, repr=False)
    _spacy_available: bool = field(default=False, init=False, repr=False)

    def initialize(self) -> None:
        if self._nlp is not None or self._spacy_available:
            return
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self._spacy_available = True
        except Exception:  # noqa: BLE001
            self._nlp = None
            self._spacy_available = False
            logger.warning("spaCy unavailable. NarrativeRoleFeatures using lexical fallback.")

    def _entity_density(self, text: str) -> float:
        self.initialize()
        if not self._spacy_available or self._nlp is None:
            return 0.0

        doc = self._nlp(text)
        entity_count = len(doc.ents)
        token_count = max(len(doc), 1)
        return entity_count / token_count

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")
        if not context.text.strip():
            return {}

        tokens = context.tokens or _tokenize(context.text)
        if not tokens:
            logger.warning("No tokens available for narrative role extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            return sum(counter.get(w, 0) for w in lexicon) / total_tokens

        hero_ratio = ratio(HERO_TERMS)
        villain_ratio = ratio(VILLAIN_TERMS)
        victim_ratio = ratio(VICTIM_TERMS)
        polarization_ratio = ratio(POLARIZATION_TERMS)

        roles = [hero_ratio, villain_ratio, victim_ratio]
        role_diversity = sum(1 for r in roles if r > 0) / len(roles)
        role_balance = 1.0 - (max(roles) - min(roles))

        entity_density = self._entity_density(context.text)

        features: Dict[str, float] = {
            "narrative_role_hero_ratio": float(hero_ratio),
            "narrative_role_villain_ratio": float(villain_ratio),
            "narrative_role_victim_ratio": float(victim_ratio),
            "narrative_role_polarization_ratio": float(polarization_ratio),
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