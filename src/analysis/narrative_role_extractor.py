# src/analysis/narrative_role_extractor.py

from __future__ import annotations

import logging
from typing import Dict, List, Set, Optional

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext

logger = logging.getLogger(__name__)


class NarrativeRoleExtractor(BaseAnalyzer):

    HERO_TERMS = {
        "protect","defend","shield","guard","secure",
        "rescue","save","help","aid","assist","support",
        "lead","guide","champion","advocate",
        "rebuild","restore","stabilize","reform",
        "fight for","stand for","stand with",
    }

    VILLAIN_TERMS = {
        "attack","assault","bomb","invade","raid","strike",
        "kill","destroy","harm","injure",
        "exploit","abuse","oppress","suppress",
        "corrupt","manipulate","undermine",
        "threaten","intimidate","coerce",
        "blame","accuse","condemn",
    }

    VICTIM_TERMS = {
        "hurt","injure","kill","attack","harm",
        "suffer","struggle","endure",
        "lose","damage","destroy",
        "target","persecute",
        "displace","flee","escape",
        "affect","impact","burden",
    }

    # -----------------------------------------------------

    def __init__(self):
        self.hero_terms = self._normalize(self.HERO_TERMS)
        self.villain_terms = self._normalize(self.VILLAIN_TERMS)
        self.victim_terms = self._normalize(self.VICTIM_TERMS)

        logger.info("NarrativeRoleExtractor initialized (optimized)")

    # -----------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, List[str]]:

        hero_entities: Set[str] = set()
        villain_entities: Set[str] = set()
        victim_entities: Set[str] = set()

        for token in ctx.doc:

            lemma = token.lemma_.lower()

            # HERO
            if lemma in self.hero_terms:
                self._assign_roles(token, hero_entities, victim_entities)

            # VILLAIN
            elif lemma in self.villain_terms:
                self._assign_roles(token, villain_entities, victim_entities)

            # VICTIM
            elif lemma in self.victim_terms:
                obj = self._get_object(token)
                if obj:
                    victim_entities.add(obj)

            # Passive victim detection
            if token.dep_ == "nsubjpass":
                entity = self._resolve_entity(token)
                if entity:
                    victim_entities.add(entity)

        return {
            "hero_entities": sorted(hero_entities),
            "villain_entities": sorted(villain_entities),
            "victim_entities": sorted(victim_entities),
        }

    # -----------------------------------------------------

    def _assign_roles(
        self,
        token,
        actor_set: Set[str],
        victim_set: Set[str],
    ):
        subject = self._get_subject(token)
        obj = self._get_object(token)

        if subject:
            actor_set.add(subject)

        if obj:
            victim_set.add(obj)

    # -----------------------------------------------------

    def _get_subject(self, token) -> Optional[str]:
        for child in token.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                return self._resolve_entity(child)
        return None

    # -----------------------------------------------------

    def _get_object(self, token) -> Optional[str]:
        for child in token.children:
            if child.dep_ in {"dobj", "pobj", "obj"}:
                return self._resolve_entity(child)
        return None

    # -----------------------------------------------------

    def _resolve_entity(self, token) -> Optional[str]:

        # Named entity span
        if token.ent_iob_ in {"B", "I"}:
            span = token.doc[token.left_edge.i : token.right_edge.i + 1]
            if span.text.strip():
                return span.text.lower()

        # Compound noun phrases
        if token.pos_ in {"NOUN", "PROPN"}:
            return token.lemma_.lower()

        # Fallback to head
        if token.head and token.head != token:
            return token.head.lemma_.lower()

        return None

    # -----------------------------------------------------

    def _normalize(self, terms: Set[str]) -> Set[str]:
        return {t.replace("_", " ").lower() for t in terms}

    # -----------------------------------------------------

    def role_scores(self, features: Dict[str, List[str]]) -> Dict[str, float]:

        heroes = len(features.get("hero_entities", []))
        villains = len(features.get("villain_entities", []))
        victims = len(features.get("victim_entities", []))

        total = max(heroes + villains + victims, 1)

        return {
            "hero_ratio": heroes / total,
            "villain_ratio": villains / total,
            "victim_ratio": victims / total,
            "hero_vs_villain": heroes - villains,
        }


# ---------------------------------------------------------
# Vector
# ---------------------------------------------------------

def narrative_role_vector(features: Dict[str, List[str]]) -> np.ndarray:

    heroes = len(features.get("hero_entities", []))
    villains = len(features.get("villain_entities", []))
    victims = len(features.get("victim_entities", []))

    return np.array(
        [
            float(heroes),
            float(villains),
            float(victims),
            float(heroes - villains),
        ],
        dtype=np.float32,
    )