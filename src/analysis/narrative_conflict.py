# src/analysis/narrative_conflict.py

from __future__ import annotations

import logging
from typing import Dict, Optional, List, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import NARRATIVE_CONFLICT_KEYS, make_vector

logger = logging.getLogger(__name__)


class NarrativeConflictAnalyzer(BaseAnalyzer):

    CONFLICT_VERBS: Set[str] = {
        "attack","assault","strike","bomb","invade","raid",
        "kill","destroy","eliminate","retaliate","counterattack",
        "fight","battle","clash",
        "oppose","challenge","confront","block","resist",
        "defy","undermine","overthrow","topple",
        "accuse","blame","criticize","condemn","denounce",
        "slam","rebuke","mock","dismiss",
        "threaten","warn","pressure","intimidate","coerce",
        "sue","investigate","prosecute","sanction","charge","impeach"
    }

    OPPOSITION_MARKERS: Set[str] = {
        "against","versus","vs","opposed","opposing",
        "conflict","confrontation","showdown","standoff",
        "rival","rivalry","competitor","adversary",
        "struggle","battle","fight","clash",
        "ideological clash","power struggle","political fight",
    }

    POLARIZATION_TERMS: Set[str] = {
        "us","we","our","ours",
        "them","they","their","others",
        "enemy","opponent","adversary",
        "elite","establishment","globalists",
        "extremists","radicals",
        "the people","ordinary people","corrupt elites",
    }

    def __init__(self):
        self.conflict_verbs = normalize_lexicon_terms(self.CONFLICT_VERBS)
        self.opposition = normalize_lexicon_terms(self.OPPOSITION_MARKERS)
        self.polarization = normalize_lexicon_terms(self.POLARIZATION_TERMS)

        logger.info("NarrativeConflictAnalyzer initialized (optimized)")

    # -----------------------------------------------------

    def analyze(
        self,
        ctx: FeatureContext,
        hero_entities: Optional[List[str]] = None,
        villain_entities: Optional[List[str]] = None,
        victim_entities: Optional[List[str]] = None,
    ) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty()

        features: Dict[str, float] = {}

        # 🔥 conflict verbs
        features["conflict_verb_ratio"] = self._conflict_verbs(ctx)

        # 🔥 opposition + polarization
        features["opposition_marker_ratio"] = self._hits(ctx, self.opposition)
        features["polarization_ratio"] = self._hits(ctx, self.polarization)

        # 🔥 actor structure
        features.update(
            self._actor_structure(
                ctx,
                hero_entities,
                villain_entities,
                victim_entities,
            )
        )

        # 🔥 punctuation
        features["conflict_exclamation_ratio"] = self._punctuation(ctx, "!")
        features["conflict_question_ratio"] = self._punctuation(ctx, "?")

        return features

    # -----------------------------------------------------

    def _conflict_verbs(self, ctx: FeatureContext) -> float:

        verbs = [t for t in ctx.doc if t.pos_ == "VERB"]

        if not verbs:
            return 0.0

        count = sum(
            1 for v in verbs if v.lemma_.lower() in self.conflict_verbs
        )

        return float(count / len(verbs))

    # -----------------------------------------------------

    def _hits(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        token_hits = sum(
            ctx.token_counts.get(term, 0)
            for term in lexicon
            if " " not in term
        )

        phrase_hits = phrase_match_count(ctx.text_lower, lexicon)

        return float((token_hits + phrase_hits) / max(ctx.n_tokens, 1))

    # -----------------------------------------------------

    def _actor_structure(
        self,
        ctx: FeatureContext,
        heroes: Optional[List[str]],
        villains: Optional[List[str]],
        victims: Optional[List[str]],
    ) -> Dict[str, float]:

        heroes = heroes or []
        villains = villains or []
        victims = victims or []

        text = ctx.text_lower

        hero_mentions = sum(text.count(h.lower()) for h in heroes)
        villain_mentions = sum(text.count(v.lower()) for v in villains)
        victim_mentions = sum(text.count(v.lower()) for v in victims)

        return {
            "hero_villain_victim_ratio":
                float(min(hero_mentions, villain_mentions) +
                      min(villain_mentions, victim_mentions)),
        }

    # -----------------------------------------------------

    def _punctuation(self, ctx: FeatureContext, symbol: str) -> float:
        count = ctx.text_lower.count(symbol)
        return float(count / max(ctx.n_tokens, 1))

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "conflict_verb_ratio": 0.0,
            "opposition_marker_ratio": 0.0,
            "polarization_ratio": 0.0,
            "hero_villain_victim_ratio": 0.0,
            "rhetorical_punctuation_ratio": 0.0,
        }


# ---------------------------------------------------------
# Vector
# ---------------------------------------------------------

def narrative_conflict_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, NARRATIVE_CONFLICT_KEYS)