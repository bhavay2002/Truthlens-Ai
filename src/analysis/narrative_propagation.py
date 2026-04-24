# src/analysis/narrative_propagation.py

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import NARRATIVE_PROPAGATION_KEYS, make_vector

logger = logging.getLogger(__name__)


class NarrativePropagationAnalyzer(BaseAnalyzer):

    CONFLICT_VERBS = {
        "violent_conflict": {
            "attack","assault","strike","bomb","invade","raid",
            "kill","destroy","eliminate","retaliate","counterattack",
            "fight","battle","clash"
        },
        "political_conflict": {
            "oppose","challenge","confront","block","resist",
            "defy","undermine","topple","overthrow"
        },
        "discursive_conflict": {
            "accuse","blame","criticize","condemn","denounce",
            "slam","rebuke","mock","dismiss"
        },
        "institutional_conflict": {
            "sue","investigate","prosecute","charge","sanction","impeach"
        },
        "coercion_conflict": {
            "threaten","warn","pressure","intimidate","coerce"
        },
    }

    OPPOSITION_MARKERS = {
        "against","versus","vs","opposed","opposing",
        "conflict","confrontation","showdown","standoff",
        "rival","rivalry","competitor","adversary",
        "struggle","battle","fight","clash",
    }

    POLARIZATION_TERMS = {
        "us","we","our","ours",
        "them","they","their","others",
        "enemy","opponent","adversary",
        "elite","establishment","globalists",
        "extremists","radicals",
    }

    CONFLICT_PHRASES = {
        "war against","fight against","battle against",
        "clash with","conflict with","power struggle",
        "political fight","ideological battle",
        "direct confrontation","rising tensions",
        "growing conflict",
    }

    # -----------------------------------------------------

    def __init__(self):

        # 🔥 normalize ONCE
        self.conflict_verbs = {
            k: normalize_lexicon_terms(v)
            for k, v in self.CONFLICT_VERBS.items()
        }

        self.opposition = normalize_lexicon_terms(self.OPPOSITION_MARKERS)
        self.polarization = normalize_lexicon_terms(self.POLARIZATION_TERMS)
        self.conflict_phrases = normalize_lexicon_terms(self.CONFLICT_PHRASES)

        logger.info("NarrativePropagationAnalyzer initialized (optimized)")

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

        features.update(self._conflict_verb_features(ctx))
        features.update(self._opposition(ctx))
        features.update(self._polarization(ctx))
        features.update(self._conflict_phrase_features(ctx))
        features.update(
            self._actor_roles(
                ctx,
                hero_entities,
                villain_entities,
                victim_entities,
            )
        )
        features.update(self._punctuation(ctx))

        return features

    # -----------------------------------------------------

    def _conflict_verb_features(self, ctx: FeatureContext) -> Dict[str, float]:

        features = {}
        total = max(ctx.n_tokens, 1)

        for category, lexicon in self.conflict_verbs.items():
            count = sum(ctx.token_counts.get(t, 0) for t in lexicon)
            features[f"{category}_ratio"] = float(count / total)

        return features

    # -----------------------------------------------------

    def _opposition(self, ctx: FeatureContext) -> Dict[str, float]:

        count = sum(ctx.token_counts.get(t, 0) for t in self.opposition)
        return {"opposition_marker_ratio": float(count / max(ctx.n_tokens, 1))}

    # -----------------------------------------------------

    def _polarization(self, ctx: FeatureContext) -> Dict[str, float]:

        count = sum(ctx.token_counts.get(t, 0) for t in self.polarization)
        return {"polarization_ratio": float(count / max(ctx.n_tokens, 1))}

    # -----------------------------------------------------

    def _conflict_phrase_features(self, ctx: FeatureContext) -> Dict[str, float]:

        hits = phrase_match_count(ctx.text_lower, self.conflict_phrases)

        return {
            "conflict_phrase_ratio": float(hits / max(ctx.n_tokens, 1))
        }

    # -----------------------------------------------------

    def _actor_roles(
        self,
        ctx: FeatureContext,
        heroes: Optional[List[str]],
        villains: Optional[List[str]],
        victims: Optional[List[str]],
    ) -> Dict[str, float]:

        text = ctx.text_lower

        heroes = heroes or []
        villains = villains or []
        victims = victims or []

        hero_mentions = sum(text.count(h.lower()) for h in heroes)
        villain_mentions = sum(text.count(v.lower()) for v in villains)
        victim_mentions = sum(text.count(v.lower()) for v in victims)

        return {
            "hero_villain_conflict_score":
                float(min(hero_mentions, villain_mentions)),
            "villain_victim_harm_score":
                float(min(villain_mentions, victim_mentions)),
            "hero_victim_protection_score":
                float(min(hero_mentions, victim_mentions)),
        }

    # -----------------------------------------------------

    def _punctuation(self, ctx: FeatureContext) -> Dict[str, float]:

        text = ctx.text_lower

        return {
            "conflict_exclamation_ratio":
                text.count("!") / max(ctx.n_tokens, 1),
            "conflict_question_ratio":
                text.count("?") / max(ctx.n_tokens, 1),
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "violent_conflict_ratio": 0.0,
            "political_conflict_ratio": 0.0,
            "discursive_conflict_ratio": 0.0,
            "institutional_conflict_ratio": 0.0,
            "coercion_conflict_ratio": 0.0,
            "opposition_marker_ratio": 0.0,
            "polarization_ratio": 0.0,
            "conflict_phrase_ratio": 0.0,
            "hero_villain_conflict_score": 0.0,
            "villain_victim_harm_score": 0.0,
            "hero_victim_protection_score": 0.0,
            "conflict_exclamation_ratio": 0.0,
            "conflict_question_ratio": 0.0,
        }


# ---------------------------------------------------------
# Vector
# ---------------------------------------------------------

def narrative_propagation_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, NARRATIVE_PROPAGATION_KEYS)