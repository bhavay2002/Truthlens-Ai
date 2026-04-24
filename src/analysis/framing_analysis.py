# src/analysis/framing_analysis.py

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    term_ratio,
    phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import FRAMING_KEYS, make_vector

logger = logging.getLogger(__name__)


class FramingAnalyzer(BaseAnalyzer):

    CONFLICT_TERMS = {
        "conflict","fight","battle","war","clash","attack","confront",
        "dispute","rival","struggle","tension","hostility","standoff",
        "confrontation","showdown","retaliation","counterattack",
        "escalation","political fight","power struggle","ideological clash"
    }

    ECONOMIC_TERMS = {
        "economy","economic","market","markets",
        "jobs","employment","unemployment","labor",
        "tax","taxes","trade","budget","spending",
        "cost","financial","finance","growth",
        "inflation","investment","recession",
        "deficit","debt","revenue","funding",
        "economic growth","economic policy","fiscal policy"
    }

    MORAL_TERMS = {
        "moral","ethic","ethics","value","values",
        "justice","fairness","right","wrong",
        "duty","principle","virtue","integrity",
        "honor","honour","conscience","morality",
        "ethical","responsibility","moral obligation",
        "social justice","human rights"
    }

    HUMAN_INTEREST_TERMS = {
        "family","children","child","community",
        "people","citizen","victim","life",
        "story","personal story","emotion",
        "suffering","experience","personal",
        "struggle","hardship","tragedy",
        "survivor","human impact","daily life"
    }

    SECURITY_TERMS = {
        "security","national security","safety",
        "threat","risk","danger","crisis",
        "terror","terrorism","extremism",
        "attack","defense","protection",
        "surveillance","law enforcement",
        "border security","military",
        "counterterrorism","emergency",
        "public safety"
    }

    BASE_KEYS = [
        "frame_conflict_score",
        "frame_economic_score",
        "frame_moral_score",
        "frame_human_interest_score",
        "frame_security_score",
    ]

    def __init__(self):
        # 🔥 Normalize once
        self.conflict = normalize_lexicon_terms(self.CONFLICT_TERMS)
        self.economic = normalize_lexicon_terms(self.ECONOMIC_TERMS)
        self.moral = normalize_lexicon_terms(self.MORAL_TERMS)
        self.human = normalize_lexicon_terms(self.HUMAN_INTEREST_TERMS)
        self.security = normalize_lexicon_terms(self.SECURITY_TERMS)

        logger.info("FramingAnalyzer initialized (optimized)")

    # ------------------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty_features()

        features: Dict[str, float] = {}

        # 🔥 Hybrid matching (token + phrase)
        features["frame_conflict_score"] = self._score(ctx, self.conflict)
        features["frame_economic_score"] = self._score(ctx, self.economic)
        features["frame_moral_score"] = self._score(ctx, self.moral)
        features["frame_human_interest_score"] = self._score(ctx, self.human)
        features["frame_security_score"] = self._score(ctx, self.security)

        features.update(self._frame_dominance(features))
        features.update(self._frame_diversity(features))

        return features

    # ------------------------------------------------------------

    def _score(self, ctx: FeatureContext, lexicon: set) -> float:

        # token-level
        token_score = term_ratio(
            ctx.token_counts,
            ctx.n_tokens,
            lexicon,
        )

        # phrase-level
        phrase_hits = phrase_match_count(
            ctx.text_lower,
            lexicon,
        )

        phrase_score = phrase_hits / max(ctx.n_tokens, 1)

        return float(token_score + phrase_score)

    # ------------------------------------------------------------

    def _frame_dominance(self, features: Dict[str, float]) -> Dict[str, float]:

        scores = [features.get(k, 0.0) for k in self.BASE_KEYS]

        return {
            "frame_dominance_score": float(max(scores)) if scores else 0.0
        }

    # ------------------------------------------------------------

    def _frame_diversity(self, features: Dict[str, float]) -> Dict[str, float]:

        scores = [features.get(k, 0.0) for k in self.BASE_KEYS]

        active = sum(1 for s in scores if s > 0)

        return {
            "frame_diversity_score": float(active / len(scores))
        }

    # ------------------------------------------------------------

    def _empty_features(self) -> Dict[str, float]:
        return {
            "frame_conflict_score": 0.0,
            "frame_economic_score": 0.0,
            "frame_moral_score": 0.0,
            "frame_human_interest_score": 0.0,
            "frame_security_score": 0.0,
            "frame_dominance_score": 0.0,
            "frame_diversity_score": 0.0,
        }


# ------------------------------------------------------------
# Vector conversion
# ------------------------------------------------------------

def framing_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, FRAMING_KEYS)