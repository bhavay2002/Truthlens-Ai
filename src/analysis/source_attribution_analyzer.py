# src/analysis/source_attribution_analyzer.py

from __future__ import annotations

import logging
import re
from typing import Dict, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import term_ratio
from src.analysis.feature_schema import SOURCE_ATTRIBUTION_KEYS, make_vector

logger = logging.getLogger(__name__)


class SourceAttributionAnalyzer(BaseAnalyzer):

    EXPERT_TERMS = {
        "expert","experts","analyst","analysts",
        "researcher","researchers","scientist","scientists",
        "professor","economist","doctor","official",
        "authority","specialist"
    }

    ANONYMOUS_TERMS = {
        "sources","source","insiders","officials",
        "people","critics","observers","commentators",
        "analysts","individuals"
    }

    CREDIBILITY_TERMS = {
        "report","study","research","analysis",
        "data","dataset","statistics",
        "evidence","survey","findings",
        "according","confirmed","documented"
    }

    ATTRIBUTION_VERBS = {
        "say","said","report","reported",
        "state","stated","claim","claimed",
        "explain","explained","note","noted",
        "argue","argued","announce","announced",
        "confirm","confirmed"
    }

    QUOTE_PATTERN = re.compile(r"[\"“”]")

    # -----------------------------------------------------

    def __init__(self):
        logger.info("SourceAttributionAnalyzer initialized (optimized)")

    # -----------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty()

        features: Dict[str, float] = {}

        # 🔥 Token-level ratios
        features["expert_attribution_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.EXPERT_TERMS
        )

        features["anonymous_source_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.ANONYMOUS_TERMS
        )

        features["credibility_indicator_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.CREDIBILITY_TERMS
        )

        features["attribution_verb_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.ATTRIBUTION_VERBS
        )

        # 🔥 Quote density
        features["quotation_ratio"] = self._quote_ratio(ctx)

        # 🔥 Named source detection
        features["named_source_ratio"] = self._entity_ratio(ctx)

        # 🔥 Credibility balance
        features["source_credibility_balance"] = (
            features["expert_attribution_ratio"]
            - features["anonymous_source_ratio"]
        )

        return features

    # -----------------------------------------------------

    def _quote_ratio(self, ctx: FeatureContext) -> float:
        quotes = len(self.QUOTE_PATTERN.findall(ctx.text_lower))
        return float(quotes / max(ctx.n_tokens, 1))

    # -----------------------------------------------------

    def _entity_ratio(self, ctx: FeatureContext) -> float:

        entities = [
            ent for ent in ctx.doc.ents
            if ent.label_ in ("PERSON", "ORG")
        ]

        return float(len(entities) / max(len(ctx.doc), 1))

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "expert_attribution_ratio": 0.0,
            "anonymous_source_ratio": 0.0,
            "credibility_indicator_ratio": 0.0,
            "attribution_verb_ratio": 0.0,
            "quotation_ratio": 0.0,
            "named_source_ratio": 0.0,
            "source_credibility_balance": 0.0,
        }


# ------------------------------------------------------------
# Vector conversion
# ------------------------------------------------------------

def source_attribution_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, SOURCE_ATTRIBUTION_KEYS)