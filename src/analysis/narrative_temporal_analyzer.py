# src/analysis/narrative_temporal_analyzer.py

from __future__ import annotations

import logging
from typing import Dict, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    term_ratio,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import NARRATIVE_TEMPORAL_KEYS, make_vector

logger = logging.getLogger(__name__)


class NarrativeTemporalAnalyzer(BaseAnalyzer):

    PAST_TERMS: Set[str] = {
        "previously","earlier","historically","formerly","once",
        "before","past","recently","prior",
        "years","decades","centuries","era","period",
        "traditionally","longstanding","historical","in the past",
    }

    CRISIS_TERMS: Set[str] = {
        "crisis","collapse","disaster","catastrophe",
        "breakdown","emergency","meltdown",
        "chaos","turmoil","instability","unrest",
        "escalation","worsening","spiral","deterioration",
        "danger","threat","risk",
    }

    URGENCY_TERMS: Set[str] = {
        "immediately","urgent","now","rapidly","quickly",
        "instantly","suddenly","swiftly",
        "critical","pressing","dire","serious",
        "act now","time is running out",
    }

    # -----------------------------------------------------

    def __init__(self):

        #  Normalize once
        self.past = normalize_lexicon_terms(self.PAST_TERMS)
        self.crisis = normalize_lexicon_terms(self.CRISIS_TERMS)
        self.urgency = normalize_lexicon_terms(self.URGENCY_TERMS)

        logger.info("NarrativeTemporalAnalyzer initialized (optimized)")

    # -----------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty()

        features: Dict[str, float] = {}

        #  Token-based ratios (fast)
        features["past_framing_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.past
        )

        features["crisis_escalation_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.crisis
        )

        features["urgency_language_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.urgency
        )

        #  Tense distribution (reuse doc)
        features.update(self._tense_distribution(ctx))

        #  Temporal contrast
        features["temporal_contrast_score"] = abs(
            features["past_framing_ratio"]
            - features["urgency_language_ratio"]
        )

        return features

    # -----------------------------------------------------

    def _tense_distribution(self, ctx: FeatureContext) -> Dict[str, float]:

        verbs = [t for t in ctx.doc if t.pos_ in {"VERB", "AUX"}]

        if not verbs:
            return {
                "past_tense_ratio": 0.0,
                "present_tense_ratio": 0.0,
                "future_tense_ratio": 0.0,
            }

        past = present = future = 0

        for token in verbs:

            tag = token.tag_
            lemma = token.lemma_.lower()

            if lemma in {"will", "shall"} or tag == "MD":
                future += 1
            elif tag in {"VBD", "VBN"}:
                past += 1
            elif tag in {"VB", "VBP", "VBZ", "VBG"}:
                present += 1

        total = max(len(verbs), 1)

        return {
            "past_tense_ratio": past / total,
            "present_tense_ratio": present / total,
            "future_tense_ratio": future / total,
        }

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "past_framing_ratio": 0.0,
            "crisis_escalation_ratio": 0.0,
            "urgency_language_ratio": 0.0,
            "past_tense_ratio": 0.0,
            "present_tense_ratio": 0.0,
            "future_tense_ratio": 0.0,
            "temporal_contrast_score": 0.0,
        }


# ---------------------------------------------------------
# Vector
# ---------------------------------------------------------

def narrative_temporal_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, NARRATIVE_TEMPORAL_KEYS)