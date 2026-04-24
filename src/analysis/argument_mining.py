# src/analysis/argument_mining.py

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
from src.analysis.feature_schema import ARGUMENT_MINING_KEYS, make_vector

logger = logging.getLogger(__name__)


class ArgumentMiningAnalyzer(BaseAnalyzer):

    CLAIM_MARKERS = {
        "therefore","thus","hence","consequently","so",
        "accordingly","as a result","for this reason",
        "it follows","it follows that",
        "this proves","this shows","this demonstrates",
        "this indicates","this confirms",
        "clearly","obviously","undoubtedly",
        "without doubt","there is no doubt",
        "in conclusion","to conclude","overall",
        "ultimately","in summary"
    }

    PREMISE_MARKERS = {
        "because","since","given","as",
        "considering","due to","owing to",
        "based on","in light of",
        "for the reason that",
        "seeing that","inasmuch as",
        "assuming that","insofar as"
    }

    SUPPORT_MARKERS = {
        "for example","for instance","as an example",
        "to illustrate","as evidence",
        "evidence","empirical evidence",
        "data shows","data suggest",
        "studies show","research indicates",
        "research shows","statistics show",
        "statistics indicate",
        "according to","reports show",
        "analysis shows","findings suggest"
    }

    CONTRAST_MARKERS = {
        "however","but","although","though",
        "yet","nevertheless","nonetheless",
        "on the other hand","in contrast",
        "by contrast","alternatively",
        "despite","despite this",
        "even though","whereas",
        "while","still"
    }

    REBUTTAL_MARKERS = {
        "however","nonetheless","still",
        "nevertheless","despite this",
        "even so","regardless",
        "that said","having said that",
        "yet still","in spite of this",
        "contrary to this",
        "despite these claims"
    }

    def __init__(self):
        # Normalize once (CRITICAL optimization)
        self.support_phrases = normalize_lexicon_terms(self.SUPPORT_MARKERS)
        self.rebuttal_phrases = normalize_lexicon_terms(self.REBUTTAL_MARKERS)

    # ------------------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty_features()

        features: Dict[str, float] = {}

        # ✅ Token-based ratios (fast)
        features["argument_claim_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.CLAIM_MARKERS
        )

        features["argument_premise_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.PREMISE_MARKERS
        )

        features["argument_contrast_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.CONTRAST_MARKERS
        )

        # ✅ Phrase-based ratios (cached regex)
        features["argument_support_ratio"] = self._phrase_ratio(
            ctx.text_lower, ctx.n_tokens, self.support_phrases
        )

        features["argument_rebuttal_ratio"] = self._phrase_ratio(
            ctx.text_lower, ctx.n_tokens, self.rebuttal_phrases
        )

        # ✅ Structural features (reuse doc)
        features.update(self._argument_density(ctx))

        features["argument_complexity"] = (
            features["argument_clause_density"]
            + features["argument_verb_density"]
        )

        return features

    # ------------------------------------------------------------

    def _phrase_ratio(
        self,
        text_lower: str,
        n_tokens: int,
        phrases: set,
    ) -> float:
        hits = phrase_match_count(text_lower, phrases)
        return float(hits / n_tokens)

    # ------------------------------------------------------------

    def _argument_density(self, ctx: FeatureContext) -> Dict[str, float]:

        doc = ctx.doc

        verbs = sum(1 for t in doc if t.pos_ == "VERB")
        clauses = sum(
            1 for t in doc if t.dep_ in {"ccomp", "xcomp", "advcl"}
        )

        total_tokens = max(len(doc), 1)

        return {
            "argument_verb_density": float(verbs / total_tokens),
            "argument_clause_density": float(clauses / total_tokens),
        }

    # ------------------------------------------------------------

    def _empty_features(self) -> Dict[str, float]:
        return {
            "argument_claim_ratio": 0.0,
            "argument_premise_ratio": 0.0,
            "argument_support_ratio": 0.0,
            "argument_contrast_ratio": 0.0,
            "argument_rebuttal_ratio": 0.0,
            "argument_verb_density": 0.0,
            "argument_clause_density": 0.0,
            "argument_complexity": 0.0,
        }


# ------------------------------------------------------------
# Vector Conversion (UNCHANGED)
# ------------------------------------------------------------

def argument_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, ARGUMENT_MINING_KEYS)