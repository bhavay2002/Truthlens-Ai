# src/analysis/information_omission_detector.py

from __future__ import annotations

import logging
from typing import Dict, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    term_ratio,
    phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import INFORMATION_OMISSION_KEYS, make_vector

logger = logging.getLogger(__name__)


class InformationOmissionDetector(BaseAnalyzer):

    COUNTERARGUMENT_MARKERS: Set[str] = {
        "however","but","although","though",
        "yet","nevertheless","nonetheless",
        "still","despite","on the other hand",
        "in contrast","alternatively","even so"
    }

    EVIDENCE_MARKERS: Set[str] = {
        "evidence","data","dataset",
        "study","studies","research",
        "analysis","report","reports",
        "statistics","survey",
        "experiment","experiments",
        "findings","results",
        "according","according to",
        "empirical","documented"
    }

    CLAIM_MARKERS: Set[str] = {
        "therefore","thus","hence",
        "consequently","accordingly",
        "clearly","obviously",
        "undoubtedly","without doubt",
        "shows","proves","demonstrates",
        "confirms","indicates"
    }

    FRAMING_MARKERS: Set[str] = {
        "clearly","obviously",
        "undeniably","without doubt",
        "everyone knows","it is clear",
        "there is no doubt"
    }

    def __init__(self):
        # 🔥 Normalize once
        self.counter = normalize_lexicon_terms(self.COUNTERARGUMENT_MARKERS)
        self.evidence = normalize_lexicon_terms(self.EVIDENCE_MARKERS)
        self.claim = normalize_lexicon_terms(self.CLAIM_MARKERS)
        self.framing = normalize_lexicon_terms(self.FRAMING_MARKERS)

        logger.info("InformationOmissionDetector initialized (optimized)")

    # ------------------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty()

        features: Dict[str, float] = {}

        counter_hits = self._hits(ctx, self.counter)
        evidence_hits = self._hits(ctx, self.evidence)
        claim_hits = self._hits(ctx, self.claim)
        framing_hits = self._hits(ctx, self.framing)

        # 🔥 Missing counterarguments
        features["missing_counterargument_score"] = float(
            1.0 - (counter_hits / max(ctx.n_tokens, 1))
        )

        # 🔥 One-sided framing
        raw = (claim_hits + framing_hits) / max(counter_hits + 1, 1)
        features["one_sided_framing_score"] = float(raw / (1.0 + raw))

        # 🔥 Evidence strength
        evidence_ratio = evidence_hits / max(ctx.n_tokens, 1)
        features["incomplete_evidence_score"] = float(1.0 - evidence_ratio)

        # 🔥 Claim vs evidence imbalance
        if claim_hits == 0:
            features["claim_evidence_imbalance"] = 0.0
        else:
            features["claim_evidence_imbalance"] = float(
                claim_hits / max(evidence_hits + 1, 1)
            )

        return features

    # ------------------------------------------------------------

    def _hits(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        token_hits = sum(
            ctx.token_counts.get(term, 0)
            for term in lexicon
            if " " not in term
        )

        phrase_hits = phrase_match_count(ctx.text_lower, lexicon)

        return float(token_hits + phrase_hits)

    # ------------------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "missing_counterargument_score": 0.0,
            "one_sided_framing_score": 0.0,
            "incomplete_evidence_score": 0.0,
            "claim_evidence_imbalance": 0.0,
        }


# ------------------------------------------------------------
# Vector conversion
# ------------------------------------------------------------

def information_omission_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, INFORMATION_OMISSION_KEYS)