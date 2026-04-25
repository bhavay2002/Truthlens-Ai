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
from src.analysis.feature_schema import NARRATIVE_TEMPORAL_KEYS, make_vector
from src.analysis.spacy_loader import get_doc

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# ANALYZER
# =========================================================

class NarrativeTemporalAnalyzer(BaseAnalyzer):

    name = "narrative_temporal"
    expected_keys = set(NARRATIVE_TEMPORAL_KEYS)

    PAST_TERMS: Set[str] = {...}
    CRISIS_TERMS: Set[str] = {...}
    URGENCY_TERMS: Set[str] = {...}

    # =========================================================

    def __init__(self):

        self.past = normalize_lexicon_terms(self.PAST_TERMS)
        self.crisis = normalize_lexicon_terms(self.CRISIS_TERMS)
        self.urgency = normalize_lexicon_terms(self.URGENCY_TERMS)

        logger.info("NarrativeTemporalAnalyzer initialized (final)")

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # 🔥 lazy-safe
        ctx.ensure_tokens()

        if ctx.safe_n_tokens() == 0:
            return self._empty()

        # shared spaCy
        doc = get_doc(ctx, task="syntax")

        # -----------------------------------------------------
        # RAW DENSITIES
        # -----------------------------------------------------

        raw = {
            "past": self._density(ctx, self.past),
            "crisis": self._density(ctx, self.crisis),
            "urgency": self._density(ctx, self.urgency),
        }

        # -----------------------------------------------------
        # NORMALIZATION
        # -----------------------------------------------------

        dist = self._normalize(raw)

        # -----------------------------------------------------
        # TENSE FEATURES
        # -----------------------------------------------------

        tense = self._tense_distribution(doc)

        # -----------------------------------------------------
        # TEMPORAL CONTRAST
        # -----------------------------------------------------

        contrast = float(np.std(list(dist.values())))

        # -----------------------------------------------------
        # TEMPORAL INTENSITY
        # -----------------------------------------------------

        intensity = float(sum(raw.values()) / (len(raw) + EPS))

        # -----------------------------------------------------
        # DIVERSITY
        # -----------------------------------------------------

        diversity = self._entropy(dist)

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        return {
            "past_framing_ratio": self._safe(dist["past"]),
            "crisis_escalation_ratio": self._safe(dist["crisis"]),
            "urgency_language_ratio": self._safe(dist["urgency"]),
            **tense,
            "temporal_contrast_score": self._safe(contrast),
            "temporal_intensity": self._safe(intensity),
            "temporal_diversity": self._safe(diversity),
        }

    # =========================================================

    def _density(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        n_tokens = ctx.safe_n_tokens()

        token_score = term_ratio(
            ctx.safe_counts(),
            n_tokens,
            lexicon,
        )

        phrase_hits = phrase_match_count(
            ctx.text_lower or "",
            lexicon,
        )

        phrase_score = phrase_hits / (n_tokens + EPS)

        # 🔥 weighted fusion
        return 0.7 * token_score + 0.3 * phrase_score

    # =========================================================

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:

        values = np.array(list(scores.values()), dtype=np.float32)

        total = float(values.sum())

        if total < EPS:
            return {k: 0.0 for k in scores}

        norm = values / (total + EPS)

        return dict(zip(scores.keys(), norm.astype(float)))

    # =========================================================

    def _tense_distribution(self, doc) -> Dict[str, float]:

        verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"}]

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

            if lemma in {"will", "shall"}:
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

    # =========================================================

    def _entropy(self, dist: Dict[str, float]) -> float:

        values = np.array(list(dist.values()), dtype=np.float32)

        if values.sum() < EPS:
            return 0.0

        probs = values / (values.sum() + EPS)

        entropy = -np.sum(probs * np.log(probs + EPS))
        max_entropy = np.log(len(probs))

        return float(entropy / (max_entropy + EPS))

    # =========================================================

    def _safe(self, value: float) -> float:

        if not np.isfinite(value):
            return 0.0

        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty(self) -> Dict[str, float]:

        return {
            "past_framing_ratio": 0.0,
            "crisis_escalation_ratio": 0.0,
            "urgency_language_ratio": 0.0,
            "past_tense_ratio": 0.0,
            "present_tense_ratio": 0.0,
            "future_tense_ratio": 0.0,
            "temporal_contrast_score": 0.0,
            "temporal_intensity": 0.0,
            "temporal_diversity": 0.0,
        }


# =========================================================
# VECTOR
# =========================================================

def narrative_temporal_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, NARRATIVE_TEMPORAL_KEYS)