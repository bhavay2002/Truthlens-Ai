# src/analysis/information_density.py

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
from src.analysis.feature_schema import INFORMATION_DENSITY_KEYS, make_vector

logger = logging.getLogger(__name__)


class InformationDensityAnalyzer(BaseAnalyzer):

    FACTUAL_TERMS: Set[str] = {
        "according", "reported", "confirmed", "stated", "announced", "revealed",
        "showed", "demonstrated", "proved", "established", "documented", "verified",
        "found", "concluded", "published", "researched", "studied", "measured",
        "recorded", "observed", "evidence", "data", "statistics", "research",
        "study", "report", "analysis", "survey", "census", "experiment",
    }

    OPINION_TERMS: Set[str] = {
        "believe", "think", "feel", "argue", "contend", "suggest", "claim",
        "assert", "maintain", "insist", "seem", "appear",
        "presumably", "probably", "likely", "apparently",
        "supposedly", "allegedly", "arguably",
        "in my view", "in my opinion", "it seems", "i believe", "we think",
    }

    CLAIM_TERMS: Set[str] = {
        "claims", "alleges", "asserts", "declares", "states", "argues",
        "contends", "maintains", "insists", "purports", "denies", "admits",
        "acknowledges", "concedes", "charges", "accuses", "blames",
        "according to", "sources say", "reportedly", "allegedly",
    }

    RHETORICAL_TERMS: Set[str] = {
        "obviously", "clearly", "undeniably", "unquestionably",
        "certainly", "absolutely", "definitely", "surely",
        "indeed", "of course",
        "needless to say", "it is clear", "everyone knows",
        "always", "never", "every", "all", "none",
        "impossible", "inevitable",
    }

    EMOTIONAL_TERMS: Set[str] = {
        "outrageous", "shocking", "disgusting", "horrifying",
        "terrible", "devastating", "catastrophic", "alarming",
        "dangerous", "frightening",
        "wonderful", "amazing", "incredible", "fantastic",
        "brilliant", "heartbreaking", "tragic", "disastrous",
        "explosive", "crisis", "threat", "attack",
        "destroy", "collapse", "panic", "fear", "rage",
    }

    MODAL_TERMS: Set[str] = {
        "should", "would", "could", "might", "must", "may",
        "shall", "will", "ought", "need", "dare",
        "used to", "had better", "would rather",
    }

    def __init__(self):
        # 🔥 Normalize once
        self.factual = normalize_lexicon_terms(self.FACTUAL_TERMS)
        self.opinion = normalize_lexicon_terms(self.OPINION_TERMS)
        self.claim = normalize_lexicon_terms(self.CLAIM_TERMS)
        self.rhetorical = normalize_lexicon_terms(self.RHETORICAL_TERMS)
        self.emotion = normalize_lexicon_terms(self.EMOTIONAL_TERMS)
        self.modal = normalize_lexicon_terms(self.MODAL_TERMS)

        logger.info("InformationDensityAnalyzer initialized (optimized)")

    # ------------------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty()

        features: Dict[str, float] = {}

        features["factual_density"] = self._density(ctx, self.factual)
        features["opinion_density"] = self._density(ctx, self.opinion)
        features["claim_density"] = self._density(ctx, self.claim)
        features["rhetorical_density"] = self._density(ctx, self.rhetorical)
        features["emotion_density"] = self._density(ctx, self.emotion)
        features["modal_density"] = self._density(ctx, self.modal)

        features["rhetorical_punctuation_density"] = self._punctuation(ctx)

        features.update(self._information_emotion_ratio(features))

        return features

    # ------------------------------------------------------------

    def _density(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        token_ratio = term_ratio(ctx.token_counts, ctx.n_tokens, lexicon)

        phrase_hits = phrase_match_count(ctx.text_lower, lexicon)
        phrase_ratio = phrase_hits / max(ctx.n_tokens, 1)

        return float(np.clip(token_ratio + phrase_ratio, 0.0, 1.0))

    # ------------------------------------------------------------

    def _punctuation(self, ctx: FeatureContext) -> float:
        count = ctx.text_lower.count("!") + ctx.text_lower.count("?")
        return float(np.clip(count / max(ctx.n_tokens, 1), 0.0, 1.0))

    # ------------------------------------------------------------

    def _information_emotion_ratio(self, features: Dict[str, float]) -> Dict[str, float]:

        factual = features.get("factual_density", 0.0)
        emotion = features.get("emotion_density", 0.0)

        eps = 1e-9

        raw = factual / max(emotion, eps)
        raw = float(np.clip(raw, 0.0, 10.0))

        return {
            "information_emotion_ratio": raw,
            "information_emotion_ratio_normalized": raw / 10.0,
        }

    # ------------------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "factual_density": 0.0,
            "opinion_density": 0.0,
            "claim_density": 0.0,
            "rhetorical_density": 0.0,
            "emotion_density": 0.0,
            "modal_density": 0.0,
            "rhetorical_punctuation_density": 0.0,
            "information_emotion_ratio": 0.0,
            "information_emotion_ratio_normalized": 0.0,
        }


# ------------------------------------------------------------
# Vector conversion
# ------------------------------------------------------------

def information_density_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, INFORMATION_DENSITY_KEYS)