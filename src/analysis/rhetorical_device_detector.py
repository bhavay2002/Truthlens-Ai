# src/analysis/rhetorical_device_detector.py

from __future__ import annotations

import logging
import re
from typing import Dict, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import (
    term_ratio,
    phrase_match_count,
    normalize_lexicon_terms,
)
from src.analysis.feature_schema import RHETORICAL_DEVICE_KEYS, make_vector

logger = logging.getLogger(__name__)


class RhetoricalDeviceDetector(BaseAnalyzer):

    EXAGGERATION_TERMS = {
        "always","never","everyone","nobody",
        "completely","totally","absolutely",
        "entirely","undeniably","inevitably",
        "catastrophe","disaster","collapse"
    }

    LOADED_LANGUAGE_TERMS = {
        "corrupt","traitor","radical",
        "extreme","dangerous","evil",
        "outrageous","shocking","disgrace",
        "tyranny","propaganda","manipulation",
        "fraud","agenda","indoctrination"
    }

    EMOTIONAL_APPEAL_TERMS = {
        "heartbreaking","tragic","devastating",
        "hope","fear","anger","rage",
        "pain","suffering","panic",
        "anxiety","outrage","despair"
    }

    FEAR_APPEAL_TERMS = {
        "threat","danger","risk","crisis",
        "attack","collapse","terror",
        "invasion","emergency","catastrophe"
    }

    INTENSIFIERS = {
        "very","extremely","highly",
        "incredibly","really","so","too"
    }

    SCAPEGOAT_PATTERNS = {
        "they are responsible",
        "they caused",
        "their fault",
        "blame them",
        "those people"
    }

    FALSE_DILEMMA_PATTERNS = {
        "either",
        "or else",
        "no alternative",
        "only choice",
        "nothing else",
        "no other option"
    }

    RHETORICAL_PUNCT_PATTERN = re.compile(r"[!?]+")

    # -----------------------------------------------------

    def __init__(self):

        # 🔥 Normalize ONCE
        self.exaggeration = normalize_lexicon_terms(self.EXAGGERATION_TERMS)
        self.loaded = normalize_lexicon_terms(self.LOADED_LANGUAGE_TERMS)
        self.emotional = normalize_lexicon_terms(self.EMOTIONAL_APPEAL_TERMS)
        self.fear = normalize_lexicon_terms(self.FEAR_APPEAL_TERMS)
        self.intensifiers = normalize_lexicon_terms(self.INTENSIFIERS)

        self.scapegoat_patterns = normalize_lexicon_terms(self.SCAPEGOAT_PATTERNS)
        self.false_dilemma_patterns = normalize_lexicon_terms(self.FALSE_DILEMMA_PATTERNS)

        logger.info("RhetoricalDeviceDetector initialized (optimized)")

    # -----------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty()

        features: Dict[str, float] = {}

        features["rhetoric_exaggeration_score"] = self._score(ctx, self.exaggeration)
        features["rhetoric_loaded_language_score"] = self._score(ctx, self.loaded)
        features["rhetoric_emotional_appeal_score"] = self._score(ctx, self.emotional)
        features["rhetoric_fear_appeal_score"] = self._score(ctx, self.fear)
        features["rhetoric_intensifier_ratio"] = self._score(ctx, self.intensifiers)

        features["rhetoric_scapegoating_score"] = self._pattern(ctx, self.scapegoat_patterns)
        features["rhetoric_false_dilemma_score"] = self._pattern(ctx, self.false_dilemma_patterns)

        features["rhetoric_punctuation_score"] = self._punctuation(ctx)

        return features

    # -----------------------------------------------------

    def _score(self, ctx: FeatureContext, lexicon: Set[str]) -> float:

        token_ratio = term_ratio(ctx.token_counts, ctx.n_tokens, lexicon)

        phrase_hits = phrase_match_count(ctx.text_lower, lexicon)
        phrase_ratio = phrase_hits / max(ctx.n_tokens, 1)

        return float(np.clip(token_ratio + phrase_ratio, 0.0, 1.0))

    # -----------------------------------------------------

    def _pattern(self, ctx: FeatureContext, patterns: Set[str]) -> float:

        hits = phrase_match_count(ctx.text_lower, patterns)

        return float(hits / max(ctx.n_tokens, 1))

    # -----------------------------------------------------

    def _punctuation(self, ctx: FeatureContext) -> float:

        matches = self.RHETORICAL_PUNCT_PATTERN.findall(ctx.text_lower)

        return float(len(matches) / max(ctx.n_tokens, 1))

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "rhetoric_exaggeration_score": 0.0,
            "rhetoric_loaded_language_score": 0.0,
            "rhetoric_emotional_appeal_score": 0.0,
            "rhetoric_fear_appeal_score": 0.0,
            "rhetoric_intensifier_ratio": 0.0,
            "rhetoric_scapegoating_score": 0.0,
            "rhetoric_false_dilemma_score": 0.0,
            "rhetoric_punctuation_score": 0.0,
        }


# ------------------------------------------------------------
# Vector conversion
# ------------------------------------------------------------

def rhetorical_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, RHETORICAL_DEVICE_KEYS)