# src/analysis/ideological_language_detector.py

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
from src.analysis.feature_schema import IDEOLOGICAL_LANGUAGE_KEYS, make_vector

logger = logging.getLogger(__name__)


class IdeologicalLanguageDetector(BaseAnalyzer):

    LIBERTY_TERMS = {
        "liberty","freedom","freedoms","rights","civil rights",
        "individual","individualism","independence","free",
        "autonomy","self determination","self governance",
        "constitutional","constitution","civil liberty",
        "limited government","personal freedom",
        "property rights","economic freedom",
        "voluntary","consent","rule of law",
        "private property","free speech","free expression"
    }

    EQUALITY_TERMS = {
        "equality","justice","fairness","equity",
        "inclusion","diversity","representation",
        "social justice","equal opportunity",
        "equal rights","redistribution",
        "oppression","systemic","systemic racism",
        "discrimination","marginalized","minorities",
        "intersectionality","injustice",
        "inequality","human rights",
        "collective","solidarity","welfare"
    }

    TRADITION_TERMS = {
        "tradition","traditional","heritage","values",
        "family","nation","national","culture",
        "identity","patriotism","patriotic",
        "faith","religion","religious",
        "community","custom","moral values",
        "national identity","social order",
        "duty","honor","loyalty"
    }

    ELITE_TERMS = {
        "elite","elites","establishment",
        "bureaucrat","bureaucracy",
        "politician","politicians",
        "powerful","ruling class",
        "globalist","globalists",
        "media","mainstream media",
        "corporate","corporations",
        "oligarch","oligarchy",
        "technocrat","technocracy",
        "lobbyist","deep state"
    }

    IDEOLOGY_PHRASES = {
        "social justice",
        "free market",
        "government control",
        "big government",
        "limited government",
        "personal freedom",
        "wealth redistribution",
        "working class",
        "middle class",
        "rule of law",
        "civil liberties",
        "identity politics",
        "economic inequality",
        "national security"
    }

    def __init__(self):
        #  Normalize once (CRITICAL)
        self.liberty = normalize_lexicon_terms(self.LIBERTY_TERMS)
        self.equality = normalize_lexicon_terms(self.EQUALITY_TERMS)
        self.tradition = normalize_lexicon_terms(self.TRADITION_TERMS)
        self.elite = normalize_lexicon_terms(self.ELITE_TERMS)
        self.phrases = normalize_lexicon_terms(self.IDEOLOGY_PHRASES)

        logger.info("IdeologicalLanguageDetector initialized (optimized)")

    # ------------------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty_features()

        features: Dict[str, float] = {}

        #  Token-level ratios (fast)
        features["liberty_language_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.liberty
        )

        features["equality_language_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.equality
        )

        features["tradition_language_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.tradition
        )

        features["anti_elite_language_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.elite
        )

        # Ideological polarity
        features["liberty_vs_equality_balance"] = (
            features["liberty_language_ratio"]
            - features["equality_language_ratio"]
        )

        #  Phrase density (cached regex)
        phrase_hits = phrase_match_count(
            ctx.text_lower,
            self.phrases,
        )

        features["ideology_phrase_density"] = float(
            phrase_hits / max(ctx.n_tokens, 1)
        )

        return features

    # ------------------------------------------------------------

    def _empty_features(self) -> Dict[str, float]:
        return {
            "liberty_language_ratio": 0.0,
            "equality_language_ratio": 0.0,
            "tradition_language_ratio": 0.0,
            "anti_elite_language_ratio": 0.0,
            "liberty_vs_equality_balance": 0.0,
            "ideology_phrase_density": 0.0,
        }


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def ideological_language_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, IDEOLOGICAL_LANGUAGE_KEYS)