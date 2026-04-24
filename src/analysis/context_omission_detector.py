# src/analysis/context_omission_detector.py

from __future__ import annotations

import logging
import re
from typing import Dict

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import term_ratio, phrase_match_count, normalize_lexicon_terms
from src.analysis.feature_schema import CONTEXT_OMISSION_KEYS, make_vector

logger = logging.getLogger(__name__)


class ContextOmissionDetector(BaseAnalyzer):

    VAGUE_REFERENCES = {
        "they","people","many","some","others",
        "experts","critics","sources","analysts",
        "officials","insiders","observers",
        "commentators","reportedly","allegedly",
        "authorities","investigators","researchers",
        "witnesses","participants","leaders",
        "lawmakers","politicians","administration",
        "supporters","opponents","activists",
        "analysts say","critics say","supporters say",
        "many believe","some claim","others argue",
        "it is said","it is believed","it is thought",
        "rumor","rumors","speculation"
    }

    ATTRIBUTION_MARKERS = {
        "according","according to","reported","reports","reportedly",
        "stated","state","stating","claimed","claim","claims",
        "said","say","says","noted","notes",
        "explained","explain","announced","announce",
        "revealed","reveal","confirmed","confirm",
        "suggested","suggest","told","wrote","writes",
        "indicated","acknowledged","commented","warned"
    }

    EVIDENCE_MARKERS = {
        "data","dataset","study","studies","report","reports",
        "research","analysis","evidence","statistics",
        "survey","poll","experiment","findings","results",
        "according to data","according to research",
        "research suggests","data indicates",
        "statistics indicate","evidence suggests"
    }

    UNCERTAINTY_MARKERS = {
        "allegedly","reportedly","apparently",
        "possibly","potentially","likely",
        "rumor","speculation","suggests",
        "appears","seems","may","might",
        "could","perhaps","it seems","it appears",
        "it is possible","it remains unclear"
    }

    QUOTE_PATTERN = re.compile(r'"')

    def __init__(self):
        # Normalize phrases once (important)
        self.vague_phrases = normalize_lexicon_terms(self.VAGUE_REFERENCES)
        self.evidence_phrases = normalize_lexicon_terms(self.EVIDENCE_MARKERS)

    # ------------------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty_features()

        features: Dict[str, float] = {}

        #  Fast token-based ratios
        features["context_vague_reference_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.VAGUE_REFERENCES
        )

        features["context_attribution_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.ATTRIBUTION_MARKERS
        )

        features["context_uncertainty_ratio"] = term_ratio(
            ctx.token_counts, ctx.n_tokens, self.UNCERTAINTY_MARKERS
        )

        #  Phrase-based (cached regex)
        features["context_evidence_ratio"] = self._phrase_ratio(
            ctx.text_lower, ctx.n_tokens, self.evidence_phrases
        )

        #  Quote density
        features["context_quote_ratio"] = self._quote_ratio(ctx.text_lower, ctx.n_tokens)

        #  Entity features (reuse doc)
        features.update(self._entity_context_features(ctx))

        #  Grounding score
        features["context_grounding_score"] = float(
            np.clip(
                0.5 * features["context_evidence_ratio"]
                + 0.5 * features["context_entity_ratio"],
                0.0,
                1.0,
            )
        )

        return features

    # ------------------------------------------------------------

    def _phrase_ratio(self, text_lower: str, n_tokens: int, phrases: set) -> float:
        hits = phrase_match_count(text_lower, phrases)
        return float(hits / n_tokens)

    # ------------------------------------------------------------

    def _quote_ratio(self, text_lower: str, n_tokens: int) -> float:
        quotes = len(self.QUOTE_PATTERN.findall(text_lower))
        return float(quotes / max(n_tokens, 1))

    # ------------------------------------------------------------

    def _entity_context_features(self, ctx: FeatureContext) -> Dict[str, float]:

        doc = ctx.doc

        total_tokens = max(len(doc), 1)

        entity_count = len(doc.ents)
        entity_ratio = entity_count / total_tokens

        entity_types = {ent.label_ for ent in doc.ents}

        return {
            "context_entity_ratio": float(entity_ratio),
            "context_entity_type_diversity": float(len(entity_types)),
        }

    # ------------------------------------------------------------

    def _empty_features(self) -> Dict[str, float]:
        return {
            "context_vague_reference_ratio": 0.0,
            "context_attribution_ratio": 0.0,
            "context_evidence_ratio": 0.0,
            "context_uncertainty_ratio": 0.0,
            "context_quote_ratio": 0.0,
            "context_entity_ratio": 0.0,
            "context_entity_type_diversity": 0.0,
            "context_grounding_score": 0.0,
        }


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def context_feature_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, CONTEXT_OMISSION_KEYS)