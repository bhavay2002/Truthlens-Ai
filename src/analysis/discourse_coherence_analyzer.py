# src/analysis/discourse_coherence_analyzer.py

from __future__ import annotations

import logging
from typing import Dict, List, Set

import numpy as np

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis._text_features import phrase_match_count, normalize_lexicon_terms
from src.analysis.feature_schema import DISCOURSE_COHERENCE_KEYS, make_vector

logger = logging.getLogger(__name__)


class DiscourseCoherenceAnalyzer(BaseAnalyzer):

    TRANSITION_MARKERS = {
        "however","nevertheless","nonetheless",
        "yet","still","though","although",
        "in contrast","by contrast",
        "on the other hand","despite this",
        "even so","alternatively",

        "therefore","thus","hence",
        "consequently","accordingly",
        "as a result","for this reason",
        "because","since","due to",

        "furthermore","moreover","additionally",
        "in addition","also","besides",
        "similarly","likewise",

        "first","second","third",
        "next","then","afterward",
        "subsequently","finally",
        "meanwhile","at the same time",

        "in conclusion","to conclude",
        "in summary","overall",
        "ultimately","in short",

        "in other words","that is",
        "to clarify","namely",
        "specifically","for example",
        "for instance"
    }

    def __init__(self):
        self.transition_phrases = normalize_lexicon_terms(self.TRANSITION_MARKERS)

    # ------------------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        doc = ctx.doc
        sentences = list(doc.sents)

        if len(sentences) < 2:
            return self._empty_features()

        #  Precompute sentence token sets ONCE
        sentence_tokens = [
            self._sentence_token_set(sent)
            for sent in sentences
        ]

        similarities = [
            self._jaccard(sentence_tokens[i], sentence_tokens[i + 1])
            for i in range(len(sentence_tokens) - 1)
        ]

        mean_sim = float(np.mean(similarities)) if similarities else 0.0

        features = {
            "sentence_coherence": mean_sim,
            "topic_drift": float(1.0 - mean_sim),
        }

        # Narrative continuity
        features.update(self._narrative_continuity(doc))

        # Transition markers (cached regex)
        features["discourse_transition_ratio"] = self._transition_ratio(
            ctx.text_lower,
            ctx.n_tokens
        )

        return features

    # ------------------------------------------------------------

    def _sentence_token_set(self, sent) -> Set[str]:
        return {
            token.lemma_.lower()
            for token in sent
            if token.is_alpha and not token.is_stop
        }

    # ------------------------------------------------------------

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 0.0
        return float(len(a & b) / max(len(a | b), 1))

    # ------------------------------------------------------------

    def _narrative_continuity(self, doc) -> Dict[str, float]:

        entities = [ent.text.lower() for ent in doc.ents]

        if not entities:
            return {"narrative_continuity": 0.0}

        unique_entities = set(entities)

        continuity = 1.0 - (len(unique_entities) / max(len(entities), 1))

        return {"narrative_continuity": float(continuity)}

    # ------------------------------------------------------------

    def _transition_ratio(self, text_lower: str, n_tokens: int) -> float:

        hits = phrase_match_count(text_lower, self.transition_phrases)

        return float(hits / max(n_tokens, 1))

    # ------------------------------------------------------------

    def _empty_features(self) -> Dict[str, float]:
        return {
            "sentence_coherence": 0.0,
            "topic_drift": 0.0,
            "narrative_continuity": 0.0,
            "discourse_transition_ratio": 0.0,
        }


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def discourse_coherence_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, DISCOURSE_COHERENCE_KEYS)