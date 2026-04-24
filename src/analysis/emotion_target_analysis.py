# src/analysis/emotion_target_analysis.py

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, DefaultDict, Tuple

import numpy as np
from spacy.matcher import PhraseMatcher

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis.feature_schema import EMOTION_TARGET_KEYS, make_vector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Emotion Terms (keep your existing dictionary)
# ---------------------------------------------------------
EMOTION_TERMS = {
    # ... (UNCHANGED: your full dict)
}


# ---------------------------------------------------------
# Optional intensity weights (can be tuned or learned)
# ---------------------------------------------------------
# Default: 1.0 if not specified
EMOTION_INTENSITY = {
    # stronger signals can be >1.0
    "anger": 1.2,
    "disgust": 1.2,
    "joy": 1.1,
    "sadness": 1.1,
    # others default to 1.0
}


# ---------------------------------------------------------
# Analyzer
# ---------------------------------------------------------

class EmotionTargetAnalyzer(BaseAnalyzer):

    def __init__(self, nlp=None):
        """
        Args:
            nlp: shared spaCy pipeline (pass from pipeline to avoid reloading)
        """
        #  O(1) lemma → emotion
        self.term_to_emotion: Dict[str, str] = {}
        self.term_weights: Dict[str, float] = {}

        for emotion, terms in EMOTION_TERMS.items():
            for t in terms:
                normalized = t.replace("_", " ").lower()
                self.term_to_emotion[normalized] = emotion
                self.term_weights[normalized] = EMOTION_INTENSITY.get(emotion, 1.0)

        #  PhraseMatcher for multi-token phrases
        self.matcher = None
        if nlp is not None:
            self.matcher = self._build_phrase_matcher(nlp)

        logger.info("EmotionTargetAnalyzer initialized (hybrid + weighted)")

    # -----------------------------------------------------

    def _build_phrase_matcher(self, nlp):
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

        patterns = []
        self.phrase_to_emotion: Dict[Tuple[str, ...], str] = {}
        self.phrase_weights: Dict[Tuple[str, ...], float] = {}

        for emotion, terms in EMOTION_TERMS.items():
            for term in terms:
                if " " in term or "_" in term:
                    text = term.replace("_", " ")
                    doc = nlp.make_doc(text)
                    patterns.append(doc)

                    key = tuple([t.lower_ for t in doc])
                    self.phrase_to_emotion[key] = emotion
                    self.phrase_weights[key] = EMOTION_INTENSITY.get(emotion, 1.0)

        if patterns:
            matcher.add("EMOTION_PHRASES", patterns)

        return matcher

    # -----------------------------------------------------

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        if ctx.n_tokens == 0:
            return self._empty_features()

        entity_emotion_map: DefaultDict[str, float] = defaultdict(float)
        emotion_type_counter: DefaultDict[str, float] = defaultdict(float)

        emotion_score_total = 0.0

        doc = ctx.doc

        # -------------------------------------------------
        #  1. Phrase matching (multi-token)
        # -------------------------------------------------

        if self.matcher:
            matches = self.matcher(doc)

            for _, start, end in matches:
                span = doc[start:end]
                key = tuple([t.lower_ for t in span])

                emotion = self.phrase_to_emotion.get(key)
                weight = self.phrase_weights.get(key, 1.0)

                if not emotion:
                    continue

                emotion_score_total += weight
                emotion_type_counter[emotion] += weight

                target = self._resolve_target(span.root)

                if target:
                    entity_emotion_map[target] += weight

        # -------------------------------------------------
        #  2. Token-level matching (fast path)
        # -------------------------------------------------

        for token in doc:

            lemma = token.lemma_.lower()
            emotion = self.term_to_emotion.get(lemma)

            if not emotion:
                continue

            weight = self.term_weights.get(lemma, 1.0)

            emotion_score_total += weight
            emotion_type_counter[emotion] += weight

            target = self._resolve_target(token)

            if target:
                entity_emotion_map[target] += weight

        # -------------------------------------------------

        total_entities = sum(entity_emotion_map.values())
        expression_ratio = emotion_score_total / max(len(doc), 1)

        emotion_types = len(emotion_type_counter)
        dominant_emotion_strength = (
            max(emotion_type_counter.values())
            if emotion_type_counter else 0.0
        )

        if total_entities == 0:
            return {
                "emotion_target_diversity": 0.0,
                "emotion_target_focus": 0.0,
                "emotion_expression_ratio": float(expression_ratio),
                "emotion_type_diversity": float(emotion_types),
                "dominant_emotion_strength": float(dominant_emotion_strength),
            }

        diversity = len(entity_emotion_map)
        dominant_target = max(entity_emotion_map.values())
        focus_score = dominant_target / max(total_entities, 1)

        return {
            "emotion_target_diversity": float(diversity),
            "emotion_target_focus": float(focus_score),
            "emotion_expression_ratio": float(expression_ratio),
            "emotion_type_diversity": float(emotion_types),
            "dominant_emotion_strength": float(dominant_emotion_strength),
        }

    # -----------------------------------------------------
    #  Hybrid Target Resolution
    # -----------------------------------------------------

    def _resolve_target(self, token) -> str | None:

        # 1️ Named entity
        if token.ent_iob_ in {"B", "I"} and token.ent_type_:
            span = token.doc[token.ent_start: token.ent_end]
            if span.text.strip():
                return span.text.lower().strip()

        # 2️ Dependency-based (subject/object)
        for child in token.children:
            if child.dep_ in {"nsubj", "dobj", "pobj"}:
                return child.lemma_.lower()

        if token.head and token.head != token:
            if token.dep_ in {"amod", "acomp"}:
                return token.head.lemma_.lower()

        # 3️ Fallback
        return token.lemma_.lower()

    # -----------------------------------------------------

    def _empty_features(self) -> Dict[str, float]:
        return {
            "emotion_target_diversity": 0.0,
            "emotion_target_focus": 0.0,
            "emotion_expression_ratio": 0.0,
            "emotion_type_diversity": 0.0,
            "dominant_emotion_strength": 0.0,
        }


# ---------------------------------------------------------
# Vector Conversion
# ---------------------------------------------------------

def emotion_target_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, EMOTION_TARGET_KEYS)