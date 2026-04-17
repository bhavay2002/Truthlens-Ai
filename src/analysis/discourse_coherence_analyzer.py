"""
File Name: discourse_coherence_analyzer.py
Module: Discourse Analysis - Coherence Measurement
Description:
    Measures discourse coherence and logical flow of arguments for the TruthLens
    AI system. The module analyzes sentence-to-sentence semantic similarity,
    narrative continuity across segments, and the presence of discourse
    transition markers.

    These signals help identify coherent argument structures versus chaotic,
    manipulative, or poorly connected discourse patterns often found in
    propaganda, misinformation, or low-quality argumentative text.

Dependencies:
    logging
    typing
    dataclasses
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Dictionary of discourse coherence features and optional numerical vector
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc

from src.analysis._nlp import get_nlp
from src.analysis._text_features import phrase_match_count
from src.analysis.feature_schema import DISCOURSE_COHERENCE_KEYS, make_vector


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class DiscourseCoherenceConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ("ner",)


# ------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------

class DiscourseCoherenceAnalyzer:

    # ----------------------------------------------------
    # Discourse transition markers
    # ----------------------------------------------------

    TRANSITION_MARKERS = {

        # contrast
        "however","nevertheless","nonetheless",
        "yet","still","though","although",
        "in contrast","by contrast",
        "on the other hand","despite this",
        "even so","alternatively",

        # cause
        "therefore","thus","hence",
        "consequently","accordingly",
        "as a result","for this reason",
        "because","since","due to",

        # addition
        "furthermore","moreover","additionally",
        "in addition","also","besides",
        "similarly","likewise",

        # sequence
        "first","second","third",
        "next","then","afterward",
        "subsequently","finally",
        "meanwhile","at the same time",

        # summary
        "in conclusion","to conclude",
        "in summary","overall",
        "ultimately","in short",

        # clarification
        "in other words","that is",
        "to clarify","namely",
        "specifically","for example",
        "for instance"
    }

    # ----------------------------------------------------

    def __init__(self, config: DiscourseCoherenceConfig | None = None):

        self.config = config or DiscourseCoherenceConfig()

        self.nlp: Language = get_nlp(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )

        logger.info(
            "DiscourseCoherenceAnalyzer initialized | model=%s",
            self.config.spacy_model,
        )

    # ------------------------------------------------------------
    # Main Analysis
    # ------------------------------------------------------------

    def analyze(self, text: str) -> Dict[str, float]:

        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("Input text must be non-empty")

        doc: Doc = self.nlp(text)
        return self.analyze_doc(doc)

    # ------------------------------------------------------------

    def analyze_doc(self, doc: Doc) -> Dict[str, float]:
        """Compute discourse coherence features from a pre-built spaCy Doc.

        Args:
            doc: A processed spaCy Doc instance.

        Returns:
            Dictionary of discourse coherence feature names to float values.
        """

        sentences = list(doc.sents)

        features: Dict[str, float] = {}

        features.update(self._sentence_coherence(sentences))
        features.update(self._topic_drift(sentences))
        features.update(self._narrative_continuity(doc))
        features.update(self._transition_usage(doc.text, doc))

        logger.debug("Discourse coherence features computed")

        return features

    # ------------------------------------------------------------
    # Sentence Coherence
    # ------------------------------------------------------------

    def _sentence_coherence(self, sentences: List) -> Dict[str, float]:

        if len(sentences) < 2:
            return {"sentence_coherence": 0.0}

        similarities = [
            sentences[i].similarity(sentences[i + 1])
            for i in range(len(sentences) - 1)
        ]

        score = float(np.mean(similarities)) if similarities else 0.0

        return {"sentence_coherence": score}

    # ------------------------------------------------------------
    # Topic Drift Detection
    # ------------------------------------------------------------

    def _topic_drift(self, sentences: List) -> Dict[str, float]:

        if len(sentences) < 2:
            return {"topic_drift": 0.0}

        similarities = [
            sentences[i].similarity(sentences[i + 1])
            for i in range(len(sentences) - 1)
        ]

        if not similarities:
            return {"topic_drift": 0.0}

        drift = 1.0 - float(np.mean(similarities))

        return {"topic_drift": drift}

    # ------------------------------------------------------------
    # Narrative Continuity
    # ------------------------------------------------------------

    def _narrative_continuity(self, doc: Doc) -> Dict[str, float]:

        entities = [ent.text.lower() for ent in doc.ents]

        if not entities:
            return {"narrative_continuity": 0.0}

        unique_entities = set(entities)

        continuity = 1.0 - (len(unique_entities) / max(len(entities), 1))

        return {"narrative_continuity": float(continuity)}

    # ------------------------------------------------------------
    # Transition Usage
    # ------------------------------------------------------------

    def _transition_usage(self, text: str, doc: Doc) -> Dict[str, float]:

        tokens = [token.text.lower() for token in doc if token.is_alpha]

        token_count = max(len(tokens), 1)

        text_lower = text.lower()

        marker_hits = phrase_match_count(text_lower, self.TRANSITION_MARKERS)

        ratio = marker_hits / token_count

        return {"discourse_transition_ratio": float(ratio)}


# ------------------------------------------------------------
# Feature Vector Conversion
# ------------------------------------------------------------

def discourse_coherence_vector(features: Dict[str, float]) -> np.ndarray:

    return make_vector(features, DISCOURSE_COHERENCE_KEYS)