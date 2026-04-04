"""
File Name: argument_mining.py
Module: Discourse Analysis - Argument Mining
Description:
    Implements argument mining utilities for the TruthLens AI system. The module
    extracts argumentation structures from text by identifying claims, premises,
    supporting statements, and argumentative discourse markers. These features
    support higher-level narrative and propaganda analysis by modeling how
    arguments are constructed within text.

Dependencies:
    logging
    typing
    collections
    dataclasses
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Argumentation feature dictionary and optional numerical vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List
from collections import Counter

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class ArgumentMiningConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ("ner",)


# ------------------------------------------------------------
# Argument Mining Analyzer
# ------------------------------------------------------------

class ArgumentMiningAnalyzer:

    # ----------------------------------------------------
    # Claim indicators
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # Premise indicators
    # ----------------------------------------------------

    PREMISE_MARKERS = {

        "because","since","given","as",
        "considering","due to","owing to",
        "based on","in light of",
        "for the reason that",
        "seeing that","inasmuch as",
        "assuming that","insofar as"
    }

    # ----------------------------------------------------
    # Supporting evidence indicators
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # Counterargument / contrast markers
    # ----------------------------------------------------

    CONTRAST_MARKERS = {

        "however","but","although","though",
        "yet","nevertheless","nonetheless",
        "on the other hand","in contrast",
        "by contrast","alternatively",
        "despite","despite this",
        "even though","whereas",
        "while","still"
    }

    # ----------------------------------------------------
    # Rebuttal markers
    # ----------------------------------------------------

    REBUTTAL_MARKERS = {

        "however","nonetheless","still",
        "nevertheless","despite this",
        "even so","regardless",
        "that said","having said that",
        "nonetheless","yet still",
        "in spite of this",
        "contrary to this",
        "despite these claims"
    }

    def __init__(self, config: ArgumentMiningConfig | None = None):

        self.config = config or ArgumentMiningConfig()

        try:
            self.nlp: Language = spacy.load(
                self.config.spacy_model,
                disable=self.config.disable_components
            )
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "ArgumentMiningAnalyzer initialized | model=%s",
            self.config.spacy_model
        )

    # ------------------------------------------------------------

    def analyze(self, text: str) -> Dict[str, float]:

        if not isinstance(text, str):
            raise ValueError("Input text must be string")

        text = text.strip()

        if not text:
            raise ValueError("Input text must be non-empty")

        doc: Doc = self.nlp(text)

        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
        ]

        token_counts = Counter(tokens)

        features: Dict[str, float] = {}

        features["argument_claim_ratio"] = self._marker_ratio(
            text, tokens, token_counts, self.CLAIM_MARKERS
        )

        features["argument_premise_ratio"] = self._marker_ratio(
            text, tokens, token_counts, self.PREMISE_MARKERS
        )

        features["argument_support_ratio"] = self._phrase_ratio(
            text, self.SUPPORT_MARKERS
        )

        features["argument_contrast_ratio"] = self._marker_ratio(
            text, tokens, token_counts, self.CONTRAST_MARKERS
        )

        features["argument_rebuttal_ratio"] = self._phrase_ratio(
            text, self.REBUTTAL_MARKERS
        )

        features.update(self._argument_density(doc))

        features["argument_complexity"] = (
            features["argument_clause_density"]
            + features["argument_verb_density"]
        )

        logger.debug("Argument mining features computed")

        return features

    # ------------------------------------------------------------

    def _marker_ratio(
        self,
        text: str,
        tokens: List[str],
        token_counts: Counter,
        markers: set
    ) -> float:

        if not tokens:
            return 0.0

        hits = sum(token_counts[t] for t in markers if t in token_counts)

        return float(hits / max(len(tokens), 1))

    # ------------------------------------------------------------

    def _phrase_ratio(
        self,
        text: str,
        phrases: set
    ) -> float:

        text_lower = text.lower()

        hits = sum(1 for phrase in phrases if phrase in text_lower)

        return float(hits / max(len(text.split()), 1))

    # ------------------------------------------------------------

    def _argument_density(self, doc: Doc) -> Dict[str, float]:

        verbs = [t for t in doc if t.pos_ == "VERB"]

        clauses = [
            t for t in doc
            if t.dep_ in {"ccomp", "xcomp", "advcl"}
        ]

        total_tokens = max(len(doc), 1)

        return {
            "argument_verb_density": float(len(verbs) / total_tokens),
            "argument_clause_density": float(len(clauses) / total_tokens),
        }


# ------------------------------------------------------------
# Feature Vector Conversion
# ------------------------------------------------------------

def argument_feature_vector(features: Dict[str, float]) -> np.ndarray:

    ordered_keys = [
        "argument_claim_ratio",
        "argument_premise_ratio",
        "argument_support_ratio",
        "argument_contrast_ratio",
        "argument_rebuttal_ratio",
        "argument_verb_density",
        "argument_clause_density",
        "argument_complexity",
    ]

    vector = np.array(
        [float(features.get(k, 0.0)) for k in ordered_keys],
        dtype=np.float32
    )

    return vector