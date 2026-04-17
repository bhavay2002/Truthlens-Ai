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
from spacy.language import Language
from spacy.tokens import Doc

from src.analysis._nlp import get_nlp
from src.analysis._text_features import (
    extract_alpha_lemmas,
    build_counter,
    phrase_match_count,
    normalize_lexicon_terms,
    term_ratio as _term_ratio_util,
)
from src.analysis.feature_schema import ARGUMENT_MINING_KEYS, make_vector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class ArgumentMiningConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple[str, ...] = ("ner",)


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

        self.nlp: Language = get_nlp(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )

        logger.info(
            "ArgumentMiningAnalyzer initialized | model=%s",
            self.config.spacy_model
        )

    # ------------------------------------------------------------

    def analyze(self, text: str, return_vector: bool = False):

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()

        if not text:
            features = {k: 0.0 for k in ARGUMENT_MINING_KEYS}
            return (
                features,
                make_vector(features, ARGUMENT_MINING_KEYS),
            ) if return_vector else features

        doc: Doc = self.nlp(text)
        return self.analyze_doc(doc)

    # ------------------------------------------------------------

    def analyze_doc(self, doc: Doc) -> Dict[str, float]:
        """Compute argument mining features from a pre-built spaCy Doc.

        Accepts a :class:`~spacy.tokens.Doc` that was already processed by a
        spaCy pipeline (typically the shared instance from the integration
        runner), avoiding redundant tokenisation.

        Args:
            doc: A processed spaCy Doc instance.

        Returns:
            Dictionary of argument mining feature names to float values.
        """

        tokens = extract_alpha_lemmas(doc)
        token_counts = build_counter(tokens)
        n_tokens = len(tokens)

        # Guard once here; all helper methods can then assume n_tokens > 0.
        if n_tokens == 0:
            return {
                "argument_claim_ratio": 0.0,
                "argument_premise_ratio": 0.0,
                "argument_support_ratio": 0.0,
                "argument_contrast_ratio": 0.0,
                "argument_rebuttal_ratio": 0.0,
                "argument_verb_density": 0.0,
                "argument_clause_density": 0.0,
                "argument_complexity": 0.0,
            }

        features: Dict[str, float] = {}

        features["argument_claim_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.CLAIM_MARKERS
        )

        features["argument_premise_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.PREMISE_MARKERS
        )

        features["argument_support_ratio"] = self._phrase_ratio(
            doc.text, n_tokens, self.SUPPORT_MARKERS
        )

        features["argument_contrast_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.CONTRAST_MARKERS
        )

        features["argument_rebuttal_ratio"] = self._phrase_ratio(
            doc.text, n_tokens, self.REBUTTAL_MARKERS
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

        text_lower = text.lower()
        hits = 0

        for marker in markers:
            if " " in marker:
                hits += text_lower.count(marker)
            else:
                hits += token_counts.get(marker, 0)

        return float(hits / len(tokens))

    # ------------------------------------------------------------

    def _phrase_ratio(
        self,
        text: str,
        n_tokens: int,
        phrases: set,
    ) -> float:

        hits = phrase_match_count(
            text.lower(),
            normalize_lexicon_terms(phrases),
        )

        return float(hits / n_tokens)

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

    return make_vector(features, ARGUMENT_MINING_KEYS)