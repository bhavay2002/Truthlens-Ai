"""
File Name: information_omission_detector.py
Module: Discourse Analysis - Information Omission Detection
Description:
    Detects advanced information omission patterns in text for the TruthLens AI
    system. This module extends basic context omission detection by identifying
    missing counterarguments, one-sided framing, and incomplete evidence chains.

    These signals help detect narratives where opposing viewpoints are absent,
    evidence is weak or incomplete, or arguments are presented without balanced
    perspectives. Such patterns are common in propaganda, misinformation, and
    highly biased discourse.

Dependencies:
    logging
    typing
    dataclasses
    collections
    numpy
    spacy
    re

Inputs:
    Raw text string

Outputs:
    Dictionary of information omission features and optional numerical vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc

from src.analysis._nlp import get_nlp
from src.analysis._text_features import extract_alpha_lemmas, build_counter
from src.analysis.feature_schema import INFORMATION_OMISSION_KEYS, make_vector


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class InformationOmissionConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ("ner",)


# ------------------------------------------------------------
# Detector
# ------------------------------------------------------------

class InformationOmissionDetector:

    # ----------------------------------------------------
    # Counterargument markers
    # ----------------------------------------------------

    COUNTERARGUMENT_MARKERS = {

        "however","but","although","though",
        "yet","nevertheless","nonetheless",
        "still","despite","despite_this",
        "on_the_other_hand","in_contrast",
        "alternatively","even_so"
    }

    # ----------------------------------------------------
    # Evidence markers
    # ----------------------------------------------------

    EVIDENCE_MARKERS = {

        "evidence","data","dataset",
        "study","studies","research",
        "analysis","report","reports",
        "statistics","survey",
        "experiment","experiments",
        "findings","results",
        "according","according_to",
        "empirical","documented"
    }

    # ----------------------------------------------------
    # Claim markers
    # ----------------------------------------------------

    CLAIM_MARKERS = {

        "therefore","thus","hence",
        "consequently","accordingly",
        "clearly","obviously",
        "undoubtedly","without_doubt",
        "shows","proves","demonstrates",
        "confirms","indicates"
    }

    # ----------------------------------------------------
    # Framing markers
    # ----------------------------------------------------

    FRAMING_MARKERS = {

        "clearly","obviously",
        "undeniably","without_doubt",
        "everyone_knows","it_is_clear",
        "there_is_no_doubt"
    }

    # ----------------------------------------------------

    def __init__(self, config: InformationOmissionConfig | None = None):

        self.config = config or InformationOmissionConfig()

        self.nlp: Language = get_nlp(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )

        logger.info(
            "InformationOmissionDetector initialized | model=%s",
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
        """Compute information omission features from a pre-built spaCy Doc.

        Args:
            doc: A processed spaCy Doc instance.

        Returns:
            Dictionary of information omission feature names to float values.
        """

        tokens: List[str] = extract_alpha_lemmas(doc)

        features: Dict[str, float] = {}

        features.update(self._missing_counterarguments(tokens))
        features.update(self._one_sided_framing(tokens))
        features.update(self._evidence_chain_strength(tokens))
        features.update(self._claim_evidence_balance(tokens))

        logger.debug("Information omission features computed")

        return features

    # ------------------------------------------------------------
    # Missing counterarguments
    # ------------------------------------------------------------

    def _missing_counterarguments(self, tokens: List[str]) -> Dict[str, float]:

        if not tokens:
            return {"missing_counterargument_score": 0.0}

        counter_hits = sum(
            1 for token in tokens
            if token in self.COUNTERARGUMENT_MARKERS
        )

        score = 1.0 - (counter_hits / max(len(tokens), 1))

        return {"missing_counterargument_score": float(score)}

    # ------------------------------------------------------------
    # One-sided framing
    # ------------------------------------------------------------

    def _one_sided_framing(self, tokens: List[str]) -> Dict[str, float]:

        if not tokens:
            return {"one_sided_framing_score": 0.0}

        claim_hits = sum(
            1 for token in tokens
            if token in self.CLAIM_MARKERS
        )

        counter_hits = sum(
            1 for token in tokens
            if token in self.COUNTERARGUMENT_MARKERS
        )

        framing_hits = sum(
            1 for token in tokens
            if token in self.FRAMING_MARKERS
        )

        score = (claim_hits + framing_hits) / max(counter_hits + 1, 1)

        return {"one_sided_framing_score": float(score)}

    # ------------------------------------------------------------
    # Evidence chain strength
    # ------------------------------------------------------------

    def _evidence_chain_strength(self, tokens: List[str]) -> Dict[str, float]:

        if not tokens:
            return {"incomplete_evidence_score": 0.0}

        counts = Counter(tokens)

        evidence_hits = sum(
            counts[token]
            for token in counts
            if token in self.EVIDENCE_MARKERS
        )

        ratio = evidence_hits / max(len(tokens), 1)

        score = 1.0 - ratio

        return {"incomplete_evidence_score": float(score)}

    # ------------------------------------------------------------
    # Claim vs Evidence balance
    # ------------------------------------------------------------

    def _claim_evidence_balance(self, tokens: List[str]) -> Dict[str, float]:

        claim_hits = sum(
            1 for token in tokens
            if token in self.CLAIM_MARKERS
        )

        evidence_hits = sum(
            1 for token in tokens
            if token in self.EVIDENCE_MARKERS
        )

        if claim_hits == 0:
            return {"claim_evidence_imbalance": 0.0}

        imbalance = claim_hits / max(evidence_hits + 1, 1)

        return {"claim_evidence_imbalance": float(imbalance)}


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def information_omission_vector(features: Dict[str, float]) -> np.ndarray:

    return make_vector(features, INFORMATION_OMISSION_KEYS)