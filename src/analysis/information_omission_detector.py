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

        self._counterargument_markers = self._normalize_lexicon(self.COUNTERARGUMENT_MARKERS)
        self._evidence_markers = self._normalize_lexicon(self.EVIDENCE_MARKERS)
        self._claim_markers = self._normalize_lexicon(self.CLAIM_MARKERS)
        self._framing_markers = self._normalize_lexicon(self.FRAMING_MARKERS)

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

        features.update(self._missing_counterarguments(doc.text, tokens))
        features.update(self._one_sided_framing(doc.text, tokens))
        features.update(self._evidence_chain_strength(doc.text, tokens))
        features.update(self._claim_evidence_balance(doc.text, tokens))

        logger.debug("Information omission features computed")

        return features

    # ------------------------------------------------------------
    # Missing counterarguments
    # ------------------------------------------------------------

    def _missing_counterarguments(self, text: str, tokens: List[str]) -> Dict[str, float]:

        if not tokens:
            return {"missing_counterargument_score": 0.0}

        counter_hits = self._count_terms(text, tokens, self._counterargument_markers)

        score = 1.0 - (counter_hits / max(len(tokens), 1))

        return {"missing_counterargument_score": float(score)}

    # ------------------------------------------------------------
    # One-sided framing
    # ------------------------------------------------------------

    def _one_sided_framing(self, text: str, tokens: List[str]) -> Dict[str, float]:

        if not tokens:
            return {"one_sided_framing_score": 0.0}

        claim_hits = self._count_terms(text, tokens, self._claim_markers)
        counter_hits = self._count_terms(text, tokens, self._counterargument_markers)
        framing_hits = self._count_terms(text, tokens, self._framing_markers)

        raw = (claim_hits + framing_hits) / max(counter_hits + 1, 1)
        score = float(raw / (1.0 + raw))

        return {"one_sided_framing_score": float(score)}

    # ------------------------------------------------------------
    # Evidence chain strength
    # ------------------------------------------------------------

    def _evidence_chain_strength(self, text: str, tokens: List[str]) -> Dict[str, float]:

        if not tokens:
            return {"incomplete_evidence_score": 0.0}

        evidence_hits = self._count_terms(text, tokens, self._evidence_markers)

        ratio = evidence_hits / max(len(tokens), 1)

        score = 1.0 - ratio

        return {"incomplete_evidence_score": float(score)}

    # ------------------------------------------------------------
    # Claim vs Evidence balance
    # ------------------------------------------------------------

    def _claim_evidence_balance(self, text: str, tokens: List[str]) -> Dict[str, float]:

        claim_hits = self._count_terms(text, tokens, self._claim_markers)
        evidence_hits = self._count_terms(text, tokens, self._evidence_markers)

        if claim_hits == 0:
            return {"claim_evidence_imbalance": 0.0}

        imbalance = claim_hits / max(evidence_hits + 1, 1)

        return {"claim_evidence_imbalance": float(imbalance)}

    def _normalize_lexicon(self, terms: set[str]) -> set[str]:
        return {t.lower().replace("_", " ").strip() for t in terms if t}

    def _count_terms(self, text: str, tokens: List[str], terms: set[str]) -> int:
        text_lower = text.lower()
        token_counts = Counter(tokens)
        hits = 0
        for term in terms:
            if " " in term:
                hits += text_lower.count(term)
            else:
                hits += token_counts.get(term, 0)
        return hits


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def information_omission_vector(features: Dict[str, float]) -> np.ndarray:

    return make_vector(features, INFORMATION_OMISSION_KEYS)