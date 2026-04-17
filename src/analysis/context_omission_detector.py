"""
File Name: context_omission_detector.py
Module: Discourse Analysis - Context Omission Detection
Description:
    Detects potential context omission patterns in text for the TruthLens AI
    system. The module analyzes linguistic signals that often indicate that
    important contextual information may be missing, simplified, or selectively
    presented. It examines discourse cues such as vague references, missing
    attribution, limited evidence markers, and abrupt claims without supporting
    context.

Dependencies:
    logging
    typing
    collections
    numpy
    spacy
    re

Inputs:
    Raw text string

Outputs:
    Dictionary of context omission features and optional numerical vector
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc

from src.analysis._nlp import get_nlp
from src.analysis._text_features import (
    extract_alpha_lemmas,
    build_counter,
    term_ratio as _term_ratio_util,
)
from src.analysis.feature_schema import CONTEXT_OMISSION_KEYS, make_vector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class ContextOmissionConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ("parser",)
    normalize_ratios: bool = True


# ------------------------------------------------------------
# Detector
# ------------------------------------------------------------

class ContextOmissionDetector:

    # ----------------------------------------------------
    # Vague references (unverifiable sources)
    # ----------------------------------------------------

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

    # ----------------------------------------------------
    # Attribution signals (source referencing)
    # ----------------------------------------------------

    ATTRIBUTION_MARKERS = {

        "according","according to",
        "reported","reports","reportedly",
        "stated","state","stating",
        "claimed","claim","claims",
        "said","say","says",
        "noted","notes",
        "explained","explain",
        "announced","announce",
        "revealed","reveal",
        "confirmed","confirm",
        "suggested","suggest",

        "told","told reporters",
        "wrote","writes",
        "indicated","indicates",
        "acknowledged","acknowledges",
        "commented","comments",
        "warned","warns"
    }

    # ----------------------------------------------------
    # Evidence signals (empirical grounding)
    # ----------------------------------------------------

    EVIDENCE_MARKERS = {

        "data","dataset",
        "study","studies",
        "report","reports",
        "research","researchers",
        "analysis","analysis shows",
        "evidence","empirical evidence",
        "statistics","statistical",
        "survey","poll","polling",
        "experiment","experiments",
        "findings","results","outcomes",

        "according to data",
        "according to research",
        "research suggests",
        "research shows",
        "data indicates",
        "data suggests",
        "statistics indicate",
        "analysis indicates",
        "evidence suggests"
    }

    # ----------------------------------------------------
    # Uncertainty / speculation signals
    # ----------------------------------------------------

    UNCERTAINTY_MARKERS = {

        "allegedly","reportedly","apparently",
        "possibly","potentially",
        "likely","unlikely",
        "rumored","rumour","rumor",
        "speculation","speculative",

        "suggests","appears","seems",
        "may","might","could",
        "can","possibly","perhaps",

        "it appears","it seems",
        "it is possible",
        "it is believed",
        "it is thought",
        "it remains unclear"
    }

    QUOTE_PATTERN = re.compile(r'"')

    def __init__(self, config: ContextOmissionConfig | None = None):

        self.config = config or ContextOmissionConfig()

        self.nlp: Language = get_nlp(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )

        logger.info(
            "ContextOmissionDetector initialized | model=%s",
            self.config.spacy_model
        )

    # ------------------------------------------------------------

    def analyze(self, text: str) -> Dict[str, float]:

        if not isinstance(text, str):
            raise ValueError("Input text must be string")

        text = text.strip()

        if not text:
            raise ValueError("Input text cannot be empty")

        doc: Doc = self.nlp(text)
        return self.analyze_doc(doc)

    # ------------------------------------------------------------

    def analyze_doc(self, doc: Doc) -> Dict[str, float]:
        """Compute context omission features from a pre-built spaCy Doc.

        Args:
            doc: A processed spaCy Doc instance.

        Returns:
            Dictionary of context omission feature names to float values.
        """

        tokens = extract_alpha_lemmas(doc)
        token_counts = build_counter(tokens)
        n_tokens = len(tokens)

        features: Dict[str, float] = {}

        features["context_vague_reference_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.VAGUE_REFERENCES
        )

        features["context_attribution_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.ATTRIBUTION_MARKERS
        )

        features["context_evidence_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.EVIDENCE_MARKERS
        )

        features["context_uncertainty_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.UNCERTAINTY_MARKERS
        )

        features["context_quote_ratio"] = self._quote_ratio(doc.text)

        features.update(self._entity_context_features(doc))

        # contextual grounding score
        features["context_grounding_score"] = (
            features["context_evidence_ratio"]
            + features["context_entity_ratio"]
        )

        logger.debug("Context omission features computed")

        return features

    # ------------------------------------------------------------
    # Lexical ratios (retained in case subclasses call this method)
    # ------------------------------------------------------------

    def _term_ratio(
        self,
        token_counts: Counter,
        tokens: List[str],
        lexicon: set,
    ) -> float:

        if not tokens:
            return 0.0

        hits = sum(token_counts[t] for t in lexicon if t in token_counts)

        return float(hits / max(len(tokens), 1))

    def _quote_ratio(self, text: str) -> float:

        quote_count = len(self.QUOTE_PATTERN.findall(text))

        return float(quote_count / max(len(text.split()), 1))

    # ------------------------------------------------------------

    def _entity_context_features(self, doc: Doc) -> Dict[str, float]:

        entities = list(doc.ents)

        total_tokens = max(len(doc), 1)

        entity_ratio = len(entities) / total_tokens

        entity_types = Counter(ent.label_ for ent in entities)

        diversity = len(entity_types)

        return {
            "context_entity_ratio": float(entity_ratio),
            "context_entity_type_diversity": float(diversity),
        }


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def context_feature_vector(features: Dict[str, float]) -> np.ndarray:

    return make_vector(features, CONTEXT_OMISSION_KEYS)