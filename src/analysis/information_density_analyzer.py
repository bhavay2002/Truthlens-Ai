"""
File Name: information_density_analyzer.py
Module: Discourse Analysis - Information Density Analysis
Description:
    Measures informational versus rhetorical density in text for the TruthLens AI
    system. The module estimates how much of a document consists of factual
    statements, opinionated language, claims, and rhetorical signals.

    These signals help differentiate factual reporting from opinion-driven or
    rhetorically persuasive writing. The extracted metrics support bias
    detection, propaganda analysis, and discourse modeling.

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
    Information density feature dictionary and optional numerical vector
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
from src.analysis._text_features import extract_alpha_lemmas, build_counter
from src.analysis.feature_schema import INFORMATION_DENSITY_KEYS, make_vector


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class InformationDensityConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ("ner",)


# ------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------

class InformationDensityAnalyzer:

    # ----------------------------------------------------
    # Factual signals
    # ----------------------------------------------------

    FACTUAL_TERMS = {

        "data","dataset","report","reports",
        "study","studies","research","analysis",
        "statistics","statistical","survey",
        "experiment","experiments","findings",
        "results","evidence","empirical",
        "according","official","documented",
        "confirmed","verified","record",
        "measurement","observed","observation"
    }

    # ----------------------------------------------------
    # Opinion signals
    # ----------------------------------------------------

    OPINION_TERMS = {

        "believe","believes","believed",
        "think","thinks","thought",
        "argue","argues","argued",
        "claim","claims","claimed",
        "suggest","suggests","suggested",
        "feel","feels","felt",
        "likely","unlikely","possibly",
        "perhaps","apparently","seems",
        "appears","assume","assumes",
        "arguably","probably"
    }

    # ----------------------------------------------------
    # Claim / inference signals
    # ----------------------------------------------------

    CLAIM_TERMS = {

        "therefore","thus","hence",
        "consequently","accordingly",
        "so","for_this_reason",
        "it_follows","this_proves",
        "this_shows","this_indicates",
        "clearly","obviously",
        "undoubtedly","without_doubt",
        "ultimately"
    }

    # ----------------------------------------------------
    # Rhetorical / persuasive signals
    # ----------------------------------------------------

    RHETORICAL_TERMS = {

        "outrageous","shocking","dangerous",
        "disaster","catastrophe","crisis",
        "threat","collapse","corrupt",
        "evil","scandal","devastating",
        "radical","extreme","unbelievable",
        "terrifying","chaos","propaganda",
        "manipulation","fraud","coverup"
    }

    # ----------------------------------------------------
    # Emotional language signals
    # ----------------------------------------------------

    EMOTIONAL_TERMS = {

        "fear","anger","outrage",
        "panic","shock","concern",
        "hope","joy","frustration",
        "sadness","anxiety","rage"
    }

    # ----------------------------------------------------
    # Modal speculation signals
    # ----------------------------------------------------

    MODAL_TERMS = {

        "may","might","could",
        "should","would","must",
        "can","cannot","possibly",
        "perhaps","likely"
    }

    RHETORICAL_PATTERN = re.compile(r"[!?]+")

    # ----------------------------------------------------

    def __init__(self, config: InformationDensityConfig | None = None):

        self.config = config or InformationDensityConfig()

        self.nlp: Language = get_nlp(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )

        self._factual_terms = {t.replace("_", " ") for t in self.FACTUAL_TERMS}
        self._opinion_terms = {t.replace("_", " ") for t in self.OPINION_TERMS}
        self._claim_terms = {t.replace("_", " ") for t in self.CLAIM_TERMS}
        self._rhetorical_terms = {t.replace("_", " ") for t in self.RHETORICAL_TERMS}
        self._emotional_terms = {t.replace("_", " ") for t in self.EMOTIONAL_TERMS}
        self._modal_terms = {t.replace("_", " ") for t in self.MODAL_TERMS}

        logger.info(
            "InformationDensityAnalyzer initialized | model=%s",
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
        """Compute information density features from a pre-built spaCy Doc.

        Builds the token counter once and reuses it across all term-ratio
        computations, eliminating repeated Counter construction.

        Args:
            doc: A processed spaCy Doc instance.

        Returns:
            Dictionary of information density feature names to float values.
        """

        tokens: List[str] = extract_alpha_lemmas(doc)
        token_counts = build_counter(tokens)
        n_tokens = len(tokens)

        features: Dict[str, float] = {}

        text_lower = doc.text.lower()
        features["factual_density"] = self._lexicon_ratio(token_counts, n_tokens, text_lower, self._factual_terms)
        features["opinion_density"] = self._lexicon_ratio(token_counts, n_tokens, text_lower, self._opinion_terms)
        features["claim_density"] = self._lexicon_ratio(token_counts, n_tokens, text_lower, self._claim_terms)
        features["rhetorical_density"] = self._lexicon_ratio(token_counts, n_tokens, text_lower, self._rhetorical_terms)
        features["emotion_density"] = self._lexicon_ratio(token_counts, n_tokens, text_lower, self._emotional_terms)
        features["modal_density"] = self._lexicon_ratio(token_counts, n_tokens, text_lower, self._modal_terms)

        features.update(self._punctuation_rhetoric(doc.text))

        features.update(self._information_emotion_ratio(features))

        logger.debug("Information density features computed")

        return features

    # ------------------------------------------------------------
    # Lexical density (kept for backward compatibility)
    # ------------------------------------------------------------

    def _term_ratio(
        self,
        tokens: List[str],
        lexicon: set,
        feature_name: str,
    ) -> Dict[str, float]:

        if not tokens:
            return {feature_name: 0.0}

        counts = Counter(tokens)

        hits = sum(counts[token] for token in counts if token in lexicon)

        ratio = hits / max(len(tokens), 1)

        return {feature_name: float(ratio)}

    # ------------------------------------------------------------
    # Rhetorical punctuation
    # ------------------------------------------------------------

    def _punctuation_rhetoric(self, text: str) -> Dict[str, float]:

        matches = self.RHETORICAL_PATTERN.findall(text)

        length = max(len(text.split()), 1)

        score = len(matches) / length

        return {"rhetorical_punctuation_density": float(score)}

    # ------------------------------------------------------------
    # Information-to-Emotion Ratio
    # ------------------------------------------------------------

    def _information_emotion_ratio(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        factual = features.get("factual_density", 0.0)
        emotion = features.get("emotion_density", 0.0)

        eps = 1e-6
        ratio = float(factual / max(emotion, eps))
        ratio = float(np.clip(ratio, 0.0, 10.0))

        return {"information_emotion_ratio": ratio}

    def _lexicon_ratio(
        self,
        token_counts: Dict[str, int],
        n_tokens: int,
        text_lower: str,
        lexicon: set,
    ) -> float:
        hits = 0
        for term in lexicon:
            if " " in term:
                hits += text_lower.count(term)
            else:
                hits += token_counts.get(term, 0)

        return float(hits / max(n_tokens, 1))


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def information_density_vector(features: Dict[str, float]) -> np.ndarray:

    return make_vector(features, INFORMATION_DENSITY_KEYS)