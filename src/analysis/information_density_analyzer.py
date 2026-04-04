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
import spacy
from spacy.language import Language
from spacy.tokens import Doc


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

        tokens: List[str] = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
        ]

        features: Dict[str, float] = {}

        features.update(self._term_ratio(tokens, self.FACTUAL_TERMS, "factual_density"))
        features.update(self._term_ratio(tokens, self.OPINION_TERMS, "opinion_density"))
        features.update(self._term_ratio(tokens, self.CLAIM_TERMS, "claim_density"))
        features.update(self._term_ratio(tokens, self.RHETORICAL_TERMS, "rhetorical_density"))
        features.update(self._term_ratio(tokens, self.EMOTIONAL_TERMS, "emotion_density"))
        features.update(self._term_ratio(tokens, self.MODAL_TERMS, "modal_density"))

        features.update(self._punctuation_rhetoric(text))

        features.update(self._information_emotion_ratio(features))

        logger.debug("Information density features computed")

        return features

    # ------------------------------------------------------------
    # Lexical density
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

        ratio = factual / max(emotion, 1e-6)

        return {"information_emotion_ratio": float(ratio)}


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def information_density_vector(features: Dict[str, float]) -> np.ndarray:

    ordered_keys = [
        "factual_density",
        "opinion_density",
        "claim_density",
        "rhetorical_density",
        "emotion_density",
        "modal_density",
        "rhetorical_punctuation_density",
        "information_emotion_ratio",
    ]

    vector = np.array(
        [float(features.get(k, 0.0)) for k in ordered_keys],
        dtype=np.float32,
    )

    return vector