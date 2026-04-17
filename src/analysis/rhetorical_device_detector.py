"""
File Name: rhetorical_device_detector.py
Module: Discourse Analysis - Rhetorical Device Detection
Description:
    Detects rhetorical persuasion techniques in text for the TruthLens AI system.
    The module identifies linguistic signals commonly associated with persuasive
    rhetoric used in propaganda, political messaging, and biased discourse.

    The detector focuses on rhetorical patterns including exaggeration,
    loaded language, emotional appeal, fear appeal, scapegoating, and
    false dilemmas. These features help quantify persuasive intensity and
    manipulation strategies present in text.

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
    Rhetorical feature dictionary and optional numerical vector
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
    phrase_match_count,
)
from src.analysis.feature_schema import RHETORICAL_DEVICE_KEYS, make_vector


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class RhetoricalDeviceConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ("ner",)


# ------------------------------------------------------------
# Detector
# ------------------------------------------------------------

class RhetoricalDeviceDetector:

    # ----------------------------------------------------
    # Exaggeration / hyperbole
    # ----------------------------------------------------

    EXAGGERATION_TERMS = {

        "always","never","everyone","nobody",
        "completely","totally","absolutely",
        "entirely","undeniably","inevitably",
        "catastrophe","disaster","collapse"
    }

    # ----------------------------------------------------
    # Loaded ideological language
    # ----------------------------------------------------

    LOADED_LANGUAGE_TERMS = {

        "corrupt","traitor","radical",
        "extreme","dangerous","evil",
        "outrageous","shocking","disgrace",
        "tyranny","propaganda","manipulation",
        "fraud","agenda","indoctrination"
    }

    # ----------------------------------------------------
    # Emotional appeal
    # ----------------------------------------------------

    EMOTIONAL_APPEAL_TERMS = {

        "heartbreaking","tragic","devastating",
        "hope","fear","anger","rage",
        "pain","suffering","panic",
        "anxiety","outrage","despair"
    }

    # ----------------------------------------------------
    # Fear appeals
    # ----------------------------------------------------

    FEAR_APPEAL_TERMS = {

        "threat","danger","risk","crisis",
        "attack","collapse","terror",
        "invasion","emergency","catastrophe"
    }

    # ----------------------------------------------------
    # Intensifiers
    # ----------------------------------------------------

    INTENSIFIERS = {

        "very","extremely","highly",
        "incredibly","really","so","too"
    }

    # ----------------------------------------------------
    # Phrase patterns
    # ----------------------------------------------------

    SCAPEGOAT_PATTERNS = {
        "they are responsible",
        "they caused",
        "their fault",
        "blame them",
        "those people"
    }

    FALSE_DILEMMA_PATTERNS = {
        "either",
        "or else",
        "no alternative",
        "only choice",
        "nothing else",
        "no other option"
    }

    RHETORICAL_PUNCT_PATTERN = re.compile(r"[!?]+")

    # ----------------------------------------------------

    def __init__(self, config: RhetoricalDeviceConfig | None = None):

        self.config = config or RhetoricalDeviceConfig()

        self.nlp: Language = get_nlp(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )

        logger.info(
            "RhetoricalDeviceDetector initialized | model=%s",
            self.config.spacy_model,
        )

    # ------------------------------------------------------------
    # Main analysis
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
        """Compute rhetorical device features from a pre-built spaCy Doc.

        Builds the token counter once and reuses it across all term-ratio
        computations, and uses word-boundary-aware phrase matching.

        Args:
            doc: A processed spaCy Doc instance.

        Returns:
            Dictionary of rhetorical device feature names to float values.
        """

        tokens: List[str] = extract_alpha_lemmas(doc)
        token_counts = build_counter(tokens)
        n_tokens = len(tokens)

        features: Dict[str, float] = {}

        features["rhetoric_exaggeration_score"] = _term_ratio_util(
            token_counts, n_tokens, self.EXAGGERATION_TERMS
        )
        features["rhetoric_loaded_language_score"] = _term_ratio_util(
            token_counts, n_tokens, self.LOADED_LANGUAGE_TERMS
        )
        features["rhetoric_emotional_appeal_score"] = _term_ratio_util(
            token_counts, n_tokens, self.EMOTIONAL_APPEAL_TERMS
        )
        features["rhetoric_fear_appeal_score"] = _term_ratio_util(
            token_counts, n_tokens, self.FEAR_APPEAL_TERMS
        )
        features["rhetoric_intensifier_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.INTENSIFIERS
        )

        features.update(self._pattern_score(doc.text, self.SCAPEGOAT_PATTERNS, "rhetoric_scapegoating_score"))
        features.update(self._pattern_score(doc.text, self.FALSE_DILEMMA_PATTERNS, "rhetoric_false_dilemma_score"))

        features.update(self._rhetorical_punctuation(doc.text))

        return features

    # ------------------------------------------------------------
    # Lexical ratios (kept for backward compatibility)
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

        hits = sum(
            counts[token] for token in counts
            if token in lexicon
        )

        ratio = hits / max(len(tokens), 1)

        return {feature_name: float(ratio)}

    # ------------------------------------------------------------
    # Phrase pattern detection
    # ------------------------------------------------------------

    def _pattern_score(
        self,
        text: str,
        patterns: set,
        feature_name: str,
    ) -> Dict[str, float]:

        text_lower = text.lower()

        hits = phrase_match_count(text_lower, patterns)

        length = max(len(text.split()), 1)

        score = hits / length

        return {feature_name: float(score)}

    # ------------------------------------------------------------
    # Rhetorical punctuation
    # ------------------------------------------------------------

    def _rhetorical_punctuation(self, text: str) -> Dict[str, float]:

        matches = self.RHETORICAL_PUNCT_PATTERN.findall(text)

        length = max(len(text.split()), 1)

        score = len(matches) / length

        return {"rhetoric_punctuation_score": float(score)}


# ------------------------------------------------------------
# Feature vector conversion
# ------------------------------------------------------------

def rhetorical_feature_vector(features: Dict[str, float]) -> np.ndarray:

    return make_vector(features, RHETORICAL_DEVICE_KEYS)