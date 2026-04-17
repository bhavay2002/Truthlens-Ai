"""
File Name: ideological_language_detector.py
Module: Ideology Analysis - Ideological Language Detection
Description:
    Detects ideological language patterns in text for the TruthLens AI system.
    The module identifies lexical signals associated with common political
    ideology narratives such as liberty/freedom rhetoric, equality/social
    justice framing, traditionalist language, and anti-elite rhetoric.

    These features help strengthen ideology classification models by providing
    interpretable signals derived directly from discourse.

    Features detected:
        - liberty rhetoric
        - equality / social justice rhetoric
        - traditionalist language
        - anti-elite rhetoric
        - ideological polarity signals

Dependencies:
    logging
    typing
    dataclasses
    collections
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Dictionary of ideological language signals and optional numerical vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Set

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
from src.analysis.feature_schema import IDEOLOGICAL_LANGUAGE_KEYS, make_vector

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class IdeologicalLanguageConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ("ner", "parser")


# ------------------------------------------------------------
# Ideological Language Detector
# ------------------------------------------------------------

class IdeologicalLanguageDetector:

    # ----------------------------------------------------
    # Liberty / Classical Liberal rhetoric
    # ----------------------------------------------------

    LIBERTY_TERMS: Set[str] = {

        "liberty","freedom","freedoms","rights","civil_rights",
        "individual","individualism","independence","free",
        "autonomy","self_determination","self_governance",
        "constitutional","constitution","civil_liberty",
        "limited_government","personal_freedom",
        "property_rights","economic_freedom",
        "voluntary","consent","rule_of_law",
        "private_property","free_speech","free_expression"
    }

    # ----------------------------------------------------
    # Equality / Social justice rhetoric
    # ----------------------------------------------------

    EQUALITY_TERMS: Set[str] = {

        "equality","justice","fairness","equity",
        "inclusion","diversity","representation",
        "social_justice","equal_opportunity",
        "equal_rights","redistribution",
        "oppression","systemic","systemic_racism",
        "discrimination","marginalized","minorities",
        "intersectionality","injustice",
        "inequality","human_rights",
        "collective","solidarity","welfare"
    }

    # ----------------------------------------------------
    # Traditionalist / Conservative rhetoric
    # ----------------------------------------------------

    TRADITION_TERMS: Set[str] = {

        "tradition","traditional","heritage","values",
        "family","nation","national","culture",
        "identity","patriotism","patriotic",
        "faith","religion","religious",
        "community","custom","moral_values",
        "national_identity","social_order",
        "duty","honor","loyalty"
    }

    # ----------------------------------------------------
    # Anti-elite / Populist rhetoric
    # ----------------------------------------------------

    ELITE_TERMS: Set[str] = {

        "elite","elites","establishment",
        "bureaucrat","bureaucracy",
        "politician","politicians",
        "powerful","ruling_class",
        "globalist","globalists",
        "media","mainstream_media",
        "corporate","corporations",
        "oligarch","oligarchy",
        "technocrat","technocracy",
        "lobbyist","deep_state"
    }

    # ----------------------------------------------------
    # Economic ideology rhetoric
    # ----------------------------------------------------

    ECONOMIC_TERMS: Set[str] = {

        "capitalism","capitalist",
        "socialism","socialist",
        "communism","communist",
        "market","free_market",
        "regulation","deregulation",
        "privatization","public_sector",
        "government_spending",
        "taxation","wealth_tax",
        "redistribution"
    }

    # ----------------------------------------------------
    # Nationalism rhetoric
    # ----------------------------------------------------

    NATIONALISM_TERMS: Set[str] = {

        "nation","nationalism","nationalist",
        "sovereignty","sovereign",
        "border","borders",
        "immigration","immigrant",
        "homeland","patriot",
        "national_security"
    }

    # ----------------------------------------------------
    # Ideological phrases (multi-word concepts)
    # ----------------------------------------------------

    IDEOLOGY_PHRASES: Set[str] = {

        "social justice",
        "free market",
        "government control",
        "big government",
        "limited government",
        "personal freedom",
        "wealth redistribution",
        "working class",
        "middle class",
        "rule of law",
        "civil liberties",
        "identity politics",
        "economic inequality",
        "national security"
    }
    # ------------------------------------------------------------

    def __init__(self, config: IdeologicalLanguageConfig | None = None):

        self.config = config or IdeologicalLanguageConfig()

        self.nlp: Language = get_nlp(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )

        logger.info(
            "IdeologicalLanguageDetector initialized | model=%s",
            self.config.spacy_model
        )

    # ------------------------------------------------------------

    def analyze(self, text: str) -> Dict[str, float]:

        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("Input text must be non-empty")

        try:
            doc: Doc = self.nlp(text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        return self.analyze_doc(doc)

    # ------------------------------------------------------------

    def analyze_doc(self, doc: Doc) -> Dict[str, float]:
        """Compute ideological language features from a pre-built spaCy Doc.

        Builds the token counter once and reuses it for all term ratios.

        Args:
            doc: A processed spaCy Doc instance.

        Returns:
            Dictionary of ideological language feature names to float values.
        """

        tokens = extract_alpha_lemmas(doc)
        token_counts = build_counter(tokens)
        n_tokens = len(tokens)

        features: Dict[str, float] = {}

        # lexical ratios
        features["liberty_language_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.LIBERTY_TERMS
        )

        features["equality_language_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.EQUALITY_TERMS
        )

        features["tradition_language_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.TRADITION_TERMS
        )

        features["anti_elite_language_ratio"] = _term_ratio_util(
            token_counts, n_tokens, self.ELITE_TERMS
        )

        # ideological polarity
        features["liberty_vs_equality_balance"] = (
            features["liberty_language_ratio"]
            - features["equality_language_ratio"]
        )

        # phrase detection (word-boundary aware)
        features["ideology_phrase_density"] = self._phrase_density(doc.text.lower())

        logger.debug("Ideological language features computed")

        return features

    # ------------------------------------------------------------

    def _term_ratio(
        self,
        token_counts: Counter,
        tokens: List[str],
        lexicon: Set[str],
    ) -> float:

        if not tokens:
            return 0.0

        hits = sum(token_counts[t] for t in lexicon if t in token_counts)

        return float(hits / max(len(tokens), 1))

    # ------------------------------------------------------------

    def _phrase_density(self, text: str) -> float:

        hits = phrase_match_count(text, self.IDEOLOGY_PHRASES)

        return float(hits / max(len(self.IDEOLOGY_PHRASES), 1))


# ------------------------------------------------------------
# Vector Conversion
# ------------------------------------------------------------

def ideological_language_vector(features: Dict[str, float]) -> np.ndarray:

    if not isinstance(features, dict):
        raise ValueError("features must be dictionary")

    return make_vector(features, IDEOLOGICAL_LANGUAGE_KEYS)