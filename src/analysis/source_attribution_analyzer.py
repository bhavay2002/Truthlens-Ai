"""
File Name: source_attribution_analyzer.py
Module: Discourse Analysis - Source Attribution Detection
Description:
    Detects attribution patterns in text for the TruthLens AI system. The module
    analyzes how information sources are referenced, identifying expert
    attribution, anonymous source usage, and credibility indicators. These
    signals help determine whether claims are supported by identifiable sources
    or vague references, which is important for misinformation analysis.

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
    Source attribution feature dictionary and optional numerical vector
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from collections import Counter
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
class SourceAttributionConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ("parser",)


# ------------------------------------------------------------
# Analyzer
# ------------------------------------------------------------

class SourceAttributionAnalyzer:

    # ----------------------------------------------------
    # Expert attribution signals
    # ----------------------------------------------------

    EXPERT_TERMS = {

        "expert","experts","analyst","analysts",
        "researcher","researchers","scientist","scientists",
        "professor","economist","doctor","official",
        "authority","specialist"
    }

    # ----------------------------------------------------
    # Anonymous sources
    # ----------------------------------------------------

    ANONYMOUS_TERMS = {

        "sources","source","insiders","officials",
        "people","critics","observers","commentators",
        "analysts","individuals"
    }

    # ----------------------------------------------------
    # Evidence / credibility indicators
    # ----------------------------------------------------

    CREDIBILITY_TERMS = {

        "report","study","research","analysis",
        "data","dataset","statistics",
        "evidence","survey","findings",
        "according","confirmed","documented"
    }

    # ----------------------------------------------------
    # Attribution verbs
    # ----------------------------------------------------

    ATTRIBUTION_VERBS = {

        "say","said","report","reported",
        "state","stated","claim","claimed",
        "explain","explained","note","noted",
        "argue","argued","announce","announced",
        "confirm","confirmed"
    }

    QUOTE_PATTERN = re.compile(r"[\"“”']")

    # ----------------------------------------------------

    def __init__(self, config: SourceAttributionConfig | None = None):

        self.config = config or SourceAttributionConfig()

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
            "SourceAttributionAnalyzer initialized | model=%s",
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

        tokens: List[str] = [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha
        ]

        features: Dict[str, float] = {}

        features.update(self._term_ratio(tokens, self.EXPERT_TERMS, "expert_attribution_ratio"))
        features.update(self._term_ratio(tokens, self.ANONYMOUS_TERMS, "anonymous_source_ratio"))
        features.update(self._term_ratio(tokens, self.CREDIBILITY_TERMS, "credibility_indicator_ratio"))
        features.update(self._term_ratio(tokens, self.ATTRIBUTION_VERBS, "attribution_verb_ratio"))

        features.update(self._quote_ratio(text))
        features.update(self._entity_source_ratio(doc))

        features.update(self._source_balance_score(features))

        return features

    # ------------------------------------------------------------
    # Lexical ratios
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
            counts[token]
            for token in counts
            if token in lexicon
        )

        ratio = hits / max(len(tokens), 1)

        return {feature_name: float(ratio)}

    # ------------------------------------------------------------
    # Quotation ratio
    # ------------------------------------------------------------

    def _quote_ratio(self, text: str) -> Dict[str, float]:

        quotes = len(self.QUOTE_PATTERN.findall(text))

        length = max(len(text.split()), 1)

        return {"quotation_ratio": float(quotes / length)}

    # ------------------------------------------------------------
    # Named entity source detection
    # ------------------------------------------------------------

    def _entity_source_ratio(self, doc: Doc) -> Dict[str, float]:

        entities = [ent for ent in doc.ents if ent.label_ in ("PERSON", "ORG")]

        ratio = len(entities) / max(len(doc), 1)

        return {"named_source_ratio": float(ratio)}

    # ------------------------------------------------------------
    # Source balance score
    # ------------------------------------------------------------

    def _source_balance_score(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:

        expert = features.get("expert_attribution_ratio", 0.0)
        anonymous = features.get("anonymous_source_ratio", 0.0)

        balance = expert - anonymous

        return {"source_credibility_balance": float(balance)}


# ------------------------------------------------------------
# Feature vector conversion
# ------------------------------------------------------------

def source_attribution_vector(features: Dict[str, float]) -> np.ndarray:

    ordered_keys = [

        "expert_attribution_ratio",
        "anonymous_source_ratio",
        "credibility_indicator_ratio",
        "attribution_verb_ratio",
        "quotation_ratio",
        "named_source_ratio",
        "source_credibility_balance",
    ]

    vector = np.array(
        [float(features.get(k, 0.0)) for k in ordered_keys],
        dtype=np.float32,
    )

    return vector