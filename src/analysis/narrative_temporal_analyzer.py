"""
File Name: narrative_temporal_analyzer.py
Module: Narrative Analysis - Temporal Narrative Structure

Description
-----------
Analyzes temporal narrative structure within text for the TruthLens AI system.

This module extracts temporal narrative signals including:

1. Historical framing (past narrative justification)
2. Crisis escalation language
3. Urgency and immediacy signals
4. Verb tense distribution (past / present / future)
5. Temporal escalation patterns

Temporal signals are widely used in propaganda, crisis reporting,
and political discourse to influence audience perception.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(slots=True)
class NarrativeTemporalConfig:

    spacy_model: str = "en_core_web_sm"


# ---------------------------------------------------------
# Temporal Analyzer
# ---------------------------------------------------------

class NarrativeTemporalAnalyzer:

    """
    Detects temporal narrative signals.
    """

    # -----------------------------------------------------
    # Historical framing
    # -----------------------------------------------------

    PAST_TERMS = {

        "previously","earlier","historically","formerly","once",
        "before","past","recently","prior",

        "years","decades","centuries","era","period",

        "traditionally","longstanding","historical","in_the_past",
    }

    # -----------------------------------------------------
    # Crisis escalation
    # -----------------------------------------------------

    CRISIS_TERMS = {

        "crisis","collapse","disaster","catastrophe",
        "breakdown","emergency","meltdown",

        "chaos","turmoil","instability","unrest",

        "escalation","worsening","spiral","deterioration",

        "danger","threat","risk",
    }

    # -----------------------------------------------------
    # Urgency signals
    # -----------------------------------------------------

    URGENCY_TERMS = {

        "immediately","urgent","now","rapidly","quickly",
        "instantly","suddenly","swiftly",

        "critical","pressing","dire","serious",

        "act_now","time_is_running_out",
    }

    # -----------------------------------------------------

    def __init__(self, config: NarrativeTemporalConfig | None = None):

        self.config = config or NarrativeTemporalConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info("NarrativeTemporalAnalyzer initialized")


    # -----------------------------------------------------
    # Main analysis
    # -----------------------------------------------------

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

        features.update(self._term_ratio(tokens, self.PAST_TERMS, "past_framing_ratio"))
        features.update(self._term_ratio(tokens, self.CRISIS_TERMS, "crisis_escalation_ratio"))
        features.update(self._term_ratio(tokens, self.URGENCY_TERMS, "urgency_language_ratio"))

        features.update(self._tense_distribution(doc))
        features.update(self._temporal_contrast(features))

        logger.debug("Temporal narrative features computed")

        return features


    # -----------------------------------------------------
    # Term ratio helper
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # Verb tense distribution
    # -----------------------------------------------------

    def _tense_distribution(self, doc: Doc) -> Dict[str, float]:

        verbs = [token for token in doc if token.pos_ == "VERB"]

        if not verbs:
            return {
                "past_tense_ratio": 0.0,
                "present_tense_ratio": 0.0,
                "future_tense_ratio": 0.0,
            }

        past = 0
        present = 0
        future = 0

        for token in verbs:

            tag = token.tag_

            if tag in {"VBD","VBN"}:
                past += 1

            elif tag in {"VBZ","VBP","VBG"}:
                present += 1

            if token.text.lower() in {"will","shall"}:
                future += 1

        total = max(len(verbs), 1)

        return {

            "past_tense_ratio": past / total,
            "present_tense_ratio": present / total,
            "future_tense_ratio": future / total,
        }


    # -----------------------------------------------------
    # Temporal narrative contrast
    # -----------------------------------------------------

    def _temporal_contrast(self, features: Dict[str, float]) -> Dict[str, float]:

        past = features.get("past_framing_ratio", 0.0)
        urgency = features.get("urgency_language_ratio", 0.0)

        contrast = abs(past - urgency)

        return {"temporal_contrast_score": contrast}


# ---------------------------------------------------------
# Vector Conversion
# ---------------------------------------------------------

def narrative_temporal_vector(features: Dict[str, float]) -> np.ndarray:

    ordered_keys = [

        "past_framing_ratio",
        "crisis_escalation_ratio",
        "urgency_language_ratio",

        "past_tense_ratio",
        "present_tense_ratio",
        "future_tense_ratio",

        "temporal_contrast_score",
    ]

    return np.array(
        [float(features.get(k, 0.0)) for k in ordered_keys],
        dtype=np.float32,
    )