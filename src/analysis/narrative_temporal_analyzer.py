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
from spacy.language import Language
from spacy.tokens import Doc

from src.analysis._nlp import get_nlp
from src.analysis._text_features import extract_alpha_lemmas, build_counter, term_ratio as _term_ratio_util
from src.analysis.feature_schema import NARRATIVE_TEMPORAL_KEYS, make_vector


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

        self.nlp: Language = get_nlp(self.config.spacy_model)

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
        return self.analyze_doc(doc)

    # -----------------------------------------------------

    def analyze_doc(self, doc: Doc) -> Dict[str, float]:
        """Compute temporal narrative features from a pre-built spaCy Doc.

        Builds the token counter once and reuses it across all term-ratio
        computations, eliminating repeated Counter construction.

        Args:
            doc: A processed spaCy Doc instance.

        Returns:
            Dictionary of temporal narrative feature names to float values.
        """

        tokens: List[str] = extract_alpha_lemmas(doc)
        token_counts = build_counter(tokens)
        n_tokens = len(tokens)

        features: Dict[str, float] = {}

        features["past_framing_ratio"] = _term_ratio_util(token_counts, n_tokens, self.PAST_TERMS)
        features["crisis_escalation_ratio"] = _term_ratio_util(token_counts, n_tokens, self.CRISIS_TERMS)
        features["urgency_language_ratio"] = _term_ratio_util(token_counts, n_tokens, self.URGENCY_TERMS)

        features.update(self._tense_distribution(doc))
        features.update(self._temporal_contrast(features))

        logger.debug("Temporal narrative features computed")

        return features


    # -----------------------------------------------------
    # Term ratio helper (kept for backward compatibility)
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

    return make_vector(features, NARRATIVE_TEMPORAL_KEYS)