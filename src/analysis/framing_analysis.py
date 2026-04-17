"""
File Name: framing_analysis.py
Module: Narrative Analysis - Media Framing Detection
Description:
    Detects media framing strategies within text for the TruthLens AI system.
    The module analyzes linguistic indicators associated with common framing
    strategies studied in political communication and media analysis research.
    These include responsibility framing, economic framing, moral framing,
    human interest framing, and conflict framing.

    The extracted features help quantify how an issue is framed within a text,
    allowing downstream modules to model narrative bias, ideological messaging,
    and propaganda patterns.

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
    Frame feature dictionary and optional numerical vector
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc

from src.analysis._nlp import get_nlp
from src.analysis._text_features import extract_alpha_lemmas, build_counter, term_ratio as _term_ratio_util
from src.analysis.feature_schema import FRAMING_KEYS, make_vector


logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

@dataclass(slots=True)
class FramingAnalysisConfig:

    spacy_model: str = "en_core_web_sm"
    disable_components: tuple = ("ner",)


# ------------------------------------------------------------
# Framing Analyzer
# ------------------------------------------------------------

class FramingAnalyzer:

    # ----------------------------------------------------
    # Conflict Frame
    # ----------------------------------------------------

    CONFLICT_TERMS = {

        "conflict","fight","battle","war","clash","attack","confront",
        "dispute","rival","struggle","tension","hostility","standoff",
        "confrontation","showdown","retaliation","counterattack",
        "escalation","political_fight","power_struggle","ideological_clash"
    }

    # ----------------------------------------------------
    # Economic Frame
    # ----------------------------------------------------

    ECONOMIC_TERMS = {

        "economy","economic","market","markets",
        "jobs","employment","unemployment","labor",
        "tax","taxes","trade","budget","spending",
        "cost","financial","finance","growth",
        "inflation","investment","recession",
        "deficit","debt","revenue","funding",
        "economic_growth","economic_policy","fiscal_policy"
    }

    # ----------------------------------------------------
    # Moral / Ethical Frame
    # ----------------------------------------------------

    MORAL_TERMS = {

        "moral","ethic","ethics","value","values",
        "justice","fairness","right","wrong",
        "duty","principle","virtue","integrity",
        "honor","honour","conscience","morality",
        "ethical","responsibility","moral_obligation",
        "social_justice","human_rights"
    }

    # ----------------------------------------------------
    # Human Interest Frame
    # ----------------------------------------------------

    HUMAN_INTEREST_TERMS = {

        "family","children","child","community",
        "people","citizen","victim","life",
        "story","personal_story","emotion",
        "suffering","experience","personal",
        "struggle","hardship","tragedy",
        "survivor","human_impact","daily_life"
    }

    # ----------------------------------------------------
    # Security / Threat Frame
    # ----------------------------------------------------

    SECURITY_TERMS = {

        "security","national_security","safety",
        "threat","risk","danger","crisis",
        "terror","terrorism","extremism",
        "attack","defense","protection",
        "surveillance","law_enforcement",
        "border_security","military",
        "counterterrorism","emergency",
        "public_safety"
    }

    # ----------------------------------------------------

    def __init__(self, config: FramingAnalysisConfig | None = None):

        self.config = config or FramingAnalysisConfig()

        self.nlp: Language = get_nlp(
            self.config.spacy_model,
            disable=self.config.disable_components,
        )

        logger.info(
            "FramingAnalyzer initialized | model=%s",
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
        """Compute framing features from a pre-built spaCy Doc.

        Builds the token counter once and reuses it for all frame scores,
        avoiding repeated Counter construction.

        Args:
            doc: A processed spaCy Doc instance.

        Returns:
            Dictionary of framing feature names to float values.
        """

        tokens: List[str] = extract_alpha_lemmas(doc)
        token_counts = build_counter(tokens)
        n_tokens = len(tokens)

        features: Dict[str, float] = {}

        features["frame_conflict_score"] = _term_ratio_util(
            token_counts, n_tokens, self.CONFLICT_TERMS
        )
        features["frame_economic_score"] = _term_ratio_util(
            token_counts, n_tokens, self.ECONOMIC_TERMS
        )
        features["frame_moral_score"] = _term_ratio_util(
            token_counts, n_tokens, self.MORAL_TERMS
        )
        features["frame_human_interest_score"] = _term_ratio_util(
            token_counts, n_tokens, self.HUMAN_INTEREST_TERMS
        )
        features["frame_security_score"] = _term_ratio_util(
            token_counts, n_tokens, self.SECURITY_TERMS
        )

        features.update(self._frame_dominance(features))
        features.update(self._frame_diversity(features))

        return features

    # ------------------------------------------------------------
    # Frame scoring (kept for backward compatibility)
    # ------------------------------------------------------------

    def _frame_score(
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
    # Frame dominance
    # ------------------------------------------------------------

    def _frame_dominance(self, features: Dict[str, float]) -> Dict[str, float]:

        frame_scores = [
            v for k, v in features.items()
            if k.startswith("frame_")
        ]

        if not frame_scores:
            return {"frame_dominance_score": 0.0}

        return {"frame_dominance_score": float(max(frame_scores))}

    # ------------------------------------------------------------
    # Frame diversity
    # ------------------------------------------------------------

    _BASE_FRAME_KEYS = {
        "frame_conflict_score",
        "frame_economic_score",
        "frame_moral_score",
        "frame_human_interest_score",
        "frame_security_score",
    }

    def _frame_diversity(self, features: Dict[str, float]) -> Dict[str, float]:

        frame_scores = [
            v for k, v in features.items()
            if k in self._BASE_FRAME_KEYS
        ]

        active_frames = sum(1 for score in frame_scores if score > 0)

        diversity = active_frames / max(len(frame_scores), 1)

        return {"frame_diversity_score": float(diversity)}


# ------------------------------------------------------------
# Vector conversion
# ------------------------------------------------------------

def framing_feature_vector(features: Dict[str, float]) -> np.ndarray:

    return make_vector(features, FRAMING_KEYS)