"""
File Name: framing_features.py
Module: Feature Engineering - Framing Features
Description:
    Extracts narrative framing signals from text. The module identifies
    common political and journalistic frames (economic, moral, security,
    human-interest, conflict) using lexicon and structural indicators.
    The extracted features help quantify how information is framed and
    presented in the text, which is useful for detecting narrative bias
    and agenda-setting patterns.

    The implementation integrates with the TruthLens feature framework
    using BaseFeature and FeatureRegistry, enabling modular feature
    extraction and configuration-driven pipelines.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing narrative framing indicators
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    """Tokenizer optimized for narrative frame analysis."""
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def _count(counter: Counter, lexicon: Set[str]) -> int:
    return sum(counter.get(w, 0) for w in lexicon)


def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    if total == 0:
        return 0.0
    return _count(counter, lexicon) / total


# ---------------------------------------------------------
# Frame Lexicons
# ---------------------------------------------------------
ECONOMIC_FRAME: Set[str] = {
    "economy","economic","tax","taxes","market","trade",
    "budget","inflation","investment","jobs","industry",
    "growth","recession","finance","spending","debt",
    "employment","income","wages","funding","cost"
}

MORAL_FRAME: Set[str] = {
    "moral","ethical","ethics","justice","values",
    "rights","fair","fairness","duty","responsibility",
    "principle","virtue","honor","integrity",
    "conscience","morality"
}

SECURITY_FRAME: Set[str] = {
    "security","defense","threat","terrorism",
    "military","attack","war","protection",
    "safety","risk","danger","border","intelligence",
    "surveillance","counterterrorism"
}

HUMAN_INTEREST_FRAME: Set[str] = {
    "family","community","children","people",
    "victim","story","life","personal",
    "citizens","individuals","families",
    "workers","residents","suffering",
    "experience","daily"
}

CONFLICT_FRAME: Set[str] = {
    "conflict","fight","battle","clash",
    "opposition","dispute","debate",
    "criticized","criticism","confrontation",
    "tension","rivalry","political"
}


# ---------------------------------------------------------
# Phrase-based framing patterns
# ---------------------------------------------------------

FRAME_PHRASES = [

    r"critics\s+argue",
    r"supporters\s+say",
    r"according\s+to\s+officials",
    r"analysts\s+believe",

    r"the\s+debate\s+over",
    r"amid\s+growing\s+concerns",
]

# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class FramingFeatures(BaseFeature):

    """
    Extract narrative frame indicators.

    Output Features
    ---------------

    frame_economic_ratio
    frame_moral_ratio
    frame_security_ratio
    frame_human_interest_ratio
    frame_conflict_ratio
    frame_responsibility_ratio
    frame_policy_ratio
    frame_crisis_ratio
    frame_identity_ratio

    frame_phrase_count
    frame_quote_density

    frame_diversity
    frame_dominance
    frame_entropy
    """

    name: str = "framing_features"
    description: str = "Narrative framing indicators"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        text = context.text
        tokens = context.tokens or _tokenize(text)

        if not tokens:
            logger.warning("No tokens available for framing feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        econ = _ratio(counter, ECONOMIC_FRAME, total_tokens)
        moral = _ratio(counter, MORAL_FRAME, total_tokens)
        security = _ratio(counter, SECURITY_FRAME, total_tokens)
        human = _ratio(counter, HUMAN_INTEREST_FRAME, total_tokens)
        conflict = _ratio(counter, CONFLICT_FRAME, total_tokens)
        

        frame_values = [
            econ, moral, security, human,
            conflict
        ]

        # -------------------------------------------------
        # phrase detection
        # -------------------------------------------------

        phrase_count = sum(
            bool(re.search(p, text.lower()))
            for p in FRAME_PHRASES
        )

        # -------------------------------------------------
        # structural narrative signals
        # -------------------------------------------------

        quote_count = text.count('"') + text.count("'")
        quote_density = quote_count / max(len(text), 1)

        # -------------------------------------------------
        # distribution metrics
        # -------------------------------------------------

        diversity = sum(1 for v in frame_values if v > 0) / len(frame_values)

        dominance = max(frame_values)

        arr = np.array(frame_values, dtype=float)

        if arr.sum() > 0:
            probs = arr / arr.sum()
            entropy = -float((probs * np.log(probs + 1e-9)).sum())
        else:
            entropy = 0.0

        features: Dict[str, float] = {

            "frame_economic_ratio": econ,
            "frame_moral_ratio": moral,
            "frame_security_ratio": security,
            "frame_human_interest_ratio": human,
            "frame_conflict_ratio": conflict,
            
            "frame_phrase_count": float(phrase_count),
            "frame_quote_density": quote_density,

            "frame_diversity": diversity,
            "frame_dominance": dominance,
            "frame_entropy": entropy,
        }

        logger.debug(
            "Framing features extracted | dominance=%.4f diversity=%.4f entropy=%.4f",
            dominance,
            diversity,
            entropy,
        )

        return features