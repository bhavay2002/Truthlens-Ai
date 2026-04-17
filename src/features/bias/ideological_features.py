"""
File Name: ideological_features.py
Module: Feature Engineering - Ideological Features
Description:
    Extracts ideology-related linguistic signals from text by matching tokens
    against curated ideological lexicons and computing distributional metrics.
    The module produces normalized ratios for left/right ideological cues,
    polarization indicators, and balance metrics. Designed to be lightweight,
    deterministic, and compatible with the TruthLens feature pipeline.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing ideology and polarization indicators
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
    """Tokenizer optimized for ideological signal detection."""
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def _count(counter: Counter, lexicon: Set[str]) -> int:
    return sum(counter.get(w, 0) for w in lexicon)


def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    if total == 0:
        return 0.0
    return _count(counter, lexicon) / total


# ---------------------------------------------------------
# Ideological Lexicons
# ---------------------------------------------------------

LEFT_LEXICON: Set[str] = {
    "equality","equity","justice","progressive",
    "climate","environment","regulation",
    "welfare","diversity","inclusion",
    "redistribution","public","universal",
    "labor","union","social","collective",
    "fairness","rights"
}

RIGHT_LEXICON: Set[str] = {
    "freedom","liberty","market","markets",
    "tax","taxes","security","patriot",
    "tradition","border","sovereignty",
    "private","deregulation","military",
    "law","order","national","authority"
}

POLARIZING_TERMS: Set[str] = {
    "elite","establishment","radical","extremist",
    "corrupt","enemy","attack","threat",
    "ideology","agenda","propaganda"
}

GROUP_REFERENCES: Set[str] = {
    "they","them","those","these",
    "people","group","community",
    "citizens","supporters","critics"
}

# ideological framing phrases
IDEOLOGY_PHRASES = [
    r"the\s+left",
    r"the\s+right",
    r"political\s+elite",
    r"radical\s+agenda",
]

COMPILED_IDEOLOGY_PHRASES = [re.compile(p) for p in IDEOLOGY_PHRASES]


# ---------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------

@dataclass
@register_feature
class IdeologicalFeatures(BaseFeature):

    """
    Extract ideology and polarization indicators.

    Output Features
    ---------------
    ideology_left_ratio
    ideology_right_ratio
    ideology_balance
    ideology_polarization_ratio
    ideology_group_reference_ratio
    ideology_phrase_count
    ideology_entropy
    ideology_signal_strength
    """

    name: str = "ideological_features"
    description: str = "Ideological framing and polarization signals"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")
        if not context.text.strip():
            return {}

        text = context.text
        text_lower = text.lower()
        tokens = context.tokens or _tokenize(text_lower)

        if not tokens:
            logger.warning("No tokens available for ideological feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        left_ratio = _ratio(counter, LEFT_LEXICON, total_tokens)
        right_ratio = _ratio(counter, RIGHT_LEXICON, total_tokens)
        polarization_ratio = _ratio(counter, POLARIZING_TERMS, total_tokens)
        group_ref_ratio = _ratio(counter, GROUP_REFERENCES, total_tokens)

        # -------------------------------------------------
        # phrase detection
        # -------------------------------------------------

        phrase_count = sum(bool(p.search(text_lower)) for p in COMPILED_IDEOLOGY_PHRASES)

        # -------------------------------------------------
        # balance metric
        # -------------------------------------------------

        signal = left_ratio + right_ratio
        balance = 1.0 - (abs(left_ratio - right_ratio) / signal) if signal > 0 else 0.0

        # -------------------------------------------------
        # entropy of ideological distribution
        # -------------------------------------------------

        arr = np.array([left_ratio, right_ratio], dtype=float)

        if arr.sum() > 0:
            probs = arr / arr.sum()
            entropy = -float((probs * np.log(probs + 1e-9)).sum())
        else:
            entropy = 0.0

        # -------------------------------------------------
        # signal strength
        # -------------------------------------------------

        signal_strength = (
            left_ratio +
            right_ratio +
            polarization_ratio
        ) / 3.0

        features: Dict[str, float] = {

            "ideology_left_ratio": left_ratio,
            "ideology_right_ratio": right_ratio,

            "ideology_balance": balance,
            "ideology_entropy": entropy,

            "ideology_polarization_ratio": polarization_ratio,
            "ideology_group_reference_ratio": group_ref_ratio,

            "ideology_phrase_count": float(phrase_count),

            "ideology_signal_strength": signal_strength,
        }

        logger.debug(
            "Ideological features extracted | left=%.4f right=%.4f balance=%.4f",
            left_ratio,
            right_ratio,
            balance,
        )

        return features