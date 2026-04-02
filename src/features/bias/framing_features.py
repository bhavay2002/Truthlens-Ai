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

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------
# Frame Lexicons
# ---------------------------------------------------------------------

ECONOMIC_FRAME: Set[str] = {
    "economy", "tax", "market", "trade", "budget", "inflation",
    "investment", "jobs", "industry", "growth"
}

MORAL_FRAME: Set[str] = {
    "moral", "ethical", "justice", "values", "rights",
    "fair", "duty", "responsibility"
}

SECURITY_FRAME: Set[str] = {
    "security", "defense", "threat", "terrorism",
    "military", "attack", "war", "protection"
}

HUMAN_INTEREST_FRAME: Set[str] = {
    "family", "community", "children", "people",
    "victim", "story", "life", "personal"
}

CONFLICT_FRAME: Set[str] = {
    "conflict", "fight", "battle", "clash",
    "opposition", "dispute", "debate", "criticized"
}


@dataclass
@register_feature
class FramingFeatures(BaseFeature):
    """
    Extracts narrative frame indicators.

    Output Features
    ---------------
    frame_economic_ratio
    frame_moral_ratio
    frame_security_ratio
    frame_human_interest_ratio
    frame_conflict_ratio
    frame_diversity
    frame_dominance
    """

    name: str = "framing_features"
    description: str = "Narrative framing indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for framing feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        econ = ratio(ECONOMIC_FRAME)
        moral = ratio(MORAL_FRAME)
        security = ratio(SECURITY_FRAME)
        human = ratio(HUMAN_INTEREST_FRAME)
        conflict = ratio(CONFLICT_FRAME)

        frame_values = [econ, moral, security, human, conflict]

        diversity = sum(1 for v in frame_values if v > 0) / len(frame_values)
        dominance = max(frame_values)

        features: Dict[str, float] = {
            "frame_economic_ratio": float(econ),
            "frame_moral_ratio": float(moral),
            "frame_security_ratio": float(security),
            "frame_human_interest_ratio": float(human),
            "frame_conflict_ratio": float(conflict),
            "frame_diversity": float(diversity),
            "frame_dominance": float(dominance),
        }

        logger.debug(
            "Framing features extracted | dominance=%.4f diversity=%.4f",
            dominance,
            diversity,
        )

        return features