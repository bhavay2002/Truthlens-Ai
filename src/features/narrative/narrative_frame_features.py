"""
File Name: narrative_frame_features.py
Module: Feature Engineering - Narrative Frame Features
Description:
    Extracts narrative framing signals from text. Frames represent the
    perspective or lens through which events are presented. This module
    detects common media and political frames such as conflict, economic,
    moral, human-interest, responsibility, and security frames.

    The implementation relies on curated lexicons and lightweight heuristics
    to estimate frame presence and intensity. The extracted features can be
    used for bias analysis, narrative detection, and media framing research.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing narrative frame indicators
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

CONFLICT_FRAME: Set[str] = {
    "conflict", "battle", "fight", "clash", "dispute", "attack", "war"
}

ECONOMIC_FRAME: Set[str] = {
    "economy", "tax", "trade", "budget", "inflation", "market", "jobs"
}

MORAL_FRAME: Set[str] = {
    "moral", "ethics", "justice", "rights", "values", "duty"
}

HUMAN_INTEREST_FRAME: Set[str] = {
    "family", "community", "children", "people", "victim", "story"
}

RESPONSIBILITY_FRAME: Set[str] = {
    "responsible", "blame", "accountable", "failure", "duty"
}

SECURITY_FRAME: Set[str] = {
    "security", "defense", "threat", "terrorism", "protection", "military"
}


@dataclass
@register_feature
class NarrativeFrameFeatures(BaseFeature):
    """
    Extract narrative framing signals.

    Output Features
    ---------------
    narrative_frame_conflict_ratio
    narrative_frame_economic_ratio
    narrative_frame_moral_ratio
    narrative_frame_human_interest_ratio
    narrative_frame_responsibility_ratio
    narrative_frame_security_ratio
    narrative_frame_diversity
    narrative_frame_dominance
    """

    name: str = "narrative_frame_features"
    description: str = "Narrative framing indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """Extract narrative frame features."""
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for narrative frame extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        conflict_ratio = ratio(CONFLICT_FRAME)
        economic_ratio = ratio(ECONOMIC_FRAME)
        moral_ratio = ratio(MORAL_FRAME)
        human_ratio = ratio(HUMAN_INTEREST_FRAME)
        responsibility_ratio = ratio(RESPONSIBILITY_FRAME)
        security_ratio = ratio(SECURITY_FRAME)

        frame_values = [
            conflict_ratio,
            economic_ratio,
            moral_ratio,
            human_ratio,
            responsibility_ratio,
            security_ratio,
        ]

        diversity = sum(1 for v in frame_values if v > 0) / len(frame_values)
        dominance = max(frame_values)

        features: Dict[str, float] = {
            "narrative_frame_conflict_ratio": float(conflict_ratio),
            "narrative_frame_economic_ratio": float(economic_ratio),
            "narrative_frame_moral_ratio": float(moral_ratio),
            "narrative_frame_human_interest_ratio": float(human_ratio),
            "narrative_frame_responsibility_ratio": float(responsibility_ratio),
            "narrative_frame_security_ratio": float(security_ratio),
            "narrative_frame_diversity": float(diversity),
            "narrative_frame_dominance": float(dominance),
        }

        logger.debug(
            "Narrative frame features extracted | dominance=%.4f diversity=%.4f",
            dominance,
            diversity,
        )

        return features