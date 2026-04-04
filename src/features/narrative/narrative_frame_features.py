"""
File Name: narrative_frame_features.py
Module: Feature Engineering - Narrative Frame Features
Description:
    Extracts narrative framing signals from text using the five
    frames present in the dataset:
    
    1. Conflict (CO)
    2. Economic (EC)
    3. Human Interest (HI)
    4. Moral (MO)
    5. Responsibility (RE)
    
    These frames are widely used in media framing research and
    political communication analysis.
    
    The implementation uses lightweight lexicon-based detection
    to produce deterministic features suitable for ML pipelines.

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


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Simple tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# Frame Lexicons (Aligned with Dataset Frames)
# ---------------------------------------------------------

CONFLICT_FRAME: Set[str] = {
    "conflict","battle","fight","clash","dispute",
    "attack","war","confrontation","showdown","rivalry"
}

ECONOMIC_FRAME: Set[str] = {
    "economy","tax","trade","budget","inflation",
    "market","jobs","growth","investment",
    "recession","financial","industry"
}

HUMAN_INTEREST_FRAME: Set[str] = {
    "family","community","children","people",
    "citizens","victim","story","life",
    "experience","emotion"
}

MORAL_FRAME: Set[str] = {
    "moral","ethics","justice","rights",
    "values","fairness","duty",
    "integrity","principle"
}

RESPONSIBILITY_FRAME: Set[str] = {
    "responsible","blame","accountable",
    "failure","fault","obligation",
    "liability","oversight"
}


EXCLAMATION_PATTERN = re.compile(r"!")
QUESTION_PATTERN = re.compile(r"\?")


# ---------------------------------------------------------
# Feature Class
# ---------------------------------------------------------

@dataclass
@register_feature
class NarrativeFrameFeatures(BaseFeature):
    """
    Extract narrative framing indicators.

    Output Features
    ---------------

    narrative_frame_conflict_ratio
    narrative_frame_economic_ratio
    narrative_frame_human_interest_ratio
    narrative_frame_moral_ratio
    narrative_frame_responsibility_ratio

    narrative_frame_diversity
    narrative_frame_dominance
    narrative_frame_balance
    narrative_frame_rhetoric_score
    """

    name: str = "narrative_frame_features"
    description: str = "Narrative framing indicators (dataset-aligned)"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for narrative frame extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            hits = sum(counter.get(w, 0) for w in lexicon)
            return hits / total_tokens

        conflict_ratio = ratio(CONFLICT_FRAME)
        economic_ratio = ratio(ECONOMIC_FRAME)
        human_ratio = ratio(HUMAN_INTEREST_FRAME)
        moral_ratio = ratio(MORAL_FRAME)
        responsibility_ratio = ratio(RESPONSIBILITY_FRAME)

        frame_values = [
            conflict_ratio,
            economic_ratio,
            human_ratio,
            moral_ratio,
            responsibility_ratio,
        ]

        # -------------------------------------------------
        # Frame diversity
        # -------------------------------------------------

        diversity = sum(1 for v in frame_values if v > 0) / len(frame_values)

        # -------------------------------------------------
        # Dominant frame
        # -------------------------------------------------

        dominance = max(frame_values)

        # -------------------------------------------------
        # Frame balance
        # -------------------------------------------------

        balance = 1.0 - (max(frame_values) - min(frame_values))

        # -------------------------------------------------
        # Rhetorical emphasis
        # -------------------------------------------------

        exclamations = len(EXCLAMATION_PATTERN.findall(context.text))
        questions = len(QUESTION_PATTERN.findall(context.text))

        rhetoric_score = (exclamations + questions) / max(len(context.text), 1)

        # -------------------------------------------------

        features: Dict[str, float] = {

            "narrative_frame_conflict_ratio": float(conflict_ratio),
            "narrative_frame_economic_ratio": float(economic_ratio),
            "narrative_frame_human_interest_ratio": float(human_ratio),
            "narrative_frame_moral_ratio": float(moral_ratio),
            "narrative_frame_responsibility_ratio": float(responsibility_ratio),

            "narrative_frame_diversity": float(diversity),
            "narrative_frame_dominance": float(dominance),
            "narrative_frame_balance": float(balance),

            "narrative_frame_rhetoric_score": float(rhetoric_score),
        }

        logger.debug(
            "Narrative frame features extracted | dominance=%.4f diversity=%.4f",
            dominance,
            diversity,
        )

        return features