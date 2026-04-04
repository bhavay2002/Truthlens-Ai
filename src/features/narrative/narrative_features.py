"""
File Name: narrative_features.py
Module: Feature Engineering - Narrative Features
Description:
    Extracts narrative structure indicators from text.

    The module detects narrative storytelling patterns commonly used in
    journalism, political messaging, and propaganda narratives.
    
    Signals include:
    
    1. Narrative roles (hero / villain / victim)
    2. Conflict framing
    3. Crisis escalation language
    4. Narrative resolution signals
    5. Polarization language
    6. Narrative progression patterns
    
    The implementation is deterministic and lexicon-based, allowing
    lightweight feature extraction within the TruthLens pipeline.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing narrative structure indicators
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
    """Fallback tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# Narrative Lexicons (Research Level)
# ---------------------------------------------------------

HERO_TERMS: Set[str] = {

    "hero","leader","defender","champion",
    "protect","save","rescue","support",
    "aid","assist","defend","help"
}

VILLAIN_TERMS: Set[str] = {

    "villain","enemy","corrupt","attacker",
    "threat","destroy","betray","abuse",
    "exploit","oppress","manipulate"
}

VICTIM_TERMS: Set[str] = {

    "victim","suffer","harm","damage",
    "loss","injured","affected","targeted",
    "displaced","hurt"
}

CONFLICT_TERMS: Set[str] = {

    "conflict","battle","fight","clash",
    "dispute","attack","war","showdown",
    "standoff","confrontation"
}

RESOLUTION_TERMS: Set[str] = {

    "resolve","agreement","peace","solution",
    "settlement","deal","compromise",
    "negotiation","reconciliation"
}

CRISIS_TERMS: Set[str] = {

    "crisis","emergency","disaster",
    "collapse","panic","chaos",
    "catastrophe","meltdown"
}

POLARIZATION_TERMS: Set[str] = {

    "us","them","enemy","opponent",
    "elite","establishment","outsiders",
    "radicals","extremists"
}


EXCLAMATION_PATTERN = re.compile(r"!")
QUESTION_PATTERN = re.compile(r"\?")


# ---------------------------------------------------------
# Feature Class
# ---------------------------------------------------------

@dataclass
@register_feature
class NarrativeFeatures(BaseFeature):
    """
    Extract narrative storytelling indicators.

    Output Features
    ---------------
    narrative_hero_ratio
    narrative_villain_ratio
    narrative_victim_ratio
    narrative_conflict_ratio
    narrative_resolution_ratio
    narrative_crisis_ratio
    narrative_polarization_ratio
    narrative_role_diversity
    narrative_conflict_intensity
    narrative_progression_score
    narrative_rhetoric_score
    """

    name: str = "narrative_features"
    description: str = "Narrative structure and role framing indicators"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for narrative feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            hits = sum(counter.get(w, 0) for w in lexicon)
            return hits / total_tokens

        hero_ratio = ratio(HERO_TERMS)
        villain_ratio = ratio(VILLAIN_TERMS)
        victim_ratio = ratio(VICTIM_TERMS)

        conflict_ratio = ratio(CONFLICT_TERMS)
        resolution_ratio = ratio(RESOLUTION_TERMS)
        crisis_ratio = ratio(CRISIS_TERMS)

        polarization_ratio = ratio(POLARIZATION_TERMS)

        # -------------------------------------------------
        # Role diversity
        # -------------------------------------------------

        role_values = [hero_ratio, villain_ratio, victim_ratio]

        role_diversity = sum(1 for v in role_values if v > 0) / len(role_values)

        # -------------------------------------------------
        # Conflict intensity
        # -------------------------------------------------

        conflict_intensity = (conflict_ratio + crisis_ratio) / 2.0

        # -------------------------------------------------
        # Narrative progression
        # conflict -> resolution structure
        # -------------------------------------------------

        progression_score = resolution_ratio - conflict_ratio

        # -------------------------------------------------
        # Rhetorical emphasis
        # -------------------------------------------------

        exclamations = len(EXCLAMATION_PATTERN.findall(context.text))
        questions = len(QUESTION_PATTERN.findall(context.text))

        rhetoric_score = (exclamations + questions) / max(len(context.text), 1)

        # -------------------------------------------------

        features: Dict[str, float] = {

            "narrative_hero_ratio": float(hero_ratio),
            "narrative_villain_ratio": float(villain_ratio),
            "narrative_victim_ratio": float(victim_ratio),

            "narrative_conflict_ratio": float(conflict_ratio),
            "narrative_resolution_ratio": float(resolution_ratio),
            "narrative_crisis_ratio": float(crisis_ratio),

            "narrative_polarization_ratio": float(polarization_ratio),

            "narrative_role_diversity": float(role_diversity),
            "narrative_conflict_intensity": float(conflict_intensity),

            "narrative_progression_score": float(progression_score),

            "narrative_rhetoric_score": float(rhetoric_score),
        }

        logger.debug(
            "Narrative features extracted | conflict=%.4f roles=%.4f",
            conflict_intensity,
            role_diversity,
        )

        return features