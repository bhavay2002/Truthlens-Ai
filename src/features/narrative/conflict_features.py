"""
File Name: conflict_features.py
Module: Feature Engineering - Conflict Features
Description:
    Extracts linguistic indicators of conflict, confrontation, and adversarial
    discourse within text.
    
    The module identifies signals frequently present in:
    
    - political rhetoric
    - ideological narratives
    - propaganda
    - polarizing media discourse
    
    Signals include:
    
    1. Confrontation language
    2. Dispute / argument framing
    3. Accusation language
    4. Aggressive rhetoric
    5. Polarization language ("us vs them")
    6. Escalation signals
    
    These features support downstream models analyzing narrative dynamics,
    media framing, and misinformation patterns.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing conflict-related discourse indicators
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
# Tokenization fallback
# ---------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Basic tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------
# Conflict Lexicons (Research Level)
# ---------------------------------------------------------

CONFRONTATION_TERMS: Set[str] = {
    "fight","battle","clash","attack","war","confront","struggle",
    "showdown","standoff","conflict","retaliate","counterattack",
}

DISPUTE_TERMS: Set[str] = {
    "dispute","argument","debate","disagreement","controversy",
    "criticized","criticise","criticize","challenge","oppose",
}

ACCUSATION_TERMS: Set[str] = {
    "accuse","blame","fault","responsible","allege","charged",
    "condemn","denounce","claim","suspect",
}

AGGRESSIVE_LANGUAGE: Set[str] = {
    "destroy","defeat","threat","enemy","hostile",
    "attack","bomb","kill","eliminate","retaliate",
}

POLARIZATION_TERMS: Set[str] = {
    "us","them","they","enemy","opponent","outsiders",
    "elite","establishment","radicals","extremists",
}

ESCALATION_TERMS: Set[str] = {
    "crisis","chaos","collapse","emergency","disaster",
    "catastrophe","meltdown","danger","threat",
}


# ---------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------

EXCLAMATION_PATTERN = re.compile(r"!")
QUESTION_PATTERN = re.compile(r"\?")


# ---------------------------------------------------------
# Feature Class
# ---------------------------------------------------------

@dataclass
@register_feature
class ConflictFeatures(BaseFeature):
    """
    Extracts indicators of conflict-oriented discourse.

    Output Features
    ---------------

    conflict_confrontation_ratio
    conflict_dispute_ratio
    conflict_accusation_ratio
    conflict_aggression_ratio
    conflict_polarization_ratio
    conflict_escalation_ratio
    conflict_intensity
    conflict_diversity
    conflict_rhetoric_score
    """

    name: str = "conflict_features"
    description: str = "Conflict and confrontation discourse indicators"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for conflict feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            hits = sum(counter.get(w, 0) for w in lexicon)
            return hits / total_tokens

        confrontation_ratio = ratio(CONFRONTATION_TERMS)
        dispute_ratio = ratio(DISPUTE_TERMS)
        accusation_ratio = ratio(ACCUSATION_TERMS)
        aggression_ratio = ratio(AGGRESSIVE_LANGUAGE)
        polarization_ratio = ratio(POLARIZATION_TERMS)
        escalation_ratio = ratio(ESCALATION_TERMS)

        values = [
            confrontation_ratio,
            dispute_ratio,
            accusation_ratio,
            aggression_ratio,
            polarization_ratio,
            escalation_ratio,
        ]

        # overall intensity
        intensity = sum(values) / len(values)

        # diversity of conflict signals
        diversity = sum(1 for v in values if v > 0) / len(values)

        # rhetorical emphasis
        exclamations = len(EXCLAMATION_PATTERN.findall(context.text))
        questions = len(QUESTION_PATTERN.findall(context.text))
        rhetoric_score = (exclamations + questions) / max(len(context.text), 1)

        features: Dict[str, float] = {

            "conflict_confrontation_ratio": float(confrontation_ratio),
            "conflict_dispute_ratio": float(dispute_ratio),
            "conflict_accusation_ratio": float(accusation_ratio),
            "conflict_aggression_ratio": float(aggression_ratio),
            "conflict_polarization_ratio": float(polarization_ratio),
            "conflict_escalation_ratio": float(escalation_ratio),

            "conflict_intensity": float(intensity),
            "conflict_diversity": float(diversity),

            "conflict_rhetoric_score": float(rhetoric_score),
        }

        logger.debug(
            "Conflict features extracted | intensity=%.4f diversity=%.4f",
            intensity,
            diversity,
        )

        return features