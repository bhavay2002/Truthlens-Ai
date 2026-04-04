"""
File Name: manipulation_patterns.py
Module: Feature Engineering - Propaganda / Manipulation Patterns
Description:
   The module extracts interpretable indicators used in misinformation
    analysis pipelines. Patterns are based on research from:

• Propaganda detection literature
• Computational rhetoric analysis
• Disinformation detection studies
• Political communication research

Detected manipulation strategies include:

    - Urgency framing
    - Fear appeals
    - Blame attribution
    - Scapegoating language
    - Absolutist rhetoric
    - Conspiracy framing
    - False dilemmas
    - Sensational exaggeration
    - Emotional intensifiers

The implementation remains deterministic and lightweight,
making it suitable for large-scale dataset processing.

Outputs integrate directly with TruthLens feature pipelines.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing text and optional tokens

Outputs:
    Dict[str, float] representing manipulation pattern indicators
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

TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


def _tokenize(text: str) -> List[str]:
    """Research-grade tokenizer for lexical pattern detection."""
    return TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------
# Manipulation Lexicons
# ---------------------------------------------------------

# ---------------------------------------------------------
# Urgency / Call-to-Action Framing
# ---------------------------------------------------------

URGENCY_TERMS: Set[str] = {
    "urgent", "urgently", "immediately", "instant", "instantly",
    "now", "today", "quick", "quickly",
    "act", "respond",
    "hurry", "rush",
    "breaking", "alert", "warning", "crisis",
    "emergency", "critical", "time", "deadline"
}


# ---------------------------------------------------------
# Fear / Threat Framing
# ---------------------------------------------------------

FEAR_TERMS: Set[str] = {
    "threat", "danger", "attack", "attacks",
    "terror", "terrorist", "risk",
    "collapse", "breakdown", "destruction",
    "catastrophe", "catastrophic",
    "disaster", "crisis", "panic",
    "fear", "chaos", "anarchy",
    "invasion", "takeover", "war",
    "threatening", "endanger", "endangered"
}


# ---------------------------------------------------------
# Blame Attribution
# ---------------------------------------------------------

BLAME_TERMS: Set[str] = {
    "blame", "fault", "responsible", "responsibility",
    "caused", "cause", "created",
    "betrayed", "betrayal",
    "failed", "failure",
    "destroyed", "ruined",
    "corrupt", "corruption",
    "lied", "lying",
    "misled", "deceived",
    "guilty"
}


# ---------------------------------------------------------
# Scapegoating / Out-Group Framing
# ---------------------------------------------------------

SCAPEGOAT_TERMS: Set[str] = {
    "they", "them", "their",
    "outsiders", "foreigners",
    "immigrants", "migrants",
    "elites", "globalists",
    "establishment", "bureaucrats",
    "media", "mainstream",
    "politicians", "government",
    "corporations",
    "liberals", "conservatives"
}


# ---------------------------------------------------------
# Absolutist / Overgeneralized Claims
# ---------------------------------------------------------

ABSOLUTE_TERMS: Set[str] = {
    "always", "never",
    "everyone", "everybody",
    "nobody",
    "all", "none",
    "everything", "nothing",
    "completely", "entirely",
    "totally", "absolutely",
    "certainly"
}


# ---------------------------------------------------------
# Conspiracy / Hidden Truth Framing
# ---------------------------------------------------------

CONSPIRACY_TERMS: Set[str] = {
    "secret", "hidden", "cover",
    "exposed", "exposing",
    "truth",
    "agenda", "scheme",
    "plot", "conspiracy",
    "they", "know",
    "controlled", "manipulated",
    "puppet",
    "propaganda"
}


# ---------------------------------------------------------
# False Dilemma / Binary Framing
# ---------------------------------------------------------

FALSE_DILEMMA_TERMS: Set[str] = {
    "either", "or",
    "choice", "choose",
    "only", "option",
    "must", "forced",
    "inevitable",
}


# ---------------------------------------------------------
# Sensationalism / Exaggeration
# ---------------------------------------------------------

EXAGGERATION_TERMS: Set[str] = {
    "shocking", "unbelievable", "incredible",
    "outrageous", "scandalous",
    "massive", "huge", "giant",
    "explosive", "bombshell",
    "devastating", "dramatic",
    "stunning",
    "historic",
}


# ---------------------------------------------------------
# Emotional Intensifiers
# ---------------------------------------------------------

INTENSIFIERS: Set[str] = {
    "very", "extremely", "incredibly",
    "absolutely", "completely",
    "totally", "highly",
    "deeply", "strongly",
    "seriously", "truly"
}

# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def _ratio(counter: Counter, lexicon: Set[str], total: int) -> float:
    count = sum(counter.get(w, 0) for w in lexicon)
    return count / total if total > 0 else 0.0


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class ManipulationPatterns(BaseFeature):

    """
    Detect rhetorical manipulation strategies.

    Output Features
    ----------------

    manipulation_urgency_ratio
    manipulation_fear_ratio
    manipulation_blame_ratio
    manipulation_scapegoat_ratio
    manipulation_absolute_ratio
    manipulation_conspiracy_ratio
    manipulation_false_dilemma_ratio
    manipulation_exaggeration_ratio
    manipulation_intensifier_ratio

    manipulation_exclamation_density
    manipulation_caps_emphasis

    manipulation_intensity
    manipulation_diversity
    """

    name: str = "manipulation_patterns"
    description: str = "Propaganda and manipulation language indicators"

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        text = context.text

        tokens = context.tokens or _tokenize(text)

        if not tokens:
            logger.warning("No tokens available for manipulation analysis")
            return {}

        total_tokens = len(tokens)
        counter = Counter(tokens)

        urgency = _ratio(counter, URGENCY_TERMS, total_tokens)
        fear = _ratio(counter, FEAR_TERMS, total_tokens)
        blame = _ratio(counter, BLAME_TERMS, total_tokens)
        scapegoat = _ratio(counter, SCAPEGOAT_TERMS, total_tokens)
        absolute = _ratio(counter, ABSOLUTE_TERMS, total_tokens)
        conspiracy = _ratio(counter, CONSPIRACY_TERMS, total_tokens)
        dilemma = _ratio(counter, FALSE_DILEMMA_TERMS, total_tokens)
        exaggeration = _ratio(counter, EXAGGERATION_TERMS, total_tokens)
        intensifier = _ratio(counter, INTENSIFIERS, total_tokens)

        # -------------------------------------------------
        # Structural heuristics
        # -------------------------------------------------

        exclamation_density = text.count("!") / max(len(text), 1)

        caps_tokens = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
        caps_ratio = caps_tokens / total_tokens

        values = [
            urgency,
            fear,
            blame,
            scapegoat,
            absolute,
            conspiracy,
            dilemma,
            exaggeration,
            intensifier,
        ]

        intensity = sum(values) / len(values)
        diversity = sum(1 for v in values if v > 0) / len(values)

        features: Dict[str, float] = {

            "manipulation_urgency_ratio": urgency,
            "manipulation_fear_ratio": fear,
            "manipulation_blame_ratio": blame,
            "manipulation_scapegoat_ratio": scapegoat,
            "manipulation_absolute_ratio": absolute,
            "manipulation_conspiracy_ratio": conspiracy,
            "manipulation_false_dilemma_ratio": dilemma,
            "manipulation_exaggeration_ratio": exaggeration,
            "manipulation_intensifier_ratio": intensifier,

            "manipulation_exclamation_density": exclamation_density,
            "manipulation_caps_emphasis": caps_ratio,

            "manipulation_intensity": intensity,
            "manipulation_diversity": diversity,
        }

        logger.debug(
            "Manipulation analysis | intensity=%.4f diversity=%.4f",
            intensity,
            diversity,
        )

        return features