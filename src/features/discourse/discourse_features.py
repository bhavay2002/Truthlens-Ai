"""
File Name: discourse_features.py
Module: Feature Engineering - Discourse Features
Description:
    Extracts discourse-level linguistic signals from text such as discourse
    markers, argumentation structure cues, rhetorical connectors, and
    logical transition indicators. These features help characterize how
    ideas are connected and structured within the text.

    Discourse features are useful for identifying persuasive language,
    argumentative structures, narrative flow, and coherence patterns.

    The implementation relies on curated discourse marker lexicons and
    lightweight heuristics to compute normalized feature ratios.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing discourse structure indicators
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
    """Basic tokenizer fallback."""
    return re.findall(r"\b\w+\b", text.lower())


# ---------------------------------------------------------------------
# Discourse Marker Lexicons
# ---------------------------------------------------------------------

CAUSAL_MARKERS: Set[str] = {
    "because",
    "since",
    "therefore",
    "thus",
    "hence",
    "consequently",
}

CONTRAST_MARKERS: Set[str] = {
    "however",
    "but",
    "although",
    "though",
    "nevertheless",
    "yet",
}

ADDITIVE_MARKERS: Set[str] = {
    "also",
    "furthermore",
    "moreover",
    "additionally",
    "besides",
}

SEQUENTIAL_MARKERS: Set[str] = {
    "first",
    "second",
    "then",
    "next",
    "finally",
}

EVIDENTIAL_MARKERS: Set[str] = {
    "according",
    "reported",
    "evidence",
    "study",
    "data",
    "research",
}


@dataclass
@register_feature
class DiscourseFeatures(BaseFeature):
    """
    Extracts discourse structure indicators.

    Output Features
    ---------------
    discourse_causal_ratio
    discourse_contrast_ratio
    discourse_additive_ratio
    discourse_sequential_ratio
    discourse_evidential_ratio
    discourse_marker_density
    discourse_diversity
    """

    name: str = "discourse_features"
    description: str = "Discourse structure and rhetorical connector features"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """Extract discourse-related features."""
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        tokens = context.tokens or _tokenize(context.text)

        if not tokens:
            logger.warning("No tokens available for discourse feature extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        causal_ratio = ratio(CAUSAL_MARKERS)
        contrast_ratio = ratio(CONTRAST_MARKERS)
        additive_ratio = ratio(ADDITIVE_MARKERS)
        sequential_ratio = ratio(SEQUENTIAL_MARKERS)
        evidential_ratio = ratio(EVIDENTIAL_MARKERS)

        marker_counts = [
            sum(counter.get(w, 0) for w in CAUSAL_MARKERS),
            sum(counter.get(w, 0) for w in CONTRAST_MARKERS),
            sum(counter.get(w, 0) for w in ADDITIVE_MARKERS),
            sum(counter.get(w, 0) for w in SEQUENTIAL_MARKERS),
            sum(counter.get(w, 0) for w in EVIDENTIAL_MARKERS),
        ]

        total_markers = sum(marker_counts)
        marker_density = total_markers / total_tokens

        diversity = sum(1 for c in marker_counts if c > 0) / len(marker_counts)

        features: Dict[str, float] = {
            "discourse_causal_ratio": float(causal_ratio),
            "discourse_contrast_ratio": float(contrast_ratio),
            "discourse_additive_ratio": float(additive_ratio),
            "discourse_sequential_ratio": float(sequential_ratio),
            "discourse_evidential_ratio": float(evidential_ratio),
            "discourse_marker_density": float(marker_density),
            "discourse_diversity": float(diversity),
        }

        logger.debug(
            "Discourse features extracted | density=%.4f diversity=%.4f",
            marker_density,
            diversity,
        )

        return features