"""
File Name: argument_structure_features.py
Module: Feature Engineering - Argument Structure Features
Description:
    Extracts argumentation structure signals from text. The module identifies
    linguistic cues related to claims, premises, evidence statements,
    counterarguments, and rhetorical questions. These features help quantify
    how arguments are constructed within discourse and can assist in
    identifying persuasive or argumentative writing patterns.

    The implementation uses curated lexical markers and lightweight
    heuristics to estimate argument structure indicators without requiring
    heavyweight NLP pipelines.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections

Inputs:
    FeatureContext containing input text and optional tokens

Outputs:
    Dict[str, float] representing argument structure indicators
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
# Argument Structure Lexicons
# ---------------------------------------------------------------------

CLAIM_MARKERS: Set[str] = {
    "therefore",
    "thus",
    "clearly",
    "obviously",
    "conclude",
    "shows",
}

PREMISE_MARKERS: Set[str] = {
    "because",
    "since",
    "given",
    "as",
    "assuming",
}

EVIDENCE_MARKERS: Set[str] = {
    "evidence",
    "study",
    "data",
    "report",
    "research",
    "analysis",
}

COUNTERARGUMENT_MARKERS: Set[str] = {
    "however",
    "although",
    "but",
    "nevertheless",
    "yet",
}

RHETORICAL_QUESTION_PATTERNS: Set[str] = {
    "why",
    "how",
    "what",
    "who",
}


@dataclass
@register_feature
class ArgumentStructureFeatures(BaseFeature):
    """
    Extracts argumentation structure indicators.

    Output Features
    ---------------
    argument_claim_ratio
    argument_premise_ratio
    argument_evidence_ratio
    argument_counterargument_ratio
    argument_rhetorical_question_ratio
    argument_structure_density
    argument_structure_diversity
    """

    name: str = "argument_structure_features"
    description: str = "Argumentation structure and reasoning indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """Extract argument structure features."""
        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")
        if not context.text.strip():
            return {}

        text = context.text
        text_lower = text.lower()

        tokens = context.tokens or _tokenize(text_lower)

        if not tokens:
            logger.warning("No tokens available for argument structure extraction")
            return {}

        counter = Counter(tokens)
        total_tokens = len(tokens)

        def ratio(lexicon: Set[str]) -> float:
            count = sum(counter.get(w, 0) for w in lexicon)
            return count / total_tokens

        claim_ratio = ratio(CLAIM_MARKERS)
        premise_ratio = ratio(PREMISE_MARKERS)
        evidence_ratio = ratio(EVIDENCE_MARKERS)
        counter_ratio = ratio(COUNTERARGUMENT_MARKERS)

        # rhetorical questions detected via question marks and interrogatives
        question_marks = text.count("?")
        interrogatives = sum(counter.get(w, 0) for w in RHETORICAL_QUESTION_PATTERNS)

        rhetorical_ratio = (question_marks + interrogatives) / max(total_tokens, 1)

        marker_counts = [
            sum(counter.get(w, 0) for w in CLAIM_MARKERS),
            sum(counter.get(w, 0) for w in PREMISE_MARKERS),
            sum(counter.get(w, 0) for w in EVIDENCE_MARKERS),
            sum(counter.get(w, 0) for w in COUNTERARGUMENT_MARKERS),
        ]

        structure_density = sum(marker_counts) / total_tokens
        structure_diversity = sum(1 for c in marker_counts if c > 0) / len(marker_counts)

        features: Dict[str, float] = {
            "argument_claim_ratio": float(claim_ratio),
            "argument_premise_ratio": float(premise_ratio),
            "argument_evidence_ratio": float(evidence_ratio),
            "argument_counterargument_ratio": float(counter_ratio),
            "argument_rhetorical_question_ratio": float(rhetorical_ratio),
            "argument_structure_density": float(structure_density),
            "argument_structure_diversity": float(structure_diversity),
        }

        logger.debug(
            "Argument structure features extracted | density=%.4f diversity=%.4f",
            structure_density,
            structure_diversity,
        )

        return features