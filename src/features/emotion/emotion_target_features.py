"""
File Name: emotion_target_features.py
Module: Feature Engineering - Emotion Target Features
Description:
    Extracts features describing the targets of emotional language in text.
    The module identifies entities or pronoun groups that co-occur with
    emotionally charged tokens and computes statistics indicating whether
    emotions are directed toward self, groups, or named entities.

    The implementation uses spaCy for entity recognition and dependency
    parsing when available. If spaCy is not available, the module falls back
    to a lightweight rule-based approach using pronoun patterns.

    These features are useful for identifying narratives where emotions are
    directed at actors, groups, institutions, or abstract targets.

Dependencies:
    dataclasses
    typing
    logging
    re
    collections
    spacy (optional)

Inputs:
    FeatureContext containing input text

Outputs:
    Dict[str, float] representing emotion-target interaction statistics
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optional spaCy NLP pipeline
# ---------------------------------------------------------------------

try:
    import spacy

    _NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:  # noqa: BLE001
    _NLP = None
    SPACY_AVAILABLE = False
    logger.warning(
        "spaCy not available. EmotionTargetFeatures will use fallback heuristics."
    )

# ---------------------------------------------------------------------
# Emotion lexicon
# ---------------------------------------------------------------------

EMOTION_WORDS = {
    "anger": {"angry", "rage", "furious", "hate"},
    "fear": {"fear", "terror", "panic", "scared"},
    "joy": {"happy", "joy", "delight", "smile"},
    "sadness": {"sad", "cry", "grief", "sorrow"},
}

# Flatten emotion vocabulary
EMOTION_VOCAB = set().union(*EMOTION_WORDS.values())

# Pronoun groups
FIRST_PERSON = {"i", "me", "my", "mine", "we", "our", "us"}
SECOND_PERSON = {"you", "your", "yours"}
THIRD_PERSON = {"he", "she", "they", "them", "their", "his", "her", "its"}


def _simple_tokenize(text: str) -> List[str]:
    """Fallback tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
@register_feature
class EmotionTargetFeatures(BaseFeature):
    """
    Detects which targets emotional language is directed toward.

    Output features include:
    - emotion_target_self_ratio
    - emotion_target_other_ratio
    - emotion_target_entity_ratio
    - emotion_target_group_ratio
    """

    name: str = "emotion_target_features"
    description: str = "Emotion direction and target detection"

    def _extract_spacy(self, text: str) -> Dict[str, float]:
        """Use spaCy to detect emotion targets."""
        doc = _NLP(text)

        self_targets = 0
        other_targets = 0
        entity_targets = 0
        group_targets = 0
        total_emotions = 0

        for token in doc:
            if token.text.lower() in EMOTION_VOCAB:
                total_emotions += 1

                # Check nearby tokens
                for neighbor in token.subtree:
                    t = neighbor.text.lower()

                    if t in FIRST_PERSON:
                        self_targets += 1
                    elif t in SECOND_PERSON or t in THIRD_PERSON:
                        other_targets += 1

                # Named entity targets
                if token.ent_type_:
                    entity_targets += 1

                # Plural noun heuristic
                if token.tag_ == "NNS":
                    group_targets += 1

        total_emotions = total_emotions or 1

        return {
            "emotion_target_self_ratio": float(self_targets / total_emotions),
            "emotion_target_other_ratio": float(other_targets / total_emotions),
            "emotion_target_entity_ratio": float(entity_targets / total_emotions),
            "emotion_target_group_ratio": float(group_targets / total_emotions),
        }

    def _extract_fallback(self, text: str) -> Dict[str, float]:
        """Fallback rule-based detection."""
        tokens = _simple_tokenize(text)

        emotion_positions = [
            i for i, t in enumerate(tokens) if t in EMOTION_VOCAB
        ]

        self_targets = 0
        other_targets = 0

        for pos in emotion_positions:
            window = tokens[max(0, pos - 3) : pos + 4]

            for w in window:
                if w in FIRST_PERSON:
                    self_targets += 1
                elif w in SECOND_PERSON or w in THIRD_PERSON:
                    other_targets += 1

        total_emotions = len(emotion_positions) or 1

        return {
            "emotion_target_self_ratio": float(self_targets / total_emotions),
            "emotion_target_other_ratio": float(other_targets / total_emotions),
            "emotion_target_entity_ratio": 0.0,
            "emotion_target_group_ratio": 0.0,
        }

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract emotion target features.
        """
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        if SPACY_AVAILABLE:
            features = self._extract_spacy(context.text)
        else:
            features = self._extract_fallback(context.text)

        logger.debug(
            "Emotion target features extracted | self=%.3f other=%.3f",
            features["emotion_target_self_ratio"],
            features["emotion_target_other_ratio"],
        )

        return features
