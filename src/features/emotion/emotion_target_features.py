from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.emotion.emotion_schema import EMOTION_TERMS

logger = logging.getLogger(__name__)

WORD_TO_EMOTION: Dict[str, str] = {}
for emotion, words in EMOTION_TERMS.items():
    for word in words:
        WORD_TO_EMOTION[word.replace("_", " ").lower()] = emotion

EMOTION_VOCAB = set(WORD_TO_EMOTION.keys())

FIRST_PERSON = {"i", "me", "my", "mine", "we", "our", "us"}
SECOND_PERSON = {"you", "your", "yours"}
THIRD_PERSON = {"he", "she", "they", "them", "their", "his", "her", "its"}


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


@dataclass
@register_feature
class EmotionTargetFeatures(BaseFeature):
    name: str = "emotion_target_features"
    description: str = "Emotion direction and target detection"

    _nlp: Any = field(default=None, init=False, repr=False)
    _spacy_available: bool = field(default=False, init=False, repr=False)

    def initialize(self) -> None:
        if self._nlp is not None or self._spacy_available:
            return
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            self._spacy_available = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("spaCy unavailable; using fallback heuristics: %s", exc)
            self._nlp = None
            self._spacy_available = False

    def _extract_spacy(self, text: str) -> Dict[str, float]:
        doc = self._nlp(text)
        doc_len = max(len(doc), 1)

        self_targets = 0
        other_targets = 0
        entity_targets = 0
        group_targets = 0
        emotion_count = 0
        active_targets = set()

        for token in doc:
            tok = token.text.lower()
            if tok not in EMOTION_VOCAB:
                continue

            emotion_count += 1

            neighborhood = {token}
            neighborhood.update(token.children)
            if token.head is not None:
                neighborhood.add(token.head)

            for neighbor in neighborhood:
                t = neighbor.text.lower()
                if t in FIRST_PERSON:
                    self_targets += 1
                    active_targets.add("self")
                elif t in SECOND_PERSON or t in THIRD_PERSON:
                    other_targets += 1
                    active_targets.add("other")

                if getattr(neighbor, "ent_type_", ""):
                    entity_targets += 1
                    active_targets.add("entity")

                if getattr(neighbor, "tag_", "") == "NNS":
                    group_targets += 1
                    active_targets.add("group")

        denom = max(emotion_count, 1)
        return {
            "emotion_target_self_ratio": self_targets / denom,
            "emotion_target_other_ratio": other_targets / denom,
            "emotion_target_entity_ratio": entity_targets / denom,
            "emotion_target_group_ratio": group_targets / denom,
            "emotion_target_density": emotion_count / doc_len,
            "emotion_target_diversity": len(active_targets) / 4.0,
        }

    def _extract_fallback(self, text: str) -> Dict[str, float]:
        tokens = _simple_tokenize(text)
        if not tokens:
            return {
                "emotion_target_self_ratio": 0.0,
                "emotion_target_other_ratio": 0.0,
                "emotion_target_entity_ratio": 0.0,
                "emotion_target_group_ratio": 0.0,
                "emotion_target_density": 0.0,
                "emotion_target_diversity": 0.0,
            }

        emotion_positions = [i for i, token in enumerate(tokens) if token in EMOTION_VOCAB]
        self_targets = 0
        other_targets = 0

        for pos in emotion_positions:
            window = tokens[max(0, pos - 3) : pos + 4]
            for w in window:
                if w in FIRST_PERSON:
                    self_targets += 1
                elif w in SECOND_PERSON or w in THIRD_PERSON:
                    other_targets += 1

        total_emotions = max(len(emotion_positions), 1)
        density = len(emotion_positions) / max(len(tokens), 1)
        diversity = 0.0
        if self_targets > 0:
            diversity += 1
        if other_targets > 0:
            diversity += 1

        return {
            "emotion_target_self_ratio": self_targets / total_emotions,
            "emotion_target_other_ratio": other_targets / total_emotions,
            "emotion_target_entity_ratio": 0.0,
            "emotion_target_group_ratio": 0.0,
            "emotion_target_density": density,
            "emotion_target_diversity": diversity / 4.0,
        }

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")
        if not context.text.strip():
            return {}

        self.initialize()
        features = self._extract_spacy(context.text) if self._spacy_available else self._extract_fallback(context.text)

        logger.debug(
            "Emotion target features extracted | self=%.3f other=%.3f",
            features["emotion_target_self_ratio"],
            features["emotion_target_other_ratio"],
        )
        return features