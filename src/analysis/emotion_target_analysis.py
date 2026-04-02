"""
File Name: emotion_target_analysis.py
Module: Emotion Analysis - Target Analysis
Description:
    Analyzes the targets toward which emotions are directed within text for the
    TruthLens AI system. The module identifies entities, actors, or groups that
    receive emotional language and estimates how emotional expressions are
    distributed across these targets. This helps identify emotionally charged
    framing directed at specific subjects within discourse.

Dependencies:
    logging
    typing
    collections
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Emotion target feature dictionary and numerical vector
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, DefaultDict, List

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmotionTargetConfig:
    """
    Configuration for EmotionTargetAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"
    use_dependency_targets: bool = True


class EmotionTargetAnalyzer:
    """
    Identifies entities or subjects receiving emotional expressions in text.
    """

    EMOTION_TERMS = {
        "anger",
        "angry",
        "furious",
        "hate",
        "fear",
        "afraid",
        "terrified",
        "joy",
        "happy",
        "delighted",
        "sad",
        "sadness",
        "disgust",
        "disgusting",
        "surprised",
        "shock",
        "trust",
    }

    def __init__(self, config: EmotionTargetConfig | None = None) -> None:
        """
        Initialize NLP pipeline for emotion target analysis.

        Args:
            config: Optional configuration object.
        """

        self.config = config or EmotionTargetConfig()

        try:
            self.nlp: Language = spacy.load(self.config.spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError(
                f"Failed to load spaCy model: {self.config.spacy_model}"
            ) from exc

        logger.info(
            "EmotionTargetAnalyzer initialized with model=%s",
            self.config.spacy_model,
        )

    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze emotional language directed toward entities or subjects.

        Args:
            text: Input text.

        Returns:
            Dictionary containing emotion target features.
        """

        if not isinstance(text, str):
            raise ValueError("Input text must be a string")

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("Input text must be a non-empty string")

        try:
            doc: Doc = self.nlp(cleaned_text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        entity_emotion_map: DefaultDict[str, int] = defaultdict(int)
        emotion_count: int = 0

        for token in doc:

            token_lower = token.text.lower()

            if token_lower in self.EMOTION_TERMS:
                emotion_count += 1
                target = self._resolve_target(token)

                if target:
                    entity_emotion_map[target] += 1

        total_entities = sum(entity_emotion_map.values())
        expression_ratio = emotion_count / max(len(doc), 1)

        features: Dict[str, float] = {}

        if total_entities == 0:
            features["emotion_target_diversity"] = 0.0
            features["emotion_target_focus"] = 0.0
            features["emotion_expression_ratio"] = float(expression_ratio)
            return features

        diversity = len(entity_emotion_map)
        dominant_target = max(entity_emotion_map.values())
        focus_score = dominant_target / max(total_entities, 1)

        features["emotion_target_diversity"] = float(diversity)
        features["emotion_target_focus"] = float(focus_score)
        features["emotion_expression_ratio"] = float(expression_ratio)

        logger.debug("Emotion target features computed")

        return features

    def _resolve_target(self, token: Token) -> str | None:
        """
        Resolve the likely emotional target of a token.

        Args:
            token: spaCy token containing emotional expression.

        Returns:
            Target entity or noun lemma.
        """

        head = token.head

        if head.ent_type_:
            return head.ent_type_

        if head.pos_ == "NOUN":
            return head.lemma_.lower()

        if self.config.use_dependency_targets:
            for child in head.children:
                if child.ent_type_:
                    return child.ent_type_
                if child.pos_ == "NOUN":
                    return child.lemma_.lower()

        return None


def emotion_target_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert emotion target features into numeric vector.

    Args:
        features: Emotion target feature dictionary.

    Returns:
        NumPy feature vector.
    """

    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary")

    if not features:
        raise ValueError("features must be a non-empty dictionary")

    numeric_values: List[float] = []

    for key, value in features.items():
        if isinstance(value, (int, float, np.number)):
            numeric_values.append(float(value))
        else:
            logger.warning("Non-numeric emotion target feature skipped: %s", key)

    if not numeric_values:
        raise ValueError("No numeric values found in features")

    try:
        vector = np.array(numeric_values, dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Emotion target vector conversion failed")
        raise RuntimeError(
            "Failed to convert emotion target features"
        ) from exc