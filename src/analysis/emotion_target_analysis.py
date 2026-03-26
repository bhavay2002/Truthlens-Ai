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

import logging
from collections import defaultdict
from typing import Dict

import numpy as np
import spacy


logger = logging.getLogger(__name__)


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

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize NLP pipeline for emotion target analysis."""

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        logger.info("EmotionTargetAnalyzer initialized")

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze emotional language directed toward entities or subjects."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        try:
            doc = self.nlp(text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        entity_emotion_map = defaultdict(int)
        emotion_count = 0

        for token in doc:

            token_lower = token.text.lower()

            if token_lower in self.EMOTION_TERMS:
                emotion_count += 1

                head = token.head

                if head.ent_type_:
                    entity_emotion_map[head.ent_type_] += 1
                elif head.pos_ == "NOUN":
                    entity_emotion_map[head.lemma_.lower()] += 1

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

        return features


def emotion_target_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert emotion target features into numeric vector."""

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = np.array(list(features.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Emotion target vector conversion failed")
        raise RuntimeError("Failed to convert emotion target features") from exc