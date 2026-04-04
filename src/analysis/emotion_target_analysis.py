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


# ---------------------------------------------------------
# Emotion Dataset Labels
# ---------------------------------------------------------

EMOTION_LABELS = {
    0: "neutral",
    1: "admiration",
    2: "approval",
    3: "gratitude",
    4: "annoyance",
    5: "amusement",
    6: "curiosity",
    7: "disapproval",
    8: "love",
    9: "optimism",
    10: "anger",
    11: "joy",
    12: "confusion",
    13: "sadness",
    14: "disappointment",
    15: "realization",
    16: "caring",
    17: "surprise",
    18: "excitement",
    19: "disgust",
}

 
# ---------------------------------------------------------
# Emotion Keywords
# ---------------------------------------------------------

EMOTION_TERMS = {

    "admiration": {
        "admire","admiration","respect","praise","commend","applaud",
        "appreciate","revere","esteem","honor","look_up_to","inspire"
    },

    "approval": {
        "approve","approval","support","endorse","accept","agree",
        "back","validate","favor","ratify","sanction"
    },

    "gratitude": {
        "thanks","thank","thankful","grateful","gratitude",
        "appreciation","indebted","obliged","much_obliged"
    },

    "annoyance": {
        "annoy","annoying","irritate","irritating","bother",
        "frustrate","frustrating","aggravate","aggravating",
        "disturb","disturbing"
    },

    "amusement": {
        "funny","amusing","hilarious","laugh","laughing",
        "entertaining","comic","comical","witty","playful"
    },

    "curiosity": {
        "curious","curiosity","wonder","wondering","intrigued",
        "interested","interest","inquisitive","question",
        "explore","exploration"
    },

    "disapproval": {
        "disapprove","disapproval","criticize","criticism",
        "condemn","condemnation","reject","denounce",
        "oppose","objection"
    },

    "love": {
        "love","adore","adoration","affection","fond",
        "fondness","cherish","devotion","passion","care_deeply"
    },

    "optimism": {
        "hope","hopeful","optimistic","optimism","positive",
        "encouraging","promising","confidence","confident",
        "bright_future"
    },

    "anger": {
        "anger","angry","furious","rage","outrage","fury",
        "irate","resent","resentment","enraged","hostile"
    },

    "joy": {
        "joy","joyful","happy","happiness","delighted",
        "delight","pleased","glad","cheerful","elated"
    },

    "confusion": {
        "confused","confusion","uncertain","uncertainty",
        "puzzled","perplexed","unclear","misunderstand",
        "ambiguous","bewildered"
    },

    "sadness": {
        "sad","sadness","depressed","depression","unhappy",
        "sorrow","sorrowful","gloomy","melancholy","grief"
    },

    "disappointment": {
        "disappointed","disappointment","letdown","dismayed",
        "discouraged","regret","regretful","frustrated_expectations"
    },

    "realization": {
        "realize","realization","realise","understand",
        "recognize","recognise","awareness","discover",
        "figure_out"
    },

    "caring": {
        "care","caring","concern","concerned","compassion",
        "empathetic","empathy","supportive","kindness"
    },

    "surprise": {
        "surprise","surprised","astonished","astonishment",
        "shocked","shock","unexpected","startled","amazed"
    },

    "excitement": {
        "excited","exciting","thrilled","thrill",
        "enthusiastic","enthusiasm","eager","anticipation"
    },

    "disgust": {
        "disgust","disgusting","gross","repulsive",
        "revolting","nauseating","sickening","abhorrent"
    },

}


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass(slots=True)
class EmotionTargetConfig:
    """
    Configuration for EmotionTargetAnalyzer.
    """

    spacy_model: str = "en_core_web_sm"
    use_dependency_targets: bool = True


# ---------------------------------------------------------
# Emotion Target Analyzer
# ---------------------------------------------------------

class EmotionTargetAnalyzer:
    """
    Identifies entities or subjects receiving emotional expressions in text.
    """

    def __init__(self, config: EmotionTargetConfig | None = None) -> None:

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

    # -----------------------------------------------------

    def analyze(self, text: str) -> Dict[str, float]:

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
        emotion_type_counter: DefaultDict[str, int] = defaultdict(int)

        # -------------------------------------------------

        for token in doc:

            token_lower = token.text.lower()

            detected_emotion = None

            for emotion, words in EMOTION_TERMS.items():
                if token_lower in words:
                    detected_emotion = emotion
                    break

            if detected_emotion:

                emotion_count += 1
                emotion_type_counter[detected_emotion] += 1

                target = self._resolve_target(token)

                if target:
                    entity_emotion_map[target] += 1

        # -------------------------------------------------

        total_entities = sum(entity_emotion_map.values())
        expression_ratio = emotion_count / max(len(doc), 1)

        features: Dict[str, float] = {}

        # Emotion diversity
        emotion_types = len(emotion_type_counter)

        # Dominant emotion
        dominant_emotion_freq = 0

        if emotion_type_counter:
            dominant_emotion_freq = max(emotion_type_counter.values())

        # -------------------------------------------------

        if total_entities == 0:

            features["emotion_target_diversity"] = 0.0
            features["emotion_target_focus"] = 0.0
            features["emotion_expression_ratio"] = float(expression_ratio)
            features["emotion_type_diversity"] = float(emotion_types)
            features["dominant_emotion_strength"] = float(dominant_emotion_freq)

            return features

        diversity = len(entity_emotion_map)
        dominant_target = max(entity_emotion_map.values())

        focus_score = dominant_target / max(total_entities, 1)

        # -------------------------------------------------

        features["emotion_target_diversity"] = float(diversity)
        features["emotion_target_focus"] = float(focus_score)
        features["emotion_expression_ratio"] = float(expression_ratio)
        features["emotion_type_diversity"] = float(emotion_types)
        features["dominant_emotion_strength"] = float(dominant_emotion_freq)

        logger.debug("Emotion target features computed")

        return features

    # -----------------------------------------------------

    def _resolve_target(self, token: Token) -> str | None:

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


# ---------------------------------------------------------
# Feature Vector Conversion
# ---------------------------------------------------------

def emotion_target_vector(features: Dict[str, float]) -> np.ndarray:

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