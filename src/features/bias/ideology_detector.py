"""
TruthLens AI
Ideology Detection Module

Detects ideological positioning and partisan framing
in news articles.

Capabilities
------------
• Political ideology classification
• Partisan framing detection
• Ideological narrative detection
• Sentence-level ideology attribution
• Transformer-based ideology classifier

Outputs
-------
ideology_label
ideology_score
partisan_frame
ideological_narratives
sentence_analysis
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

IDEOLOGY_MODEL = "roberta-base"

IDEOLOGY_LABELS = ["left", "right", "centrist", "neutral"]


# ---------------------------------------------------------
# Ideology Lexicons
# ---------------------------------------------------------

LEFT_LEXICON = {
    "climate justice",
    "wealth tax",
    "social justice",
    "systemic racism",
    "corporate greed",
    "income inequality",
    "progressive policy",
}

RIGHT_LEXICON = {
    "deep state",
    "illegal immigrants",
    "radical left",
    "globalist agenda",
    "border security",
    "traditional values",
    "patriot movement",
}

CENTRIST_LEXICON = {
    "bipartisan",
    "moderate policy",
    "balanced approach",
    "compromise",
    "middle ground",
}


# ---------------------------------------------------------
# Narrative Indicators
# ---------------------------------------------------------

LEFT_NARRATIVES = {
    "oppression",
    "inequality",
    "corporate power",
    "climate crisis",
}

RIGHT_NARRATIVES = {
    "national decline",
    "government overreach",
    "cultural decay",
    "security threat",
}


# ---------------------------------------------------------
# Data Structure
# ---------------------------------------------------------


@dataclass
class IdeologyResult:
    ideology_label: str
    ideology_score: float
    partisan_frame: str
    ideological_narratives: Dict[str, int]
    sentence_analysis: List[Dict]


# ---------------------------------------------------------
# Model Loader
# ---------------------------------------------------------


class IdeologyModels:
    _tokenizer = None
    _model = None

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(IDEOLOGY_MODEL)
        return cls._tokenizer

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                IDEOLOGY_MODEL, num_labels=len(IDEOLOGY_LABELS)
            )
        return cls._model


# ---------------------------------------------------------
# Text Processing
# ---------------------------------------------------------


def tokenize_sentences(text: str) -> List[str]:

    sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# Lexicon Ideology Detection
# ---------------------------------------------------------


def detect_lexicon_ideology(sentence: str) -> str:

    text = sentence.lower()

    if any(term in text for term in LEFT_LEXICON):
        return "left"

    if any(term in text for term in RIGHT_LEXICON):
        return "right"

    if any(term in text for term in CENTRIST_LEXICON):
        return "centrist"

    return "neutral"


# ---------------------------------------------------------
# Narrative Detection
# ---------------------------------------------------------


def detect_ideological_narratives(text: str) -> Dict:

    text_lower = text.lower()

    narratives = Counter()

    for term in LEFT_NARRATIVES:

        if term in text_lower:
            narratives["left_narrative"] += 1

    for term in RIGHT_NARRATIVES:

        if term in text_lower:
            narratives["right_narrative"] += 1

    return dict(narratives)


# ---------------------------------------------------------
# Transformer Ideology Classifier
# ---------------------------------------------------------


def transformer_ideology_classifier(sentence: str) -> Dict:

    tokenizer = IdeologyModels.get_tokenizer()
    model = IdeologyModels.get_model()

    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    label_id = torch.argmax(probs).item()

    ideology = IDEOLOGY_LABELS[label_id]

    confidence = probs[0][label_id].item()

    return {"ideology": ideology, "confidence": round(confidence, 4)}


# ---------------------------------------------------------
# Partisan Frame Detection
# ---------------------------------------------------------


def detect_partisan_frame(left_count: int, right_count: int) -> str:

    if left_count > right_count:
        return "left_frame"

    if right_count > left_count:
        return "right_frame"

    return "neutral_frame"


# ---------------------------------------------------------
# Main Detection Pipeline
# ---------------------------------------------------------


def detect_ideology(text: str) -> Dict:

    sentences = tokenize_sentences(text)

    if not sentences:

        return {
            "ideology_label": "neutral",
            "ideology_score": 0.0,
            "partisan_frame": "neutral_frame",
            "ideological_narratives": {},
            "sentence_analysis": [],
        }

    ideology_counter = Counter()

    sentence_analysis = []

    for sentence in sentences:

        lexicon_ideology = detect_lexicon_ideology(sentence)

        transformer_result = transformer_ideology_classifier(sentence)

        ideology_counter[lexicon_ideology] += 1
        ideology_counter[transformer_result["ideology"]] += 1

        sentence_analysis.append(
            {
                "sentence": sentence,
                "lexicon_ideology": lexicon_ideology,
                "transformer_prediction": transformer_result,
            }
        )

    narratives = detect_ideological_narratives(text)

    left_count = ideology_counter["left"]
    right_count = ideology_counter["right"]

    partisan_frame = detect_partisan_frame(left_count, right_count)

    dominant_ideology = (
        ideology_counter.most_common(1)[0][0]
        if ideology_counter
        else "neutral"
    )

    ideology_score = round(
        max(left_count, right_count) / max(len(sentences), 1), 4
    )

    return {
        "ideology_label": dominant_ideology,
        "ideology_score": ideology_score,
        "partisan_frame": partisan_frame,
        "ideological_narratives": narratives,
        "sentence_analysis": sentence_analysis,
    }


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    text = """
    The government must address climate justice and inequality.
    Critics warn that the radical left agenda threatens economic stability.
    """

    result = detect_ideology(text)

    print(result)
