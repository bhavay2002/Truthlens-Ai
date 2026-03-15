"""
TruthLens AI
Propaganda Technique Detection Module

Detects propaganda techniques commonly used in misinformation
and persuasive media narratives.

Techniques detected
-------------------
• Loaded Language
• Appeal to Fear
• Name Calling
• Bandwagon

Detection Methods
-----------------
• Lexicon-based detection
• Rule-based rhetorical patterns
• Transformer contextual classifier

Outputs
-------
propaganda_score
dominant_technique
detected_techniques
sentence_analysis
token_highlights
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

PROPAGANDA_MODEL = "roberta-base"

TECHNIQUE_LABELS = [
    "loaded_language",
    "appeal_to_fear",
    "name_calling",
    "bandwagon",
    "neutral",
]


# ---------------------------------------------------------
# Lexicons
# ---------------------------------------------------------

LOADED_LANGUAGE = {
    "outrageous",
    "shocking",
    "disgraceful",
    "corrupt",
    "horrible",
    "absurd",
    "catastrophic",
    "evil",
}

FEAR_WORDS = {
    "threat",
    "danger",
    "crisis",
    "collapse",
    "chaos",
    "panic",
    "disaster",
}

NAME_CALLING = {
    "traitor",
    "radical",
    "extremist",
    "criminal",
    "enemy",
    "corrupt",
}

BANDWAGON_PATTERNS = [
    r"everyone knows",
    r"the people agree",
    r"millions support",
    r"the whole country",
]

PROPAGANDA_LEXICON = LOADED_LANGUAGE | FEAR_WORDS | NAME_CALLING


# ---------------------------------------------------------
# Data Structure
# ---------------------------------------------------------


@dataclass
class PropagandaResult:
    propaganda_score: float
    dominant_technique: str
    detected_techniques: Dict[str, int]
    sentence_analysis: List[Dict]
    token_highlights: List[Dict]


# ---------------------------------------------------------
# Model Loader
# ---------------------------------------------------------


class PropagandaModels:
    _tokenizer = None
    _model = None

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(PROPAGANDA_MODEL)
        return cls._tokenizer

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                PROPAGANDA_MODEL, num_labels=len(TECHNIQUE_LABELS)
            )
        return cls._model


# ---------------------------------------------------------
# Text Processing
# ---------------------------------------------------------


def tokenize_sentences(text: str) -> List[str]:

    sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> List[str]:

    return re.findall(r"\b[a-z]+\b", text.lower())


# ---------------------------------------------------------
# Lexicon Detection
# ---------------------------------------------------------


def detect_lexicon_techniques(sentence: str) -> List[str]:

    tokens = tokenize_words(sentence)

    techniques = []

    if any(t in LOADED_LANGUAGE for t in tokens):
        techniques.append("loaded_language")

    if any(t in FEAR_WORDS for t in tokens):
        techniques.append("appeal_to_fear")

    if any(t in NAME_CALLING for t in tokens):
        techniques.append("name_calling")

    return techniques


# ---------------------------------------------------------
# Bandwagon Detection
# ---------------------------------------------------------


def detect_bandwagon(sentence: str) -> bool:

    sentence_lower = sentence.lower()

    return any(re.search(p, sentence_lower) for p in BANDWAGON_PATTERNS)


# ---------------------------------------------------------
# Transformer Classification
# ---------------------------------------------------------


def transformer_propaganda_classifier(sentence: str) -> Dict:

    tokenizer = PropagandaModels.get_tokenizer()
    model = PropagandaModels.get_model()

    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    label_id = torch.argmax(probs).item()

    technique = TECHNIQUE_LABELS[label_id]

    confidence = probs[0][label_id].item()

    return {"technique": technique, "confidence": round(confidence, 4)}


# ---------------------------------------------------------
# Token Highlighting
# ---------------------------------------------------------


def highlight_propaganda_tokens(sentence: str) -> List[Dict]:

    tokens = tokenize_words(sentence)

    highlights = []

    for idx, token in enumerate(tokens):

        if token in PROPAGANDA_LEXICON:

            highlights.append(
                {"token": token, "type": "propaganda", "position": idx}
            )

    return highlights


# ---------------------------------------------------------
# Main Detection Pipeline
# ---------------------------------------------------------


def detect_propaganda(text: str) -> Dict:

    sentences = tokenize_sentences(text)

    if not sentences:
        return {
            "propaganda_score": 0.0,
            "dominant_technique": "neutral",
            "detected_techniques": {},
            "sentence_analysis": [],
            "token_highlights": [],
        }

    sentence_analysis = []
    token_highlights = []

    technique_counter = Counter()

    for sentence in sentences:

        techniques = detect_lexicon_techniques(sentence)

        if detect_bandwagon(sentence):
            techniques.append("bandwagon")

        transformer_result = transformer_propaganda_classifier(sentence)

        techniques.append(transformer_result["technique"])

        for tech in techniques:
            technique_counter[tech] += 1

        token_highlights.extend(highlight_propaganda_tokens(sentence))

        sentence_analysis.append(
            {
                "sentence": sentence,
                "techniques": techniques,
                "transformer_prediction": transformer_result,
            }
        )

    total = sum(technique_counter.values())

    propaganda_score = round(total / max(len(sentences), 1), 4)

    dominant = (
        technique_counter.most_common(1)[0][0]
        if technique_counter
        else "neutral"
    )

    return {
        "propaganda_score": propaganda_score,
        "dominant_technique": dominant,
        "detected_techniques": dict(technique_counter),
        "sentence_analysis": sentence_analysis,
        "token_highlights": token_highlights,
    }


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    text = """
    The corrupt elites are destroying the nation.
    Everyone knows this dangerous policy will cause disaster.
    Millions support the brave leader who will save the country.
    """

    result = detect_propaganda(text)

    print(result)
