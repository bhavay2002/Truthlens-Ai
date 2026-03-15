"""
TruthLens AI
Emotional Manipulation Pattern Detection

Detects persuasive emotional manipulation strategies
in sensationalized or misleading content.

Techniques detected
-------------------
• Fear Amplification
• Outrage Triggers
• Urgency Triggers
• Conspiracy Cues
• Clickbait Patterns
• Rhetorical Manipulation

Detection Signals
-----------------
• Pattern lexicons
• Dependency parsing (spaCy)
• Transformer-based manipulation classifier
• Clickbait detection
"""

from __future__ import annotations

import re
from typing import Dict, List
from dataclasses import dataclass
from collections import Counter

import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------
# NLP Model
# ---------------------------------------------------------

NLP_MODEL = "en_core_web_sm"
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(NLP_MODEL)
    return _nlp


# ---------------------------------------------------------
# Transformer Model
# ---------------------------------------------------------

TRANSFORMER_MODEL = "roberta-base"

MANIPULATION_LABELS = [
    "fear_amplification",
    "outrage_trigger",
    "urgency_trigger",
    "conspiracy_cue",
    "neutral",
]


class ManipulationModels:
    _tokenizer = None
    _model = None

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
        return cls._tokenizer

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = AutoModelForSequenceClassification.from_pretrained(
                TRANSFORMER_MODEL, num_labels=len(MANIPULATION_LABELS)
            )
        return cls._model


# ---------------------------------------------------------
# Pattern Lexicons
# ---------------------------------------------------------

FEAR_PATTERNS = {
    "total collapse",
    "mass panic",
    "nation in danger",
    "catastrophic failure",
    "existential threat",
}

OUTRAGE_PATTERNS = {
    "this will shock you",
    "outrageous truth",
    "people are furious",
    "scandal exposed",
    "disgusting behavior",
}

URGENCY_PATTERNS = {
    "before it's too late",
    "act now",
    "urgent warning",
    "last chance",
    "don't miss this",
}

CONSPIRACY_PATTERNS = {
    "they don't want you to know",
    "hidden truth",
    "mainstream media won't tell you",
    "secret plan",
    "cover-up",
}

CLICKBAIT_PATTERNS = {
    "you won't believe",
    "what happens next",
    "this changes everything",
    "the truth revealed",
    "shocking secret",
}


# ---------------------------------------------------------
# Data Structure
# ---------------------------------------------------------


@dataclass
class ManipulationResult:
    manipulation_type: str
    manipulation_score: float
    detected_patterns: List[Dict]
    rhetorical_devices: List[Dict]
    sentence_analysis: List[Dict]


# ---------------------------------------------------------
# Sentence Tokenization
# ---------------------------------------------------------


def tokenize_sentences(text: str) -> List[str]:

    sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# Dependency-Based Persuasive Pattern Detection
# ---------------------------------------------------------


def detect_persuasive_structure(sentence: str) -> List[str]:

    doc = _get_nlp()(sentence)

    patterns = []

    for token in doc:

        # Imperative persuasion
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            if token.lemma_ in {"act", "watch", "see", "look"}:
                patterns.append("imperative_persuasion")

        # Fear appeals
        if token.lemma_ in {"threat", "destroy", "collapse"}:
            patterns.append("fear_structure")

    return patterns


# ---------------------------------------------------------
# Rhetorical Device Detection
# ---------------------------------------------------------


def detect_rhetorical_devices(sentence: str) -> List[str]:

    devices = []

    if "?" in sentence:
        devices.append("rhetorical_question")

    if re.search(r"\b(always|never|everyone|nobody)\b", sentence.lower()):
        devices.append("absolute_claim")

    return devices


# ---------------------------------------------------------
# Transformer Manipulation Classifier
# ---------------------------------------------------------


def transformer_manipulation_classifier(sentence: str) -> Dict:

    tokenizer = ManipulationModels.get_tokenizer()
    model = ManipulationModels.get_model()

    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    label_id = torch.argmax(probs).item()

    technique = MANIPULATION_LABELS[label_id]

    confidence = probs[0][label_id].item()

    return {"technique": technique, "confidence": round(confidence, 4)}


# ---------------------------------------------------------
# Pattern Matching
# ---------------------------------------------------------


def match_patterns(sentence: str, patterns: set) -> List[str]:

    sentence_lower = sentence.lower()

    matches = []

    for pattern in patterns:
        if pattern in sentence_lower:
            matches.append(pattern)

    return matches


# ---------------------------------------------------------
# Main Detection Pipeline
# ---------------------------------------------------------


def detect_emotion_manipulation(text: str) -> Dict:

    sentences = tokenize_sentences(text)

    if not sentences:
        return {
            "manipulation_type": "none",
            "manipulation_score": 0.0,
            "detected_patterns": [],
        }

    pattern_counts = Counter()
    detected_patterns = []
    rhetorical_devices = []
    sentence_analysis = []

    for sentence in sentences:

        matches = []

        matches += match_patterns(sentence, FEAR_PATTERNS)
        matches += match_patterns(sentence, OUTRAGE_PATTERNS)
        matches += match_patterns(sentence, URGENCY_PATTERNS)
        matches += match_patterns(sentence, CONSPIRACY_PATTERNS)
        matches += match_patterns(sentence, CLICKBAIT_PATTERNS)

        persuasive = detect_persuasive_structure(sentence)

        rhetorical = detect_rhetorical_devices(sentence)

        transformer_result = transformer_manipulation_classifier(sentence)

        for match in matches:
            detected_patterns.append({"sentence": sentence, "pattern": match})
            pattern_counts[match] += 1

        rhetorical_devices.extend(
            [{"sentence": sentence, "device": d} for d in rhetorical]
        )

        sentence_analysis.append(
            {
                "sentence": sentence,
                "patterns": matches,
                "persuasive_structures": persuasive,
                "transformer_prediction": transformer_result,
            }
        )

    total_patterns = sum(pattern_counts.values())

    manipulation_score = round(total_patterns / max(len(sentences), 1), 4)

    dominant_type = (
        pattern_counts.most_common(1)[0][0] if pattern_counts else "none"
    )

    return {
        "manipulation_type": dominant_type,
        "manipulation_score": manipulation_score,
        "detected_patterns": detected_patterns,
        "rhetorical_devices": rhetorical_devices,
        "sentence_analysis": sentence_analysis,
    }


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    text = """
    You won't believe this shocking secret plan!
    The media doesn't want you to know the truth.
    Act now before it's too late!
    """

    result = detect_emotion_manipulation(text)

    print(result)
