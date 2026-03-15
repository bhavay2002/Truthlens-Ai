"""
TruthLens AI
Framing Bias Detection Module

Detects how events are framed in news text using a combination of:

• Sentiment polarity
• Frame cue patterns
• Sentence embeddings
• RoBERTa frame classification
• Event-level frame detection

Outputs
-------
frame_type
frame_score
sentence_frames
event_frames
"""

from __future__ import annotations

import re
from typing import Dict, List
from dataclasses import dataclass
import numpy as np

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ROBERTA_FRAME_MODEL = "roberta-base"

FRAME_THRESHOLD_POS = 0.15
FRAME_THRESHOLD_NEG = -0.15


# ---------------------------------------------------------
# Frame Cue Lexicons
# ---------------------------------------------------------

POSITIVE_FRAMES = {
    "finally", "successfully", "achieved", "improved",
    "progress", "solution", "stabilized", "boosted"
}

NEGATIVE_FRAMES = {
    "forced", "failed", "crisis", "collapse",
    "scandal", "disaster", "controversy",
    "criticism", "pressure"
}


# ---------------------------------------------------------
# Frame Categories (Media Frames Corpus inspired)
# ---------------------------------------------------------

FRAME_LABELS = [
    "economic_frame",
    "responsibility_frame",
    "conflict_frame",
    "morality_frame",
    "human_interest_frame",
    "neutral_frame"
]


# ---------------------------------------------------------
# Data Structures
# ---------------------------------------------------------

@dataclass
class FramingResult:
    frame_type: str
    frame_score: float
    sentence_frames: List[Dict]
    event_frames: List[Dict]


# ---------------------------------------------------------
# Model Loader (Singleton style)
# ---------------------------------------------------------

class FramingModels:

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    sentiment_model = SentimentIntensityAnalyzer()

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_FRAME_MODEL)

    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        ROBERTA_FRAME_MODEL,
        num_labels=len(FRAME_LABELS)
    )


# ---------------------------------------------------------
# Text Processing
# ---------------------------------------------------------

def tokenize_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


# ---------------------------------------------------------
# Cue-based Frame Detection
# ---------------------------------------------------------

def compute_frame_cues(sentence: str) -> float:

    tokens = tokenize_words(sentence)

    pos = sum(1 for t in tokens if t in POSITIVE_FRAMES)
    neg = sum(1 for t in tokens if t in NEGATIVE_FRAMES)

    if not tokens:
        return 0.0

    return (pos - neg) / len(tokens)


# ---------------------------------------------------------
# Sentiment Framing
# ---------------------------------------------------------

def sentiment_score(sentence: str) -> float:

    scores = FramingModels.sentiment_model.polarity_scores(sentence)

    return scores["compound"]


# ---------------------------------------------------------
# RoBERTa Frame Classification
# ---------------------------------------------------------

def roberta_frame_classification(sentence: str) -> Dict:

    tokenizer = FramingModels.tokenizer
    model = FramingModels.roberta_model

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    label_id = torch.argmax(probs).item()

    frame = FRAME_LABELS[label_id]

    confidence = probs[0][label_id].item()

    return {
        "frame": frame,
        "confidence": round(confidence, 4)
    }


# ---------------------------------------------------------
# Event-Level Frame Detection
# ---------------------------------------------------------

def detect_event_frames(sentences: List[str]) -> List[Dict]:
    """
    Detect frames for individual events (sentences).
    """

    event_frames = []

    for sentence in sentences:

        frame_info = roberta_frame_classification(sentence)

        event_frames.append({
            "event": sentence,
            "frame": frame_info["frame"],
            "confidence": frame_info["confidence"]
        })

    return event_frames


# ---------------------------------------------------------
# Main Framing Detection
# ---------------------------------------------------------

def detect_framing_bias(text: str) -> Dict:

    sentences = tokenize_sentences(text)

    if not sentences:
        return {
            "frame_type": "neutral_frame",
            "framing_score": 0.0
        }

    sentiment_scores = []
    cue_scores = []

    sentence_frames = []

    for sentence in sentences:

        s_score = sentiment_score(sentence)
        c_score = compute_frame_cues(sentence)

        sentiment_scores.append(s_score)
        cue_scores.append(c_score)

        frame_info = roberta_frame_classification(sentence)

        sentence_frames.append({
            "sentence": sentence,
            "frame": frame_info["frame"],
            "confidence": frame_info["confidence"]
        })

    sentiment_avg = np.mean(sentiment_scores)
    cue_avg = np.mean(cue_scores)

    embeddings = FramingModels.embedding_model.encode(sentences)

    semantic_variance = np.var(embeddings)

    framing_score = round(
        abs(sentiment_avg) * 0.4 +
        abs(cue_avg) * 0.3 +
        semantic_variance * 0.3,
        4
    )

    # ---------------------------------------------------------
    # Frame polarity classification
    # ---------------------------------------------------------

    if sentiment_avg > FRAME_THRESHOLD_POS:
        frame_type = "positive_frame"

    elif sentiment_avg < FRAME_THRESHOLD_NEG:
        frame_type = "negative_frame"

    else:
        frame_type = "neutral_frame"

    # ---------------------------------------------------------
    # Event Frames
    # ---------------------------------------------------------

    event_frames = detect_event_frames(sentences)

    return {
        "frame_type": frame_type,
        "framing_score": framing_score,
        "sentence_frames": sentence_frames,
        "event_frames": event_frames
    }


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    text = """
    The government finally acted to stabilize the economy.
    However critics say the move was forced after months of failure.
    """

    result = detect_framing_bias(text)

    print(result)