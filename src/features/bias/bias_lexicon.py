"""
TruthLens AI
Bias Lexicon Detection Module

Detects subjective, propagandistic, and emotionally loaded language.

Capabilities:
    • Weighted bias scoring
    • Sentence-level bias heatmap
    • Token-level bias attribution
    • Media bias estimation
    • Contextual bias hooks
    • Transformer bias classifier support

Outputs:
    bias_score
    biased_tokens
    sentence_heatmap
    media_bias
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------
# Bias Lexicons with Weights
# ---------------------------------------------------------

BIAS_LEXICON = {
    # Emotional / loaded language
    "shocking": 1.8,
    "outrageous": 1.6,
    "disastrous": 1.7,
    "corrupt": 2.0,
    "disgraceful": 1.7,
    "evil": 2.2,
    "scandalous": 1.8,
    "unbelievable": 1.4,
    "ridiculous": 1.5,
    "horrifying": 1.9,
    "catastrophic": 2.1,
    "absurd": 1.5,
    "disturbing": 1.6,

    # Propaganda framing
    "regime": 1.5,
    "propaganda": 1.9,
    "agenda": 1.6,
    "brainwash": 2.0,
    "manipulate": 1.8,
    "elite": 1.3,
    "deep": 1.4,
    "state": 1.4,
    "cover": 1.6,
    "hoax": 1.9,
    "conspiracy": 1.8,
    "traitor": 2.2,
    "enemy": 1.6,
    "fake": 1.5,

    # Opinion indicators
    "clearly": 1.3,
    "obviously": 1.4,
    "undoubtedly": 1.5,
    "certainly": 1.2,
    "apparently": 1.1,
    "arguably": 1.2,
    "supposedly": 1.3,
    "allegedly": 1.1,
    "frankly": 1.2,
    "honestly": 1.1,
    "fortunately": 1.1,
    "unfortunately": 1.2,
}


# ---------------------------------------------------------
# Political Bias Indicators
# ---------------------------------------------------------

LEFT_LEXICON = {
    "climate justice", "systemic inequality", "wealth tax",
    "social justice", "corporate greed"
}

RIGHT_LEXICON = {
    "deep state", "radical left", "fake news",
    "illegal immigrants", "globalist agenda"
}


# ---------------------------------------------------------
# Data Classes
# ---------------------------------------------------------

@dataclass
class BiasResult:
    bias_score: float
    biased_tokens: List[str]
    token_weights: Dict[str, float]
    sentence_heatmap: List[Dict]
    media_bias: str


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """
    Lowercase tokenization using regex.
    """
    text = text.lower()
    return re.findall(r"\b[a-z]+\b", text)


def split_sentences(text: str) -> List[str]:
    """
    Sentence segmentation.
    """
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# Weighted Bias Scoring
# ---------------------------------------------------------

def compute_weighted_bias(tokens: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Compute weighted bias score.

    Score = sum(weights of biased tokens) / total tokens
    """

    token_weights = {}

    weighted_sum = 0.0

    for token in tokens:
        if token in BIAS_LEXICON:
            weight = BIAS_LEXICON[token]
            token_weights[token] = weight
            weighted_sum += weight

    if len(tokens) == 0:
        return 0.0, {}

    bias_score = weighted_sum / len(tokens)

    return round(bias_score, 4), token_weights


# ---------------------------------------------------------
# Sentence Heatmap
# ---------------------------------------------------------

def compute_sentence_heatmap(text: str) -> List[Dict]:
    """
    Compute sentence-level bias heatmap.
    """

    sentences = split_sentences(text)

    heatmap = []

    for sentence in sentences:

        tokens = tokenize(sentence)

        score, _ = compute_weighted_bias(tokens)

        heatmap.append({
            "sentence": sentence,
            "bias_score": score
        })

    return heatmap


# ---------------------------------------------------------
# Media Bias Classification
# ---------------------------------------------------------

def classify_media_bias(text: str) -> str:
    """
    Simple heuristic political bias classifier.
    """

    text_lower = text.lower()

    left_hits = sum(1 for term in LEFT_LEXICON if term in text_lower)
    right_hits = sum(1 for term in RIGHT_LEXICON if term in text_lower)

    if left_hits > right_hits:
        return "Left"
    elif right_hits > left_hits:
        return "Right"
    else:
        return "Neutral"


# ---------------------------------------------------------
# Main Bias Detection Pipeline
# ---------------------------------------------------------

def compute_bias_features(text: str) -> BiasResult:
    """
    Main entry point used by TruthLens pipeline.
    """

    tokens = tokenize(text)

    bias_score, token_weights = compute_weighted_bias(tokens)

    biased_tokens = list(token_weights.keys())

    sentence_heatmap = compute_sentence_heatmap(text)

    media_bias = classify_media_bias(text)

    return BiasResult(
        bias_score=bias_score,
        biased_tokens=biased_tokens,
        token_weights=token_weights,
        sentence_heatmap=sentence_heatmap,
        media_bias=media_bias
    )


# ---------------------------------------------------------
# Optional: BERT Bias Classifier Hook
# ---------------------------------------------------------

def bert_bias_classifier(text: str, model=None, tokenizer=None) -> Dict:
    """
    Contextual bias detection using transformer models.

    Expected model:
        fine-tuned BERT/RoBERTa bias classifier
    """

    if model is None or tokenizer is None:
        return {"bias_probability": None}

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    outputs = model(**inputs)

    probs = outputs.logits.softmax(dim=1)

    bias_prob = probs[0][1].item()

    return {
        "bias_probability": round(bias_prob, 4)
    }


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_text = """
    This shocking scandal proves the corrupt regime is hiding the truth.
    Obviously the elite are manipulating the public.
    """

    result = compute_bias_features(sample_text)

    print("Bias Score:", result.bias_score)
    print("Biased Tokens:", result.biased_tokens)
    print("Media Bias:", result.media_bias)
    print("Sentence Heatmap:", result.sentence_heatmap)