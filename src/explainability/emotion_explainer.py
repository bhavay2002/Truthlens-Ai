"""
TruthLens AI
Emotion Explainability Engine

Provides interpretable explanations for emotional signals
and emotional manipulation in text.

Explanation Features
--------------------
• Emotion token detection
• Token-level emotion heatmap
• Sentence-level emotion heatmaps
• Integrated Gradients attribution
• Visualization-ready heatmap matrices
• UI-ready highlighting data
"""

from __future__ import annotations

import re
from typing import Dict, List
from dataclasses import dataclass
import torch

from src.features.emotion.emotion_lexicon import DEFAULT_NRC_LEXICON
from src.features.emotion.emotion_intensity import INTENSIFIER_ADVERBS


# ---------------------------------------------------------
# Data Structure
# ---------------------------------------------------------

@dataclass
class EmotionExplanation:
    emotion_tokens: List[Dict]
    emotion_heatmap: List[Dict]
    sentence_heatmap: List[Dict]
    gradient_attribution: List[Dict]
    heatmap_matrix: List[List[float]]
    ui_highlights: List[Dict]


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def tokenize_sentences(text: str) -> List[str]:

    sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# Emotion Token Detection
# ---------------------------------------------------------

def detect_emotion_tokens(tokens: List[str]) -> List[Dict]:

    emotion_tokens = []

    for idx, token in enumerate(tokens):

        matched_emotions = []

        for emotion, words in DEFAULT_NRC_LEXICON.items():

            if token in words:
                matched_emotions.append(emotion)

        if matched_emotions:

            emotion_tokens.append({
                "token": token,
                "emotions": matched_emotions,
                "position": idx
            })

    return emotion_tokens


# ---------------------------------------------------------
# Token Emotion Intensity
# ---------------------------------------------------------

def compute_token_intensity(tokens: List[str]) -> List[Dict]:

    heatmap = []

    for idx, token in enumerate(tokens):

        intensity = 0.0

        for emotion_words in DEFAULT_NRC_LEXICON.values():
            if token in emotion_words:
                intensity += 1.0

        if token in INTENSIFIER_ADVERBS:
            intensity += 0.5

        heatmap.append({
            "token": token,
            "intensity": round(intensity, 3),
            "position": idx
        })

    return heatmap


# ---------------------------------------------------------
# Sentence-Level Emotion Heatmap
# ---------------------------------------------------------

def compute_sentence_heatmap(text: str) -> List[Dict]:

    sentences = tokenize_sentences(text)

    results = []

    for sentence in sentences:

        tokens = tokenize_words(sentence)

        token_scores = compute_token_intensity(tokens)

        sentence_intensity = sum(t["intensity"] for t in token_scores)

        normalized = round(sentence_intensity / max(len(tokens), 1), 4)

        results.append({
            "sentence": sentence,
            "emotion_intensity": normalized
        })

    return results


# ---------------------------------------------------------
# Integrated Gradients Attribution
# ---------------------------------------------------------

def compute_integrated_gradients(model, tokenizer, text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    embedding_layer = model.get_input_embeddings()
    input_ids = inputs["input_ids"]
    input_embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

    model_kwargs = {
        "inputs_embeds": input_embeddings,
        "attention_mask": inputs.get("attention_mask")
    }
    if "token_type_ids" in inputs:
        model_kwargs["token_type_ids"] = inputs["token_type_ids"]

    outputs = model(**model_kwargs)

    target = outputs.logits.max()

    target.backward()

    gradients = input_embeddings.grad

    scores = gradients.abs().sum(dim=-1).detach().cpu().numpy()[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    results = []

    for token, score in zip(tokens, scores):

        results.append({
            "token": token,
            "importance": float(score)
        })

    return results


# ---------------------------------------------------------
# Heatmap Matrix (Visualization)
# ---------------------------------------------------------

def generate_heatmap_matrix(tokens: List[str], heatmap: List[Dict]):

    matrix = []

    for token, item in zip(tokens, heatmap):

        matrix.append([item["intensity"]])

    return matrix


# ---------------------------------------------------------
# UI Highlight Layer
# ---------------------------------------------------------

def generate_ui_highlights(heatmap: List[Dict]) -> List[Dict]:

    highlights = []

    for token_data in heatmap:

        if token_data["intensity"] > 0:

            highlights.append({
                "token": token_data["token"],
                "strength": token_data["intensity"]
            })

    return highlights


# ---------------------------------------------------------
# Main Explainability Pipeline
# ---------------------------------------------------------

def explain_emotion(text: str, model=None, tokenizer=None) -> Dict:

    tokens = tokenize_words(text)

    # -----------------------------------------------------
    # Token emotion detection
    # -----------------------------------------------------

    emotion_tokens = detect_emotion_tokens(tokens)

    # -----------------------------------------------------
    # Token heatmap
    # -----------------------------------------------------

    emotion_heatmap = compute_token_intensity(tokens)

    # -----------------------------------------------------
    # Sentence heatmap
    # -----------------------------------------------------

    sentence_heatmap = compute_sentence_heatmap(text)

    # -----------------------------------------------------
    # Integrated Gradients
    # -----------------------------------------------------

    gradient_attr = []

    if model and tokenizer:
        gradient_attr = compute_integrated_gradients(model, tokenizer, text)

    # -----------------------------------------------------
    # Visualization matrix
    # -----------------------------------------------------

    heatmap_matrix = generate_heatmap_matrix(tokens, emotion_heatmap)

    # -----------------------------------------------------
    # UI highlighting
    # -----------------------------------------------------

    ui_highlights = generate_ui_highlights(emotion_heatmap)

    return {
        "emotion_tokens": emotion_tokens,
        "emotion_heatmap": emotion_heatmap,
        "sentence_heatmap": sentence_heatmap,
        "gradient_attribution": gradient_attr,
        "heatmap_matrix": heatmap_matrix,
        "ui_highlights": ui_highlights
    }


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    text = """
    The shocking crisis caused fear and panic among citizens.
    This absolutely terrible decision will destroy everything.
    """

    result = explain_emotion(text)

    print(result)