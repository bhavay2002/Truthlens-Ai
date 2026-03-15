"""
File: src/explainability/emotion_explainer.py

Purpose
-------
Explain emotional manipulation signals in text.

Features
--------
• Emotion token detection
• Token-level emotion heatmap
• Sentence-level emotion heatmap
• Integrated Gradients attribution
• Visualization-ready heatmap matrices
• UI-ready highlighting data
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from src.features.emotion.emotion_intensity import INTENSIFIER_ADVERBS
from src.features.emotion.emotion_lexicon import DEFAULT_NRC_LEXICON


@dataclass
class EmotionExplanation:
    """Structured container for emotion explanations."""

    emotion_tokens: List[Dict[str, Any]]
    emotion_heatmap: List[Dict[str, Any]]
    sentence_heatmap: List[Dict[str, Any]]
    gradient_attribution: List[Dict[str, Any]]
    heatmap_matrix: List[List[float]]
    ui_highlights: List[Dict[str, Any]]


def tokenize_words(text: str) -> List[str]:
    """Tokenize words from text."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def detect_emotion_tokens(tokens: List[str]) -> List[Dict[str, Any]]:
    """Detect tokens associated with emotions using NRC lexicon."""
    emotion_tokens: list[dict[str, Any]] = []

    for idx, token in enumerate(tokens):
        matched_emotions = [
            emotion
            for emotion, words in DEFAULT_NRC_LEXICON.items()
            if token in words
        ]

        if matched_emotions:
            emotion_tokens.append(
                {
                    "token": token,
                    "emotions": matched_emotions,
                    "position": idx,
                }
            )

    return emotion_tokens


def compute_token_intensity(tokens: List[str]) -> List[Dict[str, Any]]:
    """Compute token-level emotional intensity."""
    heatmap: list[dict[str, Any]] = []

    for idx, token in enumerate(tokens):
        intensity = 0.0

        for emotion_words in DEFAULT_NRC_LEXICON.values():
            if token in emotion_words:
                intensity += 1.0

        if token in INTENSIFIER_ADVERBS:
            intensity += 0.5

        heatmap.append(
            {
                "token": token,
                "intensity": round(intensity, 3),
                "position": idx,
            }
        )

    return heatmap


def compute_sentence_heatmap(text: str) -> List[Dict[str, Any]]:
    """Compute emotional intensity for each sentence."""
    sentences = tokenize_sentences(text)
    results: list[dict[str, Any]] = []

    for sentence in sentences:
        tokens = tokenize_words(sentence)
        token_scores = compute_token_intensity(tokens)

        sentence_intensity = sum(item["intensity"] for item in token_scores)
        normalized = round(sentence_intensity / max(len(tokens), 1), 4)

        results.append(
            {
                "sentence": sentence,
                "emotion_intensity": normalized,
            }
        )

    return results


def _resolve_model_device(model) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def compute_integrated_gradients(
    model,
    tokenizer,
    text: str,
) -> List[Dict[str, Any]]:
    """Compute gradient-based attribution for tokens."""
    device = _resolve_model_device(model)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)

    embedding_layer = model.get_input_embeddings()
    input_ids = inputs["input_ids"]

    input_embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

    model_kwargs = {
        "inputs_embeds": input_embeddings,
        "attention_mask": inputs.get("attention_mask"),
    }

    if "token_type_ids" in inputs:
        model_kwargs["token_type_ids"] = inputs["token_type_ids"]

    outputs = model(**model_kwargs)
    target = outputs.logits.max()
    target.backward()

    gradients = input_embeddings.grad
    if gradients is None:
        raise RuntimeError(
            "Failed to compute gradients for integrated gradients."
        )

    scores = gradients.abs().sum(dim=-1).detach().cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu())

    return [
        {
            "token": token,
            "importance": float(score),
        }
        for token, score in zip(tokens, scores)
    ]


def generate_heatmap_matrix(
    tokens: List[str],
    heatmap: List[Dict[str, Any]],
) -> List[List[float]]:
    """Convert token heatmap to matrix format for visualization."""
    _ = tokens
    return [[float(item["intensity"])] for item in heatmap]


def generate_ui_highlights(
    heatmap: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate UI-ready token highlighting data."""
    return [
        {
            "token": token_data["token"],
            "strength": token_data["intensity"],
        }
        for token_data in heatmap
        if token_data["intensity"] > 0
    ]


def explain_emotion(
    text: str,
    model=None,
    tokenizer=None,
) -> Dict[str, Any]:
    """Run full emotion explanation pipeline."""
    tokens = tokenize_words(text)

    emotion_tokens = detect_emotion_tokens(tokens)
    emotion_heatmap = compute_token_intensity(tokens)
    sentence_heatmap = compute_sentence_heatmap(text)

    gradient_attr: list[dict[str, Any]] = []
    if model is not None and tokenizer is not None:
        gradient_attr = compute_integrated_gradients(
            model,
            tokenizer,
            text,
        )

    heatmap_matrix = generate_heatmap_matrix(tokens, emotion_heatmap)
    ui_highlights = generate_ui_highlights(emotion_heatmap)

    return {
        "emotion_tokens": emotion_tokens,
        "emotion_heatmap": emotion_heatmap,
        "sentence_heatmap": sentence_heatmap,
        "gradient_attribution": gradient_attr,
        "heatmap_matrix": heatmap_matrix,
        "ui_highlights": ui_highlights,
    }
