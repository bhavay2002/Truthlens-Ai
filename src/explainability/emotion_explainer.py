"""
File Name: emotion_explainer.py
Module: Explainability - Emotion Analysis
Description:
    Provides utilities for explaining emotional manipulation signals in text.
    Includes lexical emotion detection, token-level emotion heatmaps,
    sentence-level intensity scoring, gradient-based attribution using
    transformer models, and visualization-ready matrices for dashboards
    and UI highlighting.

Dependencies:
    logging
    re
    dataclasses
    typing
    torch

Inputs:
    text (str)
    optional transformer model and tokenizer

Outputs:
    structured explanation dictionary containing emotion attribution
    data and visualization-ready artifacts
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    EMOTION_TERMS,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# Reverse lookup for fast emotion detection
# -------------------------------------------------------------

WORD_TO_EMOTION: Dict[str, str] = {}

for emotion, words in EMOTION_TERMS.items():
    for word in words:
        WORD_TO_EMOTION[word] = emotion


# -------------------------------------------------------------
# Intensifier vocabulary
# -------------------------------------------------------------

INTENSIFIER_ADVERBS = {

    # strong intensity amplifiers
    "very",
    "extremely",
    "highly",
    "incredibly",
    "remarkably",
    "exceptionally",
    "unbelievably",
    "extraordinarily",
    "immensely",
    "intensely",
    "profoundly",
    "dramatically",

    # common sentiment amplifiers
    "really",
    "so",
    "too",
    "quite",
    "pretty",
    "fairly",
    "rather",
    "especially",
    "particularly",
    "notably",
    "significantly",

    # conversational amplifiers
    "totally",
    "absolutely",
    "completely",
    "entirely",
    "fully",
    "perfectly",
    "truly",
    "genuinely",
    "seriously",

    # emotional amplification
    "deeply",
    "strongly",
    "wildly",
    "terribly",
    "awfully",
    "terribly",
    "desperately",
    "badly",

    # narrative emphasis (news rhetoric)
    "massively",
    "hugely",
    "greatly",
    "vastly",
    "considerably",
    "substantially",
    "overwhelmingly",

    # dramatic rhetoric indicators
    "utterly",
    "absolutely",
    "decidedly",
    "undeniably",
    "indisputably",
    "unquestionably",

}

# -------------------------------------------------------------
# Data container
# -------------------------------------------------------------

@dataclass
class EmotionExplanation:

    emotion_tokens: List[Dict[str, Any]]
    emotion_heatmap: List[Dict[str, Any]]
    sentence_heatmap: List[Dict[str, Any]]
    gradient_attribution: List[Dict[str, Any]]
    heatmap_matrix: List[List[float]]
    ui_highlights: List[Dict[str, Any]]


# -------------------------------------------------------------
# Tokenization
# -------------------------------------------------------------

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def tokenize_sentences(text: str) -> List[str]:

    sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if s.strip()]


# -------------------------------------------------------------
# Emotion token detection
# -------------------------------------------------------------

def detect_emotion_tokens(tokens: List[str]) -> List[Dict[str, Any]]:

    emotion_tokens: List[Dict[str, Any]] = []

    for idx, token in enumerate(tokens):

        emotion = WORD_TO_EMOTION.get(token)

        if emotion:

            emotion_tokens.append(
                {
                    "token": token,
                    "emotion": emotion,
                    "position": idx,
                }
            )

    return emotion_tokens


# -------------------------------------------------------------
# Token-level intensity heatmap
# -------------------------------------------------------------

def compute_token_intensity(tokens: List[str]) -> List[Dict[str, Any]]:

    heatmap: List[Dict[str, Any]] = []

    for idx, token in enumerate(tokens):

        intensity = 0.0

        if token in WORD_TO_EMOTION:
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


# -------------------------------------------------------------
# Sentence heatmap
# -------------------------------------------------------------

def compute_sentence_heatmap(text: str) -> List[Dict[str, Any]]:

    sentences = tokenize_sentences(text)

    results: List[Dict[str, Any]] = []

    for sentence in sentences:

        tokens = tokenize_words(sentence)

        token_scores = compute_token_intensity(tokens)

        total_intensity = sum(item["intensity"] for item in token_scores)

        normalized = round(total_intensity / max(len(tokens), 1), 4)

        results.append(
            {
                "sentence": sentence,
                "emotion_intensity": normalized,
            }
        )

    return results


# -------------------------------------------------------------
# Gradient attribution
# -------------------------------------------------------------

def _resolve_device(model: Any) -> Optional[torch.device]:

    try:
        return next(model.parameters()).device
    except Exception:
        return None


def compute_integrated_gradients(
    model: Any,
    tokenizer: Any,
    text: str,
) -> List[Dict[str, Any]]:

    device = _resolve_device(model)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    model.zero_grad(set_to_none=True)

    embedding_layer = model.get_input_embeddings()

    input_ids = inputs["input_ids"]

    embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

    outputs = model(
        inputs_embeds=embeddings,
        attention_mask=inputs.get("attention_mask"),
    )

    target = outputs.logits.max()

    target.backward()

    gradients = embeddings.grad

    scores = gradients.abs().sum(dim=-1).detach().cpu().numpy()[0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu())

    return [
        {"token": token, "importance": float(score)}
        for token, score in zip(tokens, scores)
    ]


# -------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------

def generate_heatmap_matrix(
    tokens: List[str],
    heatmap: List[Dict[str, Any]],
) -> List[List[float]]:

    _ = tokens

    return [[float(item["intensity"])] for item in heatmap]


def generate_ui_highlights(
    heatmap: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    return [
        {
            "token": item["token"],
            "strength": item["intensity"],
        }
        for item in heatmap
        if item["intensity"] > 0
    ]


# -------------------------------------------------------------
# Main explanation pipeline
# -------------------------------------------------------------

def explain_emotion(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    tokens = tokenize_words(text)

    emotion_tokens = detect_emotion_tokens(tokens)

    emotion_heatmap = compute_token_intensity(tokens)

    sentence_heatmap = compute_sentence_heatmap(text)

    gradient_attr: List[Dict[str, Any]] = []

    if model is not None and tokenizer is not None:

        try:

            gradient_attr = compute_integrated_gradients(
                model,
                tokenizer,
                text,
            )

        except Exception as exc:
            logger.warning("Gradient attribution failed: %s", exc)

    heatmap_matrix = generate_heatmap_matrix(tokens, emotion_heatmap)

    ui_highlights = generate_ui_highlights(emotion_heatmap)

    explanation = EmotionExplanation(
        emotion_tokens=emotion_tokens,
        emotion_heatmap=emotion_heatmap,
        sentence_heatmap=sentence_heatmap,
        gradient_attribution=gradient_attr,
        heatmap_matrix=heatmap_matrix,
        ui_highlights=ui_highlights,
    )

    return explanation.__dict__