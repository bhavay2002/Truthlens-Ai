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

try:
    from src.features.emotion.emotion_lexicon import (
        DEFAULT_NRC_LEXICON as _IMPORTED_NRC_LEXICON,
    )
except ImportError:  # pragma: no cover
    _IMPORTED_NRC_LEXICON = None

try:
    from src.features.emotion.emotion_intensity import (
        INTENSIFIER_ADVERBS as _IMPORTED_INTENSIFIER_ADVERBS,
    )
except ImportError:  # pragma: no cover
    try:
        from src.features.emotion.emotion_intensity import (
            INTENSIFIERS as _IMPORTED_INTENSIFIER_ADVERBS,
        )
    except ImportError:  # pragma: no cover
        _IMPORTED_INTENSIFIER_ADVERBS = None

logger = logging.getLogger(__name__)


_FALLBACK_NRC_LEXICON: Dict[str, set[str]] = {
    "anger": {"angry", "furious", "rage", "outrage"},
    "fear": {"fear", "afraid", "panic", "threat"},
    "joy": {"joy", "happy", "delight", "celebrate"},
    "sadness": {"sad", "grief", "mourn", "tragic"},
    "surprise": {"surprised", "shocking", "unexpected", "sudden"},
    "disgust": {"disgusting", "revolting", "repulsive"},
    "trust": {"trust", "reliable", "credible", "honest"},
    "anticipation": {"expect", "await", "forecast", "upcoming"},
}

_FALLBACK_INTENSIFIERS = {
    "very",
    "extremely",
    "highly",
    "incredibly",
    "really",
    "so",
    "too",
}


def _normalize_lexicon(raw_lexicon: Any) -> Dict[str, set[str]]:
    """Normalize lexicon structure."""
    if not isinstance(raw_lexicon, dict):
        return {}

    normalized: Dict[str, set[str]] = {}

    for emotion, words in raw_lexicon.items():
        if not isinstance(words, (list, tuple, set)):
            continue

        token_set = {
            str(word).strip().lower()
            for word in words
            if isinstance(word, str) and word.strip()
        }

        if token_set:
            normalized[str(emotion).strip().lower()] = token_set

    return normalized


DEFAULT_NRC_LEXICON = (
    _normalize_lexicon(_IMPORTED_NRC_LEXICON) or _FALLBACK_NRC_LEXICON
)

INTENSIFIER_ADVERBS = {
    str(token).strip().lower()
    for token in (
        _IMPORTED_INTENSIFIER_ADVERBS
        if _IMPORTED_INTENSIFIER_ADVERBS is not None
        else _FALLBACK_INTENSIFIERS
    )
    if isinstance(token, str) and token.strip()
}


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
    """Tokenize text into lowercase word tokens."""
    return re.findall(r"\b[a-z]+\b", text.lower())


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def detect_emotion_tokens(tokens: List[str]) -> List[Dict[str, Any]]:
    """Detect tokens associated with emotions."""
    emotion_tokens: List[Dict[str, Any]] = []

    for idx, token in enumerate(tokens):
        matched = [
            emotion
            for emotion, words in DEFAULT_NRC_LEXICON.items()
            if token in words
        ]

        if matched:
            emotion_tokens.append(
                {
                    "token": token,
                    "emotions": matched,
                    "position": idx,
                }
            )

    return emotion_tokens


def compute_token_intensity(tokens: List[str]) -> List[Dict[str, Any]]:
    """Compute token-level emotional intensity."""
    heatmap: List[Dict[str, Any]] = []

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


def _resolve_device(model: Any) -> Optional[torch.device]:
    """Resolve model device."""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def compute_integrated_gradients(
    model: Any,
    tokenizer: Any,
    text: str,
) -> List[Dict[str, Any]]:
    """Compute gradient-based attribution for tokens."""
    device = _resolve_device(model)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    if hasattr(model, "zero_grad"):
        model.zero_grad(set_to_none=True)

    embedding_layer = model.get_input_embeddings()
    input_ids = inputs["input_ids"]

    input_embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

    model_kwargs: Dict[str, Any] = {
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
        raise RuntimeError("Failed to compute gradients for attribution.")

    scores = gradients.abs().sum(dim=-1).detach().cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu())

    return [
        {"token": token, "importance": float(score)}
        for token, score in zip(tokens, scores)
    ]


def generate_heatmap_matrix(
    tokens: List[str],
    heatmap: List[Dict[str, Any]],
) -> List[List[float]]:
    """Convert token heatmap to visualization matrix."""
    _ = tokens
    return [[float(item["intensity"])] for item in heatmap]


def generate_ui_highlights(
    heatmap: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate UI-ready highlighting data."""
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
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run full emotion explanation pipeline.

    Returns structured explainability data for UI or analytics.
    """

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
        except Exception as exc:  # pragma: no cover
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