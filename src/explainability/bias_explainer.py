"""Bias explainability utilities for classification outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

try:
    import shap
except ImportError:  # pragma: no cover - environment-dependent
    shap = None  # type: ignore[assignment]

from src.features.bias.bias_lexicon import compute_bias_features


@dataclass
class BiasExplanation:
    """Structured explanation object."""

    token_importance: List[Dict[str, Any]]
    biased_token_highlights: List[str]
    sentence_bias_scores: List[Dict[str, Any]]
    attention_scores: List[Dict[str, Any]]
    bias_heatmap: List[Dict[str, Any]]


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def compute_sentence_bias(text: str) -> List[Dict[str, Any]]:
    """Compute bias score for each sentence."""
    sentences = tokenize_sentences(text)
    results: list[dict[str, Any]] = []

    for sentence in sentences:
        bias_result = compute_bias_features(sentence)
        results.append(
            {
                "sentence": sentence,
                "bias_score": bias_result.bias_score,
                "biased_tokens": bias_result.biased_tokens,
            }
        )

    return results


def _resolve_device(model) -> torch.device | None:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def _normalize_token_scores(raw_values: Any) -> np.ndarray:
    values = np.asarray(raw_values)
    if values.ndim == 0:
        return np.asarray([float(values)])
    if values.ndim == 1:
        return values.astype(float)
    # For multi-class outputs, average importance across class dimension.
    return values.mean(axis=-1).astype(float)


def compute_shap_importance(
    model, tokenizer, text: str
) -> List[Dict[str, Any]]:
    """Compute SHAP token importance."""
    if shap is None:
        raise ImportError("SHAP is not installed.")

    device = _resolve_device(model)

    def predict(texts):
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if device is not None:
            encodings = {
                key: value.to(device) for key, value in encodings.items()
            }

        with torch.no_grad():
            outputs = model(**encodings)

        logits = outputs.logits.detach().cpu().numpy()
        return logits

    explainer = shap.Explainer(predict, tokenizer)
    shap_values = explainer([text])

    tokens = list(shap_values.data[0])
    values = _normalize_token_scores(shap_values.values[0])

    return [
        {
            "token": token,
            "importance": float(value),
        }
        for token, value in zip(tokens, values)
    ]


def compute_integrated_gradients(
    model, tokenizer, text: str
) -> List[Dict[str, Any]]:
    """Compute token attribution using gradient magnitudes."""
    device = _resolve_device(model)

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

    importance = gradients.abs().sum(dim=-1).detach().cpu().numpy()[0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu())

    return [
        {
            "token": token,
            "importance": float(score),
        }
        for token, score in zip(tokens, importance)
    ]


def compute_attention_scores(
    model, tokenizer, text: str
) -> List[Dict[str, Any]]:
    """Extract attention-based token importance."""
    device = _resolve_device(model)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions[-1]
    attention_matrix = attentions.mean(dim=1)[0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].detach().cpu()
    )
    token_scores = attention_matrix.mean(axis=0)

    return [
        {
            "token": token,
            "attention": float(score),
        }
        for token, score in zip(tokens, token_scores)
    ]


def extract_biased_tokens(
    token_importance: List[Dict[str, Any]],
    threshold: float = 0.05,
) -> List[str]:
    """Extract tokens above the given absolute-importance threshold."""
    return [
        str(item["token"])
        for item in token_importance
        if abs(float(item["importance"])) >= threshold
    ]


def generate_bias_heatmap(
    token_importance: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate heatmap-ready token importance scores."""
    return [
        {
            "token": item["token"],
            "bias_strength": round(abs(float(item["importance"])), 4),
        }
        for item in token_importance
    ]


def explain_bias(model, tokenizer, text: str) -> Dict[str, Any]:
    """Run complete bias explanation pipeline."""
    if model is None or tokenizer is None:
        raise ValueError(
            "model and tokenizer are required for bias explanation."
        )

    shap_importance = compute_shap_importance(model, tokenizer, text)
    biased_tokens = extract_biased_tokens(shap_importance)

    ig_importance = compute_integrated_gradients(model, tokenizer, text)
    attention_scores = compute_attention_scores(model, tokenizer, text)
    sentence_scores = compute_sentence_bias(text)
    heatmap = generate_bias_heatmap(shap_importance)

    return {
        "token_importance": shap_importance,
        "integrated_gradients": ig_importance,
        "biased_token_highlights": biased_tokens,
        "sentence_bias_scores": sentence_scores,
        "attention_scores": attention_scores,
        "bias_heatmap": heatmap,
    }
