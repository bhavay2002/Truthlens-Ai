"""
File Name: bias_explainer.py
Module: Explainability - Bias Analysis
Description:
    Provides interpretability utilities for bias detection outputs within
    the TruthLens AI system. The module analyzes text using lexical bias
    signals, SHAP token importance, integrated gradients, and transformer
    attention scores to generate human-readable bias explanations.

Dependencies:
    logging
    re
    dataclasses
    typing
    numpy
    torch
    shap (optional)

Inputs:
    model : transformer model
    tokenizer : tokenizer compatible with the model
    text : input text

Outputs:
    Structured explanation dictionary with token importance, sentence bias
    scores, attention weights, and visualization-ready heatmaps.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore

try:
    from src.features.bias.bias_lexicon import (
        compute_bias_features as _external_compute_bias_features,
    )
except ImportError:  # pragma: no cover
    _external_compute_bias_features = None

logger = logging.getLogger(__name__)


_FALLBACK_BIAS_TERMS = {
    "biased",
    "corrupt",
    "crooked",
    "disgraceful",
    "disgusting",
    "elite",
    "evil",
    "fake",
    "horrible",
    "manipulate",
    "radical",
    "rigged",
    "shocking",
    "terrible",
    "unbelievable",
}


@dataclass
class _FallbackBiasFeatures:
    bias_score: float
    biased_tokens: List[str]


@dataclass
class BiasExplanation:
    token_importance: List[Dict[str, Any]]
    integrated_gradients: List[Dict[str, Any]]
    biased_token_highlights: List[str]
    sentence_bias_scores: List[Dict[str, Any]]
    attention_scores: List[Dict[str, Any]]
    bias_heatmap: List[Dict[str, Any]]


def _compute_bias_features_compat(text: str) -> Any:
    if _external_compute_bias_features is not None:
        result = _external_compute_bias_features(text)

        if not hasattr(result, "bias_score") or not hasattr(
            result, "biased_tokens"
        ):
            raise RuntimeError(
                "compute_bias_features must return object with "
                "'bias_score' and 'biased_tokens'."
            )

        return result

    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    matched = [token for token in tokens if token in _FALLBACK_BIAS_TERMS]

    unique_tokens: List[str] = []
    seen: set[str] = set()

    for token in matched:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)

    score = len(matched) / max(len(tokens), 1)

    return _FallbackBiasFeatures(
        bias_score=round(float(score), 4),
        biased_tokens=unique_tokens,
    )


def tokenize_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def compute_sentence_bias(text: str) -> List[Dict[str, Any]]:
    sentences = tokenize_sentences(text)
    results: List[Dict[str, Any]] = []

    for sentence in sentences:
        bias_result = _compute_bias_features_compat(sentence)

        results.append(
            {
                "sentence": sentence,
                "bias_score": bias_result.bias_score,
                "biased_tokens": bias_result.biased_tokens,
            }
        )

    return results


def _resolve_device(model) -> Optional[torch.device]:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return None


def _normalize_token_scores(values: Any) -> np.ndarray:
    values = np.asarray(values)

    if values.ndim == 0:
        return np.asarray([float(values)])

    if values.ndim == 1:
        return values.astype(float)

    return values.mean(axis=-1).astype(float)


def compute_shap_importance(
    model,
    tokenizer,
    text: str,
) -> List[Dict[str, Any]]:
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
            encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)

        logits = outputs.logits.detach().cpu().numpy()
        return logits

    explainer = shap.Explainer(predict, tokenizer)
    shap_values = explainer([text])

    tokens = list(shap_values.data[0])
    values = _normalize_token_scores(shap_values.values[0])

    return [
        {"token": token, "importance": float(value)}
        for token, value in zip(tokens, values)
    ]


def compute_integrated_gradients(
    model,
    tokenizer,
    text: str,
) -> List[Dict[str, Any]]:
    device = _resolve_device(model)

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

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
        raise RuntimeError("Gradient computation failed.")

    importance = gradients.abs().sum(dim=-1).detach().cpu().numpy()[0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu())

    return [
        {"token": token, "importance": float(score)}
        for token, score in zip(tokens, importance)
    ]


def compute_attention_scores(
    model,
    tokenizer,
    text: str,
) -> List[Dict[str, Any]]:
    device = _resolve_device(model)

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions[-1]

    attention_matrix = attentions.mean(dim=1)[0].detach().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0].detach().cpu()
    )

    token_scores = attention_matrix.mean(axis=0)

    return [
        {"token": token, "attention": float(score)}
        for token, score in zip(tokens, token_scores)
    ]


def extract_biased_tokens(
    token_importance: List[Dict[str, Any]],
    threshold: float = 0.05,
) -> List[str]:
    return [
        str(item["token"])
        for item in token_importance
        if abs(float(item["importance"])) >= threshold
    ]


def generate_bias_heatmap(
    token_importance: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [
        {
            "token": item["token"],
            "bias_strength": round(abs(float(item["importance"])), 4),
        }
        for item in token_importance
    ]


def explain_bias(
    model,
    tokenizer,
    text: str,
) -> Dict[str, Any]:
    if model is None or tokenizer is None:
        raise ValueError("model and tokenizer are required.")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    try:
        shap_importance = compute_shap_importance(model, tokenizer, text)
    except Exception as exc:
        logger.warning(
            "SHAP attribution failed, using gradient fallback: %s",
            exc,
        )
        shap_importance = compute_integrated_gradients(
            model,
            tokenizer,
            text,
        )

    biased_tokens = extract_biased_tokens(shap_importance)

    ig_importance = compute_integrated_gradients(model, tokenizer, text)

    attention_scores = compute_attention_scores(model, tokenizer, text)

    sentence_scores = compute_sentence_bias(text)

    heatmap = generate_bias_heatmap(shap_importance)

    explanation = BiasExplanation(
        token_importance=shap_importance,
        integrated_gradients=ig_importance,
        biased_token_highlights=biased_tokens,
        sentence_bias_scores=sentence_scores,
        attention_scores=attention_scores,
        bias_heatmap=heatmap,
    )

    return explanation.__dict__
