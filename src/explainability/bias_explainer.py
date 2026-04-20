from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.explainability.token_alignment import align_tokens
from src.explainability.utils_validation import validate_tokens_scores

try:
    import shap
except ImportError:  # optional
    shap = None

logger = logging.getLogger(__name__)

BIAS_TERMS = {
    "radical","extremist","far_left","far_right","socialist","communist","fascist","elitist",
    "ultra","hardline","reactionary","authoritarian","corrupt","crooked","rigged","fraud",
    "fraudulent","illegitimate","dishonest","deceptive","manipulative","propaganda",
    "brainwashing","indoctrination","fake","fake_news","misleading","fabricated","distorted",
    "biased","partisan","agenda_driven","media_bias","spin","coverup","elite","establishment",
    "bureaucrat","globalist","oligarch","corporate_elite","power_elite","political_elite",
    "technocrat","disgraceful","outrageous","shocking","terrible","horrible","evil","dangerous",
    "disgusting","absurd","ridiculous","conspiracy","scheme","plot","agenda","hidden_agenda",
    "secret_plan","manipulated","controlled","staged","engineered"
}


@dataclass
class BiasExplanation:
    token_importance: List[Dict[str, Any]]
    integrated_gradients: List[Dict[str, Any]]
    attention_scores: List[Dict[str, Any]]
    biased_tokens: List[str]
    sentence_bias_scores: List[Dict[str, Any]]
    bias_intensity: float
    bias_heatmap: List[Dict[str, Any]]


def _resolve_device(model: Any) -> Optional[torch.device]:
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def tokenize_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def normalize_scores(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    min_v = values.min()
    max_v = values.max()
    if max_v - min_v < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - min_v) / (max_v - min_v)


def compute_lexicon_bias(text: str):
    tokens = re.findall(r"\b[a-z]+\b", text.lower())
    matched = [t for t in tokens if t in BIAS_TERMS]
    score = len(matched) / max(len(tokens), 1)
    return score, list(set(matched))


def compute_sentence_bias(text):
    results = []
    for sentence in tokenize_sentences(text):
        score, tokens = compute_lexicon_bias(sentence)
        results.append({"sentence": sentence, "bias_score": score, "biased_tokens": tokens})
    return results


def compute_shap_importance(model, tokenizer, text):
    if shap is None:
        raise ImportError("SHAP not installed")

    device = _resolve_device(model)

    def predict(texts):
        encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if device is not None:
            encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings)
        return outputs.logits.detach().cpu().numpy()

    explainer = shap.Explainer(predict, tokenizer)
    shap_values = explainer([text])

    tokens = list(shap_values.data[0])
    values = np.asarray(shap_values.values[0], dtype=float)
    if values.ndim > 1:
        values = values.mean(axis=-1)

    return [{"token": t, "importance": float(v)} for t, v in zip(tokens, values)]


def compute_integrated_gradients(model, tokenizer, text):
    device = _resolve_device(model)
    inputs = tokenizer(text, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    model.zero_grad(set_to_none=True)
    embeddings = model.get_input_embeddings()(inputs["input_ids"]).detach().requires_grad_(True)
    outputs = model(inputs_embeds=embeddings, attention_mask=inputs.get("attention_mask"))
    outputs.logits.max().backward()

    grads = embeddings.grad.abs().sum(dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu())
    return [{"token": t, "importance": float(g)} for t, g in zip(tokens, grads.detach().cpu().numpy())]


def compute_attention_scores(model, tokenizer, text):
    device = _resolve_device(model)
    inputs = tokenizer(text, return_tensors="pt")
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    matrix = outputs.attentions[-1].mean(dim=1)[0]
    token_scores = matrix.mean(dim=0)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].detach().cpu())
    return [{"token": t, "attention": float(a)} for t, a in zip(tokens, token_scores.detach().cpu().numpy())]


def compute_bias_intensity(token_importance):
    vals = [abs(t["importance"]) for t in token_importance]
    return float(np.mean(vals)) if vals else 0.0


def extract_biased_tokens(token_importance, threshold=0.05):
    return [t["token"] for t in token_importance if abs(t["importance"]) >= threshold]


def generate_bias_heatmap(token_importance):
    return [{"token": t["token"], "bias_strength": abs(float(t["importance"]))} for t in token_importance]


def explain_bias(model, tokenizer, text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")

    integrated_gradients_result: List[Dict[str, Any]] = []
    try:
        shap_importance = compute_shap_importance(model, tokenizer, text)
    except Exception as e:
        logger.warning("SHAP failed: %s", e)
        integrated_gradients_result = compute_integrated_gradients(model, tokenizer, text)
        shap_importance = integrated_gradients_result

    tokens = [t["token"] for t in shap_importance]
    scores = normalize_scores([t["importance"] for t in shap_importance])
    validate_tokens_scores(tokens, scores)
    tokens, scores = align_tokens(tokens, scores)

    token_importance = [{"token": t, "importance": float(s)} for t, s in zip(tokens, scores)]
    attention_scores = compute_attention_scores(model, tokenizer, text)

    if not integrated_gradients_result:
        integrated_gradients_result = compute_integrated_gradients(model, tokenizer, text)

    explanation = BiasExplanation(
        token_importance=token_importance,
        integrated_gradients=integrated_gradients_result,
        attention_scores=attention_scores,
        biased_tokens=extract_biased_tokens(token_importance),
        sentence_bias_scores=compute_sentence_bias(text),
        bias_intensity=compute_bias_intensity(token_importance),
        bias_heatmap=generate_bias_heatmap(token_importance),
    )
    return explanation.__dict__
