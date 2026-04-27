from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.explainability.token_alignment import align_tokens
from src.explainability.utils_validation import validate_tokens_scores
from src.explainability.attention_rollout import AttentionRollout

try:
    import shap
except ImportError:
    shap = None

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# DATA MODEL
# =========================================================

@dataclass
class BiasExplanation:
    tokens: List[str]
    importance: List[float]

    shap: List[float]
    integrated_gradients: List[float]
    attention: List[float]

    fused_importance: List[float]

    biased_tokens: List[str]
    bias_intensity: float

    method_weights: Dict[str, float]


# =========================================================
# UTILS
# =========================================================

def _normalize(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    x = np.maximum(x, 0)
    return x / (np.sum(x) + EPS)


def _safe_mean(arrs):
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return None
    return np.mean(arrs, axis=0)


# =========================================================
# CORE METHODS
# =========================================================

def compute_shap(model, tokenizer, text):
    if shap is None:
        return None

    try:
        def predict(texts):
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                out = model(**enc)
            return out.logits.detach().cpu().numpy()

        explainer = shap.Explainer(predict, tokenizer)
        sv = explainer([text])

        values = sv.values[0]
        if values.ndim > 1:
            values = values.mean(axis=-1)

        return _normalize(values)

    except Exception:
        return None


def compute_ig(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")

    emb = model.get_input_embeddings()(inputs["input_ids"]).detach().requires_grad_(True)

    out = model(inputs_embeds=emb)
    out.logits.max().backward()

    grads = emb.grad.abs().sum(dim=-1)[0].detach().cpu().numpy()

    return _normalize(grads)


def compute_attention_rollout(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    rollout_out = AttentionRollout().compute_rollout(
        attentions=outputs.attentions,
        tokens=tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    )

    return np.asarray(rollout_out.importance, dtype=float)


# =========================================================
# FUSION ENGINE 🔥
# =========================================================

def fuse_methods(shap_vals, ig_vals, attn_vals):

    weights = {
        "shap": 0.4 if shap_vals is not None else 0.0,
        "ig": 0.3 if ig_vals is not None else 0.0,
        "attn": 0.3 if attn_vals is not None else 0.0,
    }

    total = sum(weights.values()) + EPS
    weights = {k: v / total for k, v in weights.items()}

    fused = (
        (weights["shap"] * shap_vals if shap_vals is not None else 0) +
        (weights["ig"] * ig_vals if ig_vals is not None else 0) +
        (weights["attn"] * attn_vals if attn_vals is not None else 0)
    )

    return _normalize(fused), weights


# =========================================================
# MAIN API
# =========================================================

def explain_bias(model, tokenizer, text):

    if not text.strip():
        raise ValueError("Empty text")

    tokens = tokenizer.tokenize(text)

    shap_vals = compute_shap(model, tokenizer, text)
    ig_vals = compute_ig(model, tokenizer, text)
    attn_vals = compute_attention_rollout(model, tokenizer, text)

    # alignment — use first available array (avoids numpy boolean ambiguity)
    base = next((v for v in [shap_vals, ig_vals, attn_vals] if v is not None), None)
    if base is None:
        raise RuntimeError("All explanation methods failed for bias explainer")
    tokens, base = align_tokens(tokens, base)

    shap_vals = shap_vals if shap_vals is not None else np.zeros_like(base)
    ig_vals = ig_vals if ig_vals is not None else np.zeros_like(base)
    attn_vals = attn_vals if attn_vals is not None else np.zeros_like(base)

    fused, weights = fuse_methods(shap_vals, ig_vals, attn_vals)

    validate_tokens_scores(tokens, fused)

    biased_tokens = [
        t for t, s in zip(tokens, fused) if s > 0.05
    ]

    return BiasExplanation(
        tokens=tokens,
        importance=fused.tolist(),

        shap=shap_vals.tolist(),
        integrated_gradients=ig_vals.tolist(),
        attention=attn_vals.tolist(),

        fused_importance=fused.tolist(),

        biased_tokens=biased_tokens,
        bias_intensity=float(np.mean(fused)),

        method_weights=weights,
    ).__dict__