"""src/explainability/bias_explainer.py

Bias explainer — fuses SHAP, integrated gradients, and attention rollout
on the TruthLens multitask model.

Audit fixes
-----------
* **CRIT-1**: the previous implementation called ``model(...).logits`` which
  doesn't exist on the multitask wrapper. All three primitives now route
  through ``model.encoder`` + ``model.heads[task]`` (matching
  ``src/aggregation/score_explainer.py``).
* **FAITH-2**: ``compute_ig`` now performs a real Riemann-sum integration
  over an interpolation path rather than a single-step gradient×input.
* **GPU-3**: tokenizer outputs are explicitly moved to ``model.device``.
* **PERF-3**: SHAP explainers are cached at module level (keyed by
  tokenizer identity + task) so repeated calls don't reconstruct the
  ``shap.maskers.Text`` masker per article.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.explainability.token_alignment import align_tokens
from src.explainability.utils_validation import validate_tokens_scores
from src.explainability.attention_rollout import AttentionRollout

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore

logger = logging.getLogger(__name__)
EPS = 1e-12

DEFAULT_TASK = "bias"
DEFAULT_IG_STEPS = 16

# PERF-3: module-level LRU cache for the (expensive) shap.Explainer.
_SHAP_CACHE_LOCK = threading.RLock()
_SHAP_CACHE_MAX = 4
_SHAP_EXPLAINER_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()


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

def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0)
    return x / (np.sum(x) + EPS)


def _model_device(model) -> torch.device:
    """GPU-3: discover the model's device by walking its parameters."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _is_multitask(model) -> bool:
    """Detect the TruthLens multitask wrapper (``encoder`` + ``heads``)."""
    return hasattr(model, "encoder") and hasattr(model, "heads")


def _resolve_task(model, task: str) -> str:
    """Pick a callable task head on the multitask model.

    Falls back to the first available head if the requested ``task`` is
    not present (older checkpoints used different head names).
    """
    if not _is_multitask(model):
        return task
    heads = model.heads
    if task in heads:
        return task
    try:
        return next(iter(heads.keys()))
    except StopIteration:  # pragma: no cover - empty multitask wrapper
        return task


def _forward_logits(model, enc, *, task: str = DEFAULT_TASK) -> torch.Tensor:
    """Single forward pass that returns task-head logits.

    Works on both the multitask wrapper and a vanilla HF model that
    exposes ``.logits``. CRIT-1.
    """
    if _is_multitask(model):
        out = model.encoder(**enc)
        cls = out.last_hidden_state[:, 0]
        return model.heads[_resolve_task(model, task)](cls)
    out = model(**enc)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict) and "logits" in out:
        return out["logits"]
    raise RuntimeError("Model output does not expose logits")


# =========================================================
# SHAP (CRIT-1 + PERF-3)
# =========================================================

def _shap_cache_key(tokenizer, task: str) -> Tuple[Any, ...]:
    return (id(tokenizer), task)


def _get_shap_explainer(model, tokenizer, *, task: str = DEFAULT_TASK):
    """PERF-3: cache the SHAP Explainer instance per (tokenizer, task).

    The Text masker construction is expensive; without caching it ran
    on every article.
    """
    if shap is None:
        return None

    key = _shap_cache_key(tokenizer, task)

    with _SHAP_CACHE_LOCK:
        cached = _SHAP_EXPLAINER_CACHE.get(key)
        if cached is not None:
            _SHAP_EXPLAINER_CACHE.move_to_end(key)
            return cached

    device = _model_device(model)

    def predict(texts):
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = _forward_logits(model, enc, task=task)
        return logits.detach().cpu().numpy()

    explainer = shap.Explainer(predict, tokenizer)

    with _SHAP_CACHE_LOCK:
        _SHAP_EXPLAINER_CACHE[key] = explainer
        _SHAP_EXPLAINER_CACHE.move_to_end(key)
        while len(_SHAP_EXPLAINER_CACHE) > _SHAP_CACHE_MAX:
            _SHAP_EXPLAINER_CACHE.popitem(last=False)

    return explainer


def compute_shap(model, tokenizer, text, *, task: str = DEFAULT_TASK):
    if shap is None:
        return None

    try:
        explainer = _get_shap_explainer(model, tokenizer, task=task)
        if explainer is None:
            return None

        sv = explainer([text])
        values = sv.values[0]
        if values.ndim > 1:
            values = values.mean(axis=-1)

        return _normalize(values)
    except Exception as exc:
        logger.warning("compute_shap failed: %s", exc)
        return None


# =========================================================
# INTEGRATED GRADIENTS (CRIT-1 + FAITH-2 + GPU-3)
# =========================================================

def compute_ig(
    model,
    tokenizer,
    text,
    *,
    task: str = DEFAULT_TASK,
    target_idx: int = 0,
    steps: int = DEFAULT_IG_STEPS,
):
    """Real path-integrated gradients on the multitask model.

    Mirrors ``src.aggregation.score_explainer._integrated_gradients``:
    interpolates ``steps`` points between a zero baseline and the input
    embedding, averages gradients along the path, and integrates.
    """
    if not _is_multitask(model):
        # GPU-3: explicit device placement for non-multitask fallback.
        device = _model_device(model)
        enc = tokenizer(text, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        emb_layer = model.get_input_embeddings()
        emb = emb_layer(enc["input_ids"]).detach()
        baseline = torch.zeros_like(emb)

        alphas = torch.linspace(0.0, 1.0, steps, device=device, dtype=emb.dtype)
        alphas = alphas.view(-1, 1, 1, 1)
        scaled = (baseline + alphas * (emb - baseline)).flatten(0, 1)
        scaled = scaled.detach().requires_grad_(True)

        out = model(inputs_embeds=scaled)
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, dict):
            logits = out.get("logits")
        if logits is None:
            raise RuntimeError("Model output does not expose logits")

        idx = min(target_idx, logits.shape[-1] - 1)
        logits[:, idx].sum().backward()

        grads = scaled.grad.view(steps, *emb.shape).mean(dim=0)
        ig = ((emb - baseline) * grads).sum(-1)[0].detach().cpu().numpy()
        return _normalize(ig)

    # Multitask path — CRIT-1 / FAITH-2.
    device = _model_device(model)
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    emb = model.encoder.embeddings(enc["input_ids"]).detach()
    baseline = torch.zeros_like(emb)

    alphas = torch.linspace(0.0, 1.0, steps, device=device, dtype=emb.dtype)
    alphas = alphas.view(-1, 1, 1, 1)
    scaled = (baseline + alphas * (emb - baseline)).flatten(0, 1)
    scaled = scaled.detach().requires_grad_(True)

    attn = enc.get("attention_mask")
    if attn is not None:
        attn_rep = attn.repeat(steps, 1)
    else:
        attn_rep = None

    out = model.encoder(inputs_embeds=scaled, attention_mask=attn_rep)
    head = model.heads[_resolve_task(model, task)]
    logits = head(out.last_hidden_state[:, 0])

    idx = min(target_idx, logits.shape[-1] - 1)
    if hasattr(model, "zero_grad"):
        model.zero_grad()
    logits[:, idx].sum().backward()

    grads = scaled.grad.view(steps, *emb.shape).mean(dim=0)
    ig = ((emb - baseline) * grads).sum(-1)[0].detach().cpu().numpy()

    # Zero out attribution on padding positions so normalisation is not
    # diluted by attention-masked tokens.
    if attn is not None:
        mask = attn[0].detach().cpu().numpy().astype(np.float64)
        if mask.shape == ig.shape:
            ig = ig * mask

    return _normalize(ig)


# =========================================================
# ATTENTION ROLLOUT (CRIT-1 + GPU-3)
# =========================================================

def compute_attention_rollout(
    model,
    tokenizer,
    text,
    *,
    task: str = DEFAULT_TASK,
):
    device = _model_device(model)
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    if _is_multitask(model):
        with torch.no_grad():
            outputs = model.encoder(**enc, output_attentions=True)
        attentions = outputs.attentions
    else:
        with torch.no_grad():
            outputs = model(**enc, output_attentions=True)
        attentions = getattr(outputs, "attentions", None)
        if attentions is None and isinstance(outputs, dict):
            attentions = outputs.get("attentions")

    if not attentions:
        return None

    rollout_out = AttentionRollout().compute_rollout(
        attentions=list(attentions),
        tokens=tokenizer.convert_ids_to_tokens(enc["input_ids"][0]),
    )

    return np.asarray(rollout_out.importance, dtype=float)


# =========================================================
# FUSION ENGINE
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

def explain_bias(model, tokenizer, text, *, task: str = DEFAULT_TASK):

    if not text.strip():
        raise ValueError("Empty text")

    tokens = tokenizer.tokenize(text)

    shap_vals = compute_shap(model, tokenizer, text, task=task)
    ig_vals = None
    attn_vals = None
    try:
        ig_vals = compute_ig(model, tokenizer, text, task=task)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("compute_ig failed: %s", exc)
    try:
        attn_vals = compute_attention_rollout(model, tokenizer, text, task=task)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("compute_attention_rollout failed: %s", exc)

    base = next((v for v in [shap_vals, ig_vals, attn_vals] if v is not None), None)
    if base is None:
        raise RuntimeError("All explanation methods failed for bias explainer")

    # TOKEN ALIGNMENT FIX: each attribution array is in subword-token space.
    # align_tokens reduces them to word-level independently so all three arrays
    # share the same length before fusion.  The canonical word-level token list
    # is derived from the base array; each non-None method is reduced separately.
    subword_tokens = list(tokens)

    tokens, base = align_tokens(subword_tokens, base)
    base = np.array(base, dtype=np.float32)

    def _align_or_zero(vals: Optional[np.ndarray]) -> np.ndarray:
        if vals is None:
            return np.zeros_like(base)
        _, aligned = align_tokens(subword_tokens, vals)
        return np.array(aligned, dtype=np.float32)

    shap_vals = _align_or_zero(shap_vals)
    ig_vals = _align_or_zero(ig_vals)
    attn_vals = _align_or_zero(attn_vals)

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


def clear_shap_cache() -> None:
    """Test/utility helper for resetting the PERF-3 cache."""
    with _SHAP_CACHE_LOCK:
        _SHAP_EXPLAINER_CACHE.clear()
