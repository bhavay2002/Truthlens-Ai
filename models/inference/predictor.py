"""Compatibility inference helpers for legacy tests and callers."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from src.utils.input_validation import ensure_non_empty_text, ensure_non_empty_text_list


DEFAULT_FAKE_INDEX = 1

_cached_tokenizer: Optional[Any] = None
_cached_model: Optional[Any] = None


def load_model_and_tokenizer() -> Tuple[Any, Any]:
    global _cached_tokenizer, _cached_model

    if _cached_tokenizer is not None and _cached_model is not None:
        return _cached_tokenizer, _cached_model

    from src.models.registry.model_registry import ModelRegistry

    assets = ModelRegistry.load_model()
    tokenizer = assets["tokenizer"]
    model = assets["model"]
    model.eval()

    _cached_tokenizer = tokenizer
    _cached_model = model

    return tokenizer, model


def _get_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _prepare_texts_for_inference(texts: List[str]) -> List[str]:
    normalized = ensure_non_empty_text_list(texts)
    return [str(text).strip() for text in normalized]


def _resolve_fake_index(model: Any) -> int:
    label2id = getattr(model.config, "label2id", None) or {}
    for label, idx in label2id.items():
        if str(label).upper() == "FAKE":
            try:
                return int(idx)
            except (TypeError, ValueError):
                break

    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        if str(label).upper() == "FAKE":
            try:
                return int(idx)
            except (TypeError, ValueError):
                break

    return DEFAULT_FAKE_INDEX


def predict_batch(texts: List[str]) -> List[List[float]]:
    """Return [[prob_real, prob_fake], ...] for each text — used by LIME."""
    if not texts:
        return []

    tokenizer, model = load_model_and_tokenizer()
    device = _get_device(model)
    batch_texts = _prepare_texts_for_inference(texts)

    inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    return probs.cpu().tolist()


def predict(text: str) -> dict[str, float | str]:
    ensure_non_empty_text(text)

    tokenizer, model = load_model_and_tokenizer()
    device = _get_device(model)
    batch_texts = _prepare_texts_for_inference([text])

    inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    probs_cpu = probs.cpu()
    fake_index = _resolve_fake_index(model)
    pred_index = int(torch.argmax(probs_cpu, dim=1).item())

    fake_probability = float(probs_cpu[0, fake_index].item())
    confidence = float(torch.max(probs_cpu, dim=1).values.item())

    label = "Fake" if pred_index == fake_index else "Real"

    return {
        "label": label,
        "fake_probability": fake_probability,
        "confidence": confidence,
    }
