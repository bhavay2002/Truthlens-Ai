"""Compatibility inference helpers for legacy tests and callers."""

from __future__ import annotations

from typing import Any, List

import torch

from src.utils.input_validation import ensure_non_empty_text, ensure_non_empty_text_list


DEFAULT_FAKE_INDEX = 1


def load_model_and_tokenizer() -> tuple[Any, Any]:
    from src.models.registry.model_registry import ModelRegistry

    assets = ModelRegistry.load_model()
    return assets["tokenizer"], assets["model"]


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


def predict(text: str) -> dict[str, float | str]:
    ensure_non_empty_text(text)

    tokenizer, model = load_model_and_tokenizer()
    batch_texts = _prepare_texts_for_inference([text])

    inputs = tokenizer(
        batch_texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    fake_index = _resolve_fake_index(model)
    pred_index = int(torch.argmax(probs, dim=1).item())

    fake_probability = float(probs[0, fake_index].item())
    confidence = float(torch.max(probs, dim=1).values.item())

    label = "Fake" if pred_index == fake_index else "Real"

    return {
        "label": label,
        "fake_probability": fake_probability,
        "confidence": confidence,
    }
