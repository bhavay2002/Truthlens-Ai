"""
File Name: predict_api.py
Module: src.inference

Singleton-style functional inference API.

This module is the canonical location for the function-based ``predict``,
``predict_batch``, and ``batch_predict`` entry-points used by the FastAPI
service and by tests.

It wraps :class:`src.models.inference.predictor.Predictor` (the class-based
predictor) with lazy, process-wide singleton state so callers do not have to
manage model loading themselves.
"""

from typing import Any, Dict, List

import torch

from src.models.inference.predictor import Predictor
from src.models.registry.model_registry import ModelRegistry
from src.utils.device_utils import autocast_context, get_device, move_to_device
from src.utils.input_validation import ensure_non_empty_text
from src.utils.settings import load_settings

_SETTINGS = load_settings()

_predictor: Predictor | None = None
_tokenizer = None
_device = get_device()


def _load_assets():
    global _predictor, _tokenizer

    if _predictor is None or _tokenizer is None:
        assets = ModelRegistry.load_model()

        model = assets["model"]
        _tokenizer = assets["tokenizer"]

        model = move_to_device(model, _device)

        _predictor = Predictor(model=model)

    return _predictor, _tokenizer


def _prepare_inputs(texts: List[str]):
    _, tokenizer = _load_assets()

    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=_SETTINGS.model.max_length,
        return_tensors="pt",
    )

    return move_to_device(inputs, _device)


def predict(text: str) -> Dict[str, Any]:
    ensure_non_empty_text(text)

    predictor, _ = _load_assets()
    inputs = _prepare_inputs([text])

    with torch.no_grad():
        with autocast_context():
            outputs = predictor.predict_batch(inputs)

    return predictor.build_fake_real_output(outputs)


def predict_batch(texts: List[str]) -> List[List[float]]:
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list of strings")

    for t in texts:
        ensure_non_empty_text(t)

    predictor, _ = _load_assets()
    inputs = _prepare_inputs(texts)

    with torch.no_grad():
        with autocast_context():
            return predictor.predict_batch_pairs(inputs)


def batch_predict(texts: List[str]) -> List[Dict[str, Any]]:
    probs = predict_batch(texts)

    results: List[Dict[str, Any]] = []
    for prob_real, prob_fake in probs:
        results.append(
            {
                "fake_probability": float(prob_fake),
                "label": "Fake" if prob_fake > 0.5 else "Real",
            }
        )

    return results


# Backwards-compatible alias
predict.batch_predict = batch_predict
