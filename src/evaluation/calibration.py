from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.evaluation.reliability_diagram import ReliabilityDiagram
# CFG6: ``TemperatureScaler`` now lives in ``src.models.calibration``.
# We re-export it from this module so the public ``src.evaluation.
# calibration.TemperatureScaler`` symbol stays stable, while the
# import arrow runs ``evaluation -> models`` like every other
# production-stack import (the previous arrangement had the models
# layer importing from evaluation, which was a layering violation).
from src.models.calibration.temperature_scaling import TemperatureScaler  # noqa: F401

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# VALIDATION
# =========================================================

def _validate_inputs(y_true, probs) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)

    if y.shape[0] != p.shape[0]:
        raise ValueError("Mismatch in samples between y_true and probs")

    return y, p


# =========================================================
# ACTIVATIONS
# =========================================================

def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + EPS)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


# =========================================================
# CONFIDENCE
# =========================================================

def extract_confidence(probs: np.ndarray, *, task_type: str = "multiclass") -> np.ndarray:
    """Top-class confidence with multilabel-aware semantics."""
    arr = np.asarray(probs, dtype=float)

    if task_type == "multilabel":
        return np.mean(np.maximum(arr, 1.0 - arr), axis=1)

    if arr.ndim == 1:
        # binary as P(class=1) — confidence is max(p, 1-p)
        return np.maximum(arr, 1.0 - arr)

    if arr.ndim == 2 and arr.shape[1] == 2 and task_type == "binary":
        return np.max(arr, axis=1)

    return np.max(arr, axis=1)


# =========================================================
# TEMPERATURE SCALER
# =========================================================
# CFG6: ``TemperatureScaler`` is now defined in
# ``src.models.calibration.temperature_scaling`` and re-exported above.
# The class is intentionally NOT redefined here.


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    *,
    max_iter: int = 50,
) -> float:
    """Fit a single scalar temperature on validation logits."""
    logits_t = torch.tensor(np.asarray(logits), dtype=torch.float32)
    labels_t = torch.tensor(np.asarray(labels))

    model = TemperatureScaler()
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=max_iter)

    if task_type == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
        labels_long = labels_t.long()

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(model(logits_t), labels_long)
            loss.backward()
            return loss

    elif task_type == "binary":
        loss_fn = nn.BCEWithLogitsLoss()
        labels_float = labels_t.float()

        def closure():
            optimizer.zero_grad()
            scaled = model(logits_t).reshape(-1)
            loss = loss_fn(scaled, labels_float.reshape(-1))
            loss.backward()
            return loss

    elif task_type == "multilabel":
        loss_fn = nn.BCEWithLogitsLoss()
        labels_float = labels_t.float()

        def closure():
            optimizer.zero_grad()
            scaled = model(logits_t)
            loss = loss_fn(scaled, labels_float)
            loss.backward()
            return loss

    else:
        raise ValueError(f"Unsupported task_type for temperature scaling: {task_type}")

    optimizer.step(closure)

    T = float(model.temperature.detach().cpu().item())
    if not np.isfinite(T) or T <= 0:
        logger.warning("Temperature optimization produced invalid value %s; falling back to 1.0", T)
        T = 1.0

    logger.info("[CALIBRATION] Learned temperature: %.4f", T)
    return T


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return np.asarray(logits, dtype=float) / max(T, EPS)


# =========================================================
# BRIER SCORE
# =========================================================

def brier_score(y_true, probs, task_type: str) -> float:
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)

    if task_type == "multiclass":
        if p.ndim != 2:
            raise ValueError("multiclass brier requires 2D probs")
        one_hot = np.eye(p.shape[1])[y]
        return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))

    if task_type == "binary":
        if p.ndim == 2 and p.shape[1] == 2:
            p = p[:, 1]
        return float(np.mean((p.reshape(-1) - y.reshape(-1)) ** 2))

    if task_type == "multilabel":
        return float(np.mean((p - y) ** 2))

    raise ValueError(f"Unsupported task_type: {task_type}")


# =========================================================
# ECE (Expected Calibration Error)
# =========================================================

def expected_calibration_error(
    y_true,
    probs,
    n_bins: int = 10,
    *,
    task_type: str = "binary",
) -> float:
    """ECE for a 1D confidence vector or 2D (multiclass) prob matrix."""
    y, p = _validate_inputs(y_true, probs)

    if p.ndim == 2 and task_type == "multiclass":
        confidence = np.max(p, axis=1)
        preds = np.argmax(p, axis=1)
        correct = (preds == y).astype(float)
    elif p.ndim == 2 and task_type == "binary" and p.shape[1] == 2:
        confidence = p[:, 1]
        correct = (y == 1).astype(float)
    else:
        confidence = p.reshape(-1)
        correct = y.reshape(-1).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidence, bins) - 1, 0, n_bins - 1)

    counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    sum_acc = np.bincount(bin_ids, weights=correct, minlength=n_bins)
    sum_conf = np.bincount(bin_ids, weights=confidence, minlength=n_bins)

    safe = np.where(counts > 0, counts, 1.0)
    bin_acc = sum_acc / safe
    bin_conf = sum_conf / safe

    ece = float(np.sum(counts / max(counts.sum(), 1.0) * np.abs(bin_acc - bin_conf)))
    return ece


def classwise_ece(y_true, probs, n_bins=10) -> Dict[str, float]:
    y, p = _validate_inputs(y_true, probs)
    if p.ndim != 2:
        raise ValueError("classwise_ece requires 2D probs")

    return {
        f"class_{c}": expected_calibration_error(
            (y == c).astype(int), p[:, c], n_bins, task_type="binary"
        )
        for c in range(p.shape[1])
    }


def multilabel_ece(y_true, probs, n_bins=10) -> Dict[str, Any]:
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)

    per_label = [
        expected_calibration_error(
            y[:, i].astype(int), p[:, i], n_bins, task_type="binary"
        )
        for i in range(p.shape[1])
    ]

    return {
        "macro_ece": float(np.mean(per_label)),
        "per_label_ece": per_label,
    }


# =========================================================
# RELIABILITY HELPER
# =========================================================

def compute_reliability(y_true, probs, n_bins: int = 10, *, task_type: str = "multiclass"):
    try:
        diagram = ReliabilityDiagram(n_bins=n_bins)
        return diagram.compute(probs=probs, y_true=y_true, task_type=task_type)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Reliability diagram failed: %s", e)
        return {}


# =========================================================
# FULL PIPELINE — fit + apply
# =========================================================

def compute_calibration(
    logits: Optional[np.ndarray],
    y_true: Iterable,
    task_type: str,
    *,
    apply_temp_scaling: bool = True,
    temperature: Optional[float] = None,
    n_bins: int = 10,
    return_confidence_array: bool = False,
) -> Dict[str, Any]:
    """Fit (or apply pre-fit) temperature scaling and compute calibration metrics.

    Pass ``temperature`` to use a previously fitted value (split fit/apply across
    validation/test data). Otherwise the temperature is fitted on ``logits`` for
    backward compatibility.
    """
    if logits is None:
        raise ValueError("logits required")

    logits_arr = np.asarray(logits, dtype=float)
    y_true_arr = np.asarray(y_true)

    T: Optional[float] = None
    if temperature is not None:
        T = float(temperature)
        scaled = apply_temperature(logits_arr, T)
    elif apply_temp_scaling:
        try:
            T = fit_temperature(logits_arr, y_true_arr, task_type)
            scaled = apply_temperature(logits_arr, T)
        except Exception as exc:
            logger.warning("Temperature fitting failed: %s", exc)
            T = None
            scaled = logits_arr
    else:
        scaled = logits_arr

    if task_type == "multiclass":
        probs = softmax(scaled)
    elif task_type in ("binary", "multilabel"):
        probs = sigmoid(scaled)
    else:
        raise ValueError(f"Invalid task_type: {task_type}")

    confidence = extract_confidence(probs, task_type=task_type)

    results: Dict[str, Any] = {
        "task_type": task_type,
        "mean_confidence": float(np.mean(confidence)),
        "std_confidence": float(np.std(confidence)),
    }

    if return_confidence_array:
        results["confidence"] = confidence.tolist()

    if task_type == "multilabel":
        results.update(multilabel_ece(y_true_arr, probs, n_bins))
    elif task_type == "binary":
        if probs.ndim == 2 and probs.shape[1] == 2:
            results["ece"] = expected_calibration_error(
                y_true_arr, probs, n_bins, task_type="binary"
            )
        else:
            results["ece"] = expected_calibration_error(
                y_true_arr, probs, n_bins, task_type="binary"
            )
    else:  # multiclass
        results["ece"] = expected_calibration_error(
            y_true_arr, probs, n_bins, task_type="multiclass"
        )
        results["classwise_ece"] = classwise_ece(y_true_arr, probs, n_bins)

    results["reliability_diagram"] = compute_reliability(
        y_true_arr, probs, n_bins, task_type=task_type
    )

    results["brier"] = brier_score(y_true_arr, probs, task_type)

    if T is not None:
        results["temperature"] = T

    return results


def fit_calibration(
    val_logits: np.ndarray,
    val_y_true: Iterable,
    task_type: str,
) -> Optional[float]:
    """Convenience wrapper exposing temperature fitting for the fit-then-apply flow."""
    if task_type not in ("binary", "multiclass", "multilabel"):
        raise ValueError(f"Invalid task_type: {task_type}")

    return fit_temperature(np.asarray(val_logits, dtype=float), np.asarray(val_y_true), task_type)


__all__ = [
    "TemperatureScaler",
    "apply_temperature",
    "brier_score",
    "classwise_ece",
    "compute_calibration",
    "compute_reliability",
    "expected_calibration_error",
    "extract_confidence",
    "fit_calibration",
    "fit_temperature",
    "multilabel_ece",
    "sigmoid",
    "softmax",
]
