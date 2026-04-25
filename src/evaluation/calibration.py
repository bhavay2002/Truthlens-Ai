from __future__ import annotations

import logging
from typing import Iterable, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.calibration import CalibrationMetricConfig, CalibrationMetrics

# 🔥 NEW
from src.evaluation.reliability_diagram import ReliabilityDiagram

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# VALIDATION
# =========================================================

def _validate_inputs(y_true, probs):
    y = np.asarray(y_true)
    p = np.asarray(probs)

    if y.shape[0] != p.shape[0]:
        raise ValueError("Mismatch in samples")

    return y, p


# =========================================================
# TEMPERATURE SCALER
# =========================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / torch.clamp(self.temperature, min=EPS)


# =========================================================
# TEMPERATURE FITTING
# =========================================================

def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    max_iter: int = 50,
) -> float:

    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels)

    model = TemperatureScaler()
    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=max_iter)

    if task_type == "multiclass":

        loss_fn = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(model(logits), labels.long())
            loss.backward()
            return loss

    elif task_type == "binary":

        loss_fn = nn.BCEWithLogitsLoss()
        labels = labels.float()

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(model(logits).squeeze(-1), labels)
            loss.backward()
            return loss

    else:
        raise ValueError("Temperature scaling not supported for multilabel")

    optimizer.step(closure)

    T = float(model.temperature.detach().cpu().item())
    logger.info(f"[CALIBRATION] Learned temperature: {T:.4f}")

    return T


def apply_temperature(logits: np.ndarray, T: float):
    return logits / max(T, EPS)


# =========================================================
# ACTIVATIONS
# =========================================================

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + EPS)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================================================
# CONFIDENCE (NEW)
# =========================================================

def extract_confidence(probs: np.ndarray) -> np.ndarray:
    if probs.ndim == 1:
        return probs
    return np.max(probs, axis=1)


# =========================================================
# BRIER SCORE
# =========================================================

def brier_score(y_true, probs, task_type: str):
    y = np.asarray(y_true)
    p = np.asarray(probs)

    if task_type == "multiclass":
        one_hot = np.eye(p.shape[1])[y]
        return float(np.mean((p - one_hot) ** 2))

    return float(np.mean((p - y) ** 2))


# =========================================================
# ECE
# =========================================================

def expected_calibration_error(y_true, probs, n_bins=10):
    y, p = _validate_inputs(y_true, probs)

    metric = CalibrationMetrics(CalibrationMetricConfig(n_bins=n_bins))

    ece = metric.expected_calibration_error(
        torch.from_numpy(p.astype(np.float32)),
        torch.from_numpy(y),
    )

    return float(ece.item() if isinstance(ece, torch.Tensor) else ece)


# =========================================================
# CLASSWISE ECE
# =========================================================

def classwise_ece(y_true, probs, n_bins=10):
    y, p = _validate_inputs(y_true, probs)

    num_classes = p.shape[1]
    result = {}

    for c in range(num_classes):
        y_bin = (y == c).astype(int)
        p_c = p[:, c]
        result[f"class_{c}"] = expected_calibration_error(y_bin, p_c, n_bins)

    return result


# =========================================================
# MULTILABEL ECE
# =========================================================

def multilabel_ece(y_true, probs, n_bins=10):
    y = np.asarray(y_true)
    p = np.asarray(probs)

    per_label = [
        expected_calibration_error(y[:, i], p[:, i], n_bins)
        for i in range(p.shape[1])
    ]

    return {
        "macro_ece": float(np.mean(per_label)),
        "per_label_ece": per_label,
    }


# =========================================================
# 🔥 RELIABILITY DIAGRAM (NEW)
# =========================================================

def compute_reliability(y_true, probs, n_bins=10):

    try:
        diagram = ReliabilityDiagram(n_bins=n_bins)
        return diagram.compute(probs, y_true)
    except Exception as e:
        logger.warning(f"Reliability diagram failed: {e}")
        return {}


# =========================================================
# FULL PIPELINE (UPDATED)
# =========================================================

def compute_calibration(
    logits: Optional[np.ndarray],
    y_true: Iterable,
    task_type: str,
    *,
    apply_temp_scaling: bool = True,
    n_bins: int = 10,
) -> Dict[str, Any]:

    if logits is None:
        raise ValueError("logits required")

    y_true = np.asarray(y_true)

    # ---------------------------
    # TEMPERATURE SCALING
    # ---------------------------
    if apply_temp_scaling and task_type in ("multiclass", "binary"):
        T = fit_temperature(logits, y_true, task_type)
        logits = apply_temperature(logits, T)
    else:
        T = None

    # ---------------------------
    # PROBABILITIES
    # ---------------------------
    if task_type == "multiclass":
        probs = softmax(logits)

    elif task_type in ("binary", "multilabel"):
        probs = sigmoid(logits)

    else:
        raise ValueError(f"Invalid task_type: {task_type}")

    results: Dict[str, Any] = {}

    # ---------------------------
    # CONFIDENCE
    # ---------------------------
    results["confidence"] = extract_confidence(probs).tolist()

    # ---------------------------
    # ECE
    # ---------------------------
    if task_type == "multilabel":
        results.update(multilabel_ece(y_true, probs, n_bins))
    else:
        results["ece"] = expected_calibration_error(y_true, probs, n_bins)

        if probs.ndim == 2:
            results["classwise_ece"] = classwise_ece(y_true, probs, n_bins)

    # ---------------------------
    # RELIABILITY DIAGRAM
    # ---------------------------
    results["reliability_diagram"] = compute_reliability(y_true, probs, n_bins)

    # ---------------------------
    # BRIER SCORE
    # ---------------------------
    results["brier"] = brier_score(y_true, probs, task_type)

    # ---------------------------
    # TEMPERATURE
    # ---------------------------
    if T is not None:
        results["temperature"] = T

    return results