"""
File: calibration.py
"""

from __future__ import annotations

import logging
from typing import Iterable, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.calibration import CalibrationMetricConfig, CalibrationMetrics

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
# TEMPERATURE SCALING (NEW 🔥)
# =========================================================
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(logits, labels, max_iter=50):
    logits = torch.tensor(logits, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)

    model = TemperatureScaler()

    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=max_iter)

    def loss_fn():
        optimizer.zero_grad()
        scaled_logits = model(logits)
        loss = nn.CrossEntropyLoss()(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(loss_fn)

    T = model.temperature.item()

    logger.info(f"[CALIBRATION] Learned temperature: {T:.4f}")

    return T


def apply_temperature(logits, T):
    return logits / T


# =========================================================
# ACTIVATIONS
# =========================================================
def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================================================
# CLASS-WISE ECE (NEW 🔥)
# =========================================================
def classwise_ece(y_true, probs, n_bins=10):
    y, p = _validate_inputs(y_true, probs)

    num_classes = p.shape[1]
    results = {}

    for c in range(num_classes):
        y_binary = (y == c).astype(int)
        p_class = p[:, c]

        ece = expected_calibration_error(y_binary, p_class, n_bins)
        results[f"class_{c}"] = ece

    return results


# =========================================================
# MAIN ECE
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
# FULL CALIBRATION PIPELINE (UPGRADED 🔥)
# =========================================================
def compute_calibration(
    logits: Optional[np.ndarray],
    y_true: Iterable,
    task_type: str,
    *,
    apply_temp_scaling: bool = True,
    n_bins: int = 10,
) -> Dict[str, Any]:

    y_true = np.asarray(y_true)

    if logits is None:
        raise ValueError("logits required for calibration pipeline")

    # ---------------------------
    # TEMPERATURE SCALING
    # ---------------------------
    if apply_temp_scaling and task_type != "multilabel":
        T = fit_temperature(logits, y_true)
        logits = apply_temperature(logits, T)
    else:
        T = None

    # ---------------------------
    # PROBS
    # ---------------------------
    if task_type == "multiclass":
        probs = softmax(logits)
    elif task_type == "binary":
        probs = sigmoid(logits)
    else:
        probs = sigmoid(logits)

    results = {}

    # ---------------------------
    # GLOBAL ECE
    # ---------------------------
    if task_type == "multilabel":
        results["ece"] = float(
            np.mean([
                expected_calibration_error(y_true[:, i], probs[:, i], n_bins)
                for i in range(probs.shape[1])
            ])
        )
    else:
        results["ece"] = expected_calibration_error(y_true, probs, n_bins)

    # ---------------------------
    # CLASS-WISE ECE
    # ---------------------------
    if probs.ndim == 2:
        results["classwise_ece"] = classwise_ece(y_true, probs, n_bins)

    # ---------------------------
    # TEMPERATURE
    # ---------------------------
    if T is not None:
        results["temperature"] = float(T)

    return results