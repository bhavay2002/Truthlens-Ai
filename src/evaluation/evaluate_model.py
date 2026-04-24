#File: evaluate_model.py (REFRACTORED - MULTI-TASK READY)

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from .metrics import compute_classification_metrics


# =========================================================
# UTIL
# =========================================================
def _to_serializable_label(label: Any) -> Any:
    if isinstance(label, np.generic):
        return label.item()
    return label


def _softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================================================
# POSTPROCESS
# =========================================================
def _postprocess_logits(logits, task_type):
    if task_type == "multiclass":
        probs = _softmax(logits)
        preds = np.argmax(probs, axis=1)

    elif task_type == "binary":
        probs = _sigmoid(logits).reshape(-1)
        preds = (probs > 0.5).astype(int)

    elif task_type == "multilabel":
        probs = _sigmoid(logits)
        preds = (probs > 0.5).astype(int)

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return preds, probs


# =========================================================
# MODEL INFERENCE
# =========================================================
def _predict_model(
    model,
    X,
    task: str,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    outputs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size]).to(device)
            out = model.predict(batch, task=task)
            logits = out["logits"].detach().cpu().numpy()
            outputs.append(logits)

    return np.vstack(outputs)


# =========================================================
# CORE EVALUATE
# =========================================================
def evaluate(
    y_true: Iterable,
    y_pred: Optional[Iterable] = None,
    y_proba: Optional[Iterable] = None,
    *,
    model=None,
    X: Optional[np.ndarray] = None,
    task: Optional[str] = None,
    task_type: str = "multiclass",
    from_logits: bool = False,
) -> Dict[str, Any]:
    """
    Multi-mode evaluation:

    Modes:
    1. Legacy:
        evaluate(y_true, y_pred)

    2. With probabilities:
        evaluate(y_true, y_pred, y_proba)

    3. Model-based:
        evaluate(y_true, model=..., X=..., task="bias")
    """

    # =====================================================
    # MODEL MODE
    # =====================================================
    if model is not None:
        if X is None or task is None:
            raise ValueError("model mode requires X and task")

        logits = _predict_model(model, X, task)

        y_pred, y_proba = _postprocess_logits(logits, task_type)

    # =====================================================
    # STANDARD MODE
    # =====================================================
    y_true_arr = np.asarray(y_true)

    if y_true_arr.size == 0:
        raise ValueError("y_true cannot be empty")

    # MULTILABEL SUPPORT
    is_multilabel = y_true_arr.ndim == 2

    if not is_multilabel:
        y_pred_arr = np.asarray(y_pred)

        if y_pred_arr.ndim != 1:
            raise ValueError("y_pred must be 1D for non-multilabel")

        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError("Shape mismatch")

    else:
        y_pred_arr = np.asarray(y_pred)

        if y_pred_arr.shape != y_true_arr.shape:
            raise ValueError("Multilabel shapes must match")

    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
    else:
        y_proba_arr = None

    # =====================================================
    # METRICS
    # =====================================================
    metrics = compute_classification_metrics(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        y_proba=y_proba_arr,
        task_type=task_type,
    )

    # =====================================================
    # DATASET STATS
    # =====================================================
    if not is_multilabel:
        labels, counts = np.unique(y_true_arr, return_counts=True)

        dataset_stats = {
            "num_samples": int(y_true_arr.shape[0]),
            "num_classes": int(labels.size),
            "class_counts": {
                str(_to_serializable_label(label)): int(count)
                for label, count in zip(labels, counts)
            },
        }
    else:
        dataset_stats = {
            "num_samples": int(y_true_arr.shape[0]),
            "num_labels": int(y_true_arr.shape[1]),
            "label_density": float(np.mean(y_true_arr)),
        }

    metrics["task"] = task
    metrics["task_type"] = task_type
    metrics["dataset_stats"] = dataset_stats

    return metrics