#File: evaluate_model.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, List

import numpy as np
import torch
from transformers import AutoTokenizer

from src.utils.device_utils import move_batch, autocast_context
from src.utils.metrics_utils import compute_task_metrics
from src.config.task_config import TASK_CONFIG


# =========================================================
# ACTIVATIONS
# =========================================================

def _softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================================================
# TOKENIZATION
# =========================================================

def _tokenize(tokenizer, texts: List[str], max_length=512):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# =========================================================
# MODEL PREDICT (FIXED)
# =========================================================

def _predict_model(
    model,
    texts: List[str],
    task: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    outputs = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            encoded = _tokenize(tokenizer, batch_texts)
            encoded = move_batch(encoded, device)

            with autocast_context():
                out = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    task=task,
                )

            logits = out["logits"].detach().cpu().numpy()
            outputs.append(logits)

    return np.vstack(outputs)


# =========================================================
# POSTPROCESS (TASK-AWARE)
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
# CORE EVALUATION
# =========================================================

def evaluate(
    y_true: Iterable,
    y_pred: Optional[Iterable] = None,
    y_proba: Optional[Iterable] = None,
    *,
    model=None,
    texts: Optional[List[str]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:

    # =====================================================
    # VALIDATE TASK
    # =====================================================
    if task is not None:
        if task not in TASK_CONFIG:
            raise ValueError(f"Unknown task: {task}")

        task_type = TASK_CONFIG[task]["type"]
        num_labels = TASK_CONFIG[task]["num_labels"]
    else:
        raise ValueError("task must be provided")

    # =====================================================
    # MODEL MODE
    # =====================================================
    if model is not None:
        if texts is None or tokenizer is None:
            raise ValueError("model mode requires texts + tokenizer")

        logits = _predict_model(
            model=model,
            texts=texts,
            task=task,
            tokenizer=tokenizer,
        )

        y_pred, y_proba = _postprocess_logits(logits, task_type)

    # =====================================================
    # NUMPY MODE
    # =====================================================
    y_true_arr = np.asarray(y_true)

    if y_true_arr.size == 0:
        raise ValueError("y_true cannot be empty")

    is_multilabel = y_true_arr.ndim == 2

    y_pred_arr = np.asarray(y_pred)
    y_proba_arr = np.asarray(y_proba) if y_proba is not None else None

    # =====================================================
    # SHAPE VALIDATION
    # =====================================================
    if not is_multilabel:
        if y_pred_arr.ndim != 1:
            raise ValueError("y_pred must be 1D")

        if y_true_arr.shape != y_pred_arr.shape:
            raise ValueError("Shape mismatch")

    else:
        if y_pred_arr.shape != y_true_arr.shape:
            raise ValueError("Multilabel mismatch")

    # =====================================================
    # METRICS (TASK-AWARE)
    # =====================================================
    metrics = compute_task_metrics(
        logits=torch.tensor(y_proba_arr if y_proba_arr is not None else y_pred_arr),
        labels=torch.tensor(y_true_arr),
        task_type=task_type,
        num_labels=num_labels,
    )

    # =====================================================
    # DATASET STATS
    # =====================================================
    if not is_multilabel:
        labels, counts = np.unique(y_true_arr, return_counts=True)

        dataset_stats = {
            "num_samples": int(len(y_true_arr)),
            "num_classes": int(len(labels)),
            "class_distribution": {
                str(l): int(c) for l, c in zip(labels, counts)
            },
        }
    else:
        dataset_stats = {
            "num_samples": int(y_true_arr.shape[0]),
            "num_labels": int(y_true_arr.shape[1]),
            "label_density": float(np.mean(y_true_arr)),
        }

    return {
        "task": task,
        "task_type": task_type,
        "metrics": metrics,
        "dataset_stats": dataset_stats,
    }