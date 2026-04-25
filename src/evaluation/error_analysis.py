from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)


# =========================================================
# CORE HELPERS
# =========================================================

def _to_numpy(x):
    return np.asarray(x)


def _top_k_indices(arr, k=10, largest=True):
    if largest:
        return np.argsort(-arr)[:k]
    return np.argsort(arr)[:k]


# =========================================================
# BINARY ERROR ANALYSIS
# =========================================================

def analyze_binary_errors(
    y_true,
    y_pred,
    probs: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    results = {
        "false_positives": int(len(fp_idx)),
        "false_negatives": int(len(fn_idx)),
    }

    # -----------------------------------------------------
    # HARD EXAMPLES (confidence-based)
    # -----------------------------------------------------
    if probs is not None:
        probs = _to_numpy(probs)

        # confident wrong predictions
        fp_conf = probs[fp_idx]
        fn_conf = probs[fn_idx]

        fp_hard = fp_idx[_top_k_indices(fp_conf, k=top_k)]
        fn_hard = fn_idx[_top_k_indices(1 - fn_conf, k=top_k)]

        results["top_false_positives"] = _build_samples(fp_hard, texts, probs)
        results["top_false_negatives"] = _build_samples(fn_hard, texts, probs)

    return results


# =========================================================
# MULTICLASS ERROR ANALYSIS
# =========================================================

def analyze_multiclass_errors(
    y_true,
    y_pred,
    probs: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    incorrect = np.where(y_true != y_pred)[0]

    results = {
        "total_errors": int(len(incorrect)),
    }

    # -----------------------------------------------------
    # CONFUSION PAIRS
    # -----------------------------------------------------
    pairs = list(zip(y_true[incorrect], y_pred[incorrect]))
    pair_counts = pd.Series(pairs).value_counts().to_dict()

    results["confusion_pairs"] = {
        f"{k[0]}→{k[1]}": int(v) for k, v in pair_counts.items()
    }

    # -----------------------------------------------------
    # HARD EXAMPLES
    # -----------------------------------------------------
    if probs is not None:
        probs = _to_numpy(probs)

        confidence = np.max(probs, axis=1)
        wrong_conf = confidence[incorrect]

        hard_idx = incorrect[_top_k_indices(wrong_conf, k=top_k)]

        results["hard_examples"] = _build_samples(hard_idx, texts, confidence)

    return results


# =========================================================
# MULTILABEL ERROR ANALYSIS
# =========================================================

def analyze_multilabel_errors(
    y_true,
    y_pred,
    probs: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    errors = (y_true != y_pred).astype(int)

    per_label_errors = errors.sum(axis=0)

    results = {
        "per_label_error_count": per_label_errors.tolist(),
        "total_error_labels": int(errors.sum()),
    }

    # -----------------------------------------------------
    # HARD SAMPLES
    # -----------------------------------------------------
    sample_errors = errors.sum(axis=1)

    hard_idx = _top_k_indices(sample_errors, k=top_k)

    results["hard_samples"] = _build_samples(hard_idx, texts, sample_errors)

    return results


# =========================================================
# SAMPLE BUILDER
# =========================================================

def _build_samples(indices, texts, scores):

    samples = []

    for idx in indices:
        sample = {
            "index": int(idx),
            "score": float(scores[idx]) if scores is not None else None,
        }

        if texts is not None:
            sample["text"] = texts[idx]

        samples.append(sample)

    return samples


# =========================================================
# MAIN API
# =========================================================

def error_analysis(
    y_true,
    y_pred,
    *,
    probs: Optional[np.ndarray] = None,
    texts: Optional[List[str]] = None,
    task: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:

    task_type = get_task_type(task) if task else None

    logger.info(f"[ERROR ANALYSIS] Task={task} Type={task_type}")

    if task_type == "binary":
        return analyze_binary_errors(
            y_true, y_pred, probs, texts, top_k
        )

    elif task_type == "multiclass":
        return analyze_multiclass_errors(
            y_true, y_pred, probs, texts, top_k
        )

    elif task_type == "multilabel":
        return analyze_multilabel_errors(
            y_true, y_pred, probs, texts, top_k
        )

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")