"""
File: task_correlation.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# VALIDATION
# =========================================================
def _to_dataframe(predictions: Dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(predictions, pd.DataFrame):
        df = predictions.copy()
    elif isinstance(predictions, dict):
        df = pd.DataFrame(predictions)
    else:
        raise TypeError("predictions must be dict or DataFrame")

    if df.empty:
        raise ValueError("Empty predictions")

    if df.shape[1] < 2:
        raise ValueError("Need at least 2 tasks")

    return df


# =========================================================
# PROBABILITY FLATTENING
# =========================================================
def _flatten_predictions(predictions: Dict[str, Any]) -> pd.DataFrame:
    flat = {}

    for task, values in predictions.items():
        arr = np.asarray(values)

        if arr.ndim == 1:
            flat[task] = arr

        elif arr.ndim == 2:
            # multilabel or probabilities
            for i in range(arr.shape[1]):
                flat[f"{task}_{i}"] = arr[:, i]

        else:
            raise ValueError(f"Unsupported shape for {task}")

    return pd.DataFrame(flat)


# =========================================================
# MAIN CORRELATION
# =========================================================
def compute_task_correlation(
    predictions: Dict[str, Any] | pd.DataFrame,
    *,
    use_probabilities: bool = True,
    method: Literal["pearson", "spearman"] = "pearson",
) -> pd.DataFrame:

    logger.info("Computing advanced task correlation")

    if isinstance(predictions, dict) and use_probabilities:
        df = _flatten_predictions(predictions)
    else:
        df = _to_dataframe(predictions)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError("Insufficient numeric data")

    corr = df.corr(method=method)

    return corr


# =========================================================
# TASK-LEVEL AGGREGATION
# =========================================================
def aggregate_task_correlation(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate label-level correlations into task-level.
    """

    task_map = {}

    for col in corr.columns:
        task = col.split("_")[0]
        task_map.setdefault(task, []).append(col)

    agg = pd.DataFrame(index=task_map.keys(), columns=task_map.keys())

    for t1, cols1 in task_map.items():
        for t2, cols2 in task_map.items():

            vals = []
            for c1 in cols1:
                for c2 in cols2:
                    vals.append(corr.loc[c1, c2])

            agg.loc[t1, t2] = np.mean(vals)

    return agg.astype(float)


# =========================================================
# SAVE
# =========================================================
def save_correlation_matrix(
    corr: pd.DataFrame,
    path: str | Path
) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    corr.to_csv(path)

    logger.info("Saved correlation matrix: %s", path)

    return path