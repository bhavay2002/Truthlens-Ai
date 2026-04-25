from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Literal, Optional

import numpy as np
import pandas as pd

from src.config.task_config import TASK_CONFIG

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# NORMALIZATION
# =========================================================

def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / (df.std(ddof=0) + EPS)


# =========================================================
# 🔥 ROBUST CLIPPING (NEW)
# =========================================================

def _winsorize(df: pd.DataFrame, lower=0.01, upper=0.99):
    return df.clip(
        lower=df.quantile(lower),
        upper=df.quantile(upper),
        axis=1,
    )


# =========================================================
# TASK FEATURE EXTRACTION
# =========================================================

def _extract_task_features(predictions: Dict[str, Any]) -> pd.DataFrame:

    features = {}

    for task, values in predictions.items():

        arr = np.asarray(values)
        task_type = TASK_CONFIG[task]["type"]

        if task_type == "binary":
            features[task] = arr.reshape(-1)

        elif task_type == "multiclass":

            if arr.ndim == 1:
                features[task] = arr
            else:
                for i in range(arr.shape[1]):
                    features[f"{task}_class_{i}"] = arr[:, i]

        elif task_type == "multilabel":
            for i in range(arr.shape[1]):
                features[f"{task}_label_{i}"] = arr[:, i]

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    return pd.DataFrame(features)


# =========================================================
# 🔥 MAIN CORRELATION (UPGRADED)
# =========================================================

def compute_task_correlation(
    predictions: Dict[str, Any] | pd.DataFrame,
    *,
    normalize: bool = True,
    method: Literal["pearson", "spearman"] = "spearman",
    robust: bool = True,  # 🔥 NEW
    confidence: Optional[np.ndarray] = None,
    uncertainty: Optional[np.ndarray] = None,
    graph_signal: Optional[np.ndarray] = None,
) -> pd.DataFrame:

    logger.info(f"[CORRELATION] computing (method={method})")

    if isinstance(predictions, dict):
        df = _extract_task_features(predictions)
    else:
        df = predictions.copy()

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 2:
        raise ValueError("Insufficient data")

    # -------------------------
    # 🔥 ROBUST HANDLING
    # -------------------------
    if robust:
        df = _winsorize(df)

    # -------------------------
    # NORMALIZATION
    # -------------------------
    if normalize:
        df = _normalize(df)

    # -------------------------
    # 🔥 ADD AUX SIGNALS
    # -------------------------
    if confidence is not None:
        df["global_confidence"] = confidence

    if uncertainty is not None:
        df["global_uncertainty"] = uncertainty

    if graph_signal is not None:
        df["graph_signal"] = graph_signal

    # -------------------------
    # CORRELATION
    # -------------------------
    corr = df.corr(method=method)

    # -------------------------
    # STABILITY
    # -------------------------
    corr = corr.replace([np.inf, -np.inf], 0.0)
    corr = corr.fillna(0.0)

    return corr


# =========================================================
# 🔥 AGGREGATION (UPGRADED)
# =========================================================

def aggregate_task_correlation(corr: pd.DataFrame) -> pd.DataFrame:

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

            vals = np.asarray(vals)

            # 🔥 confidence weighting via variance
            weight = np.var(vals) + EPS

            agg.loc[t1, t2] = float(np.mean(vals) * weight)

    agg = agg.astype(float)

    # diagonal stability
    for t in agg.index:
        agg.loc[t, t] = 1.0

    return agg


# =========================================================
# 🔥 MONITORING SIGNALS (NEW)
# =========================================================

def correlation_statistics(corr: pd.DataFrame) -> Dict[str, float]:

    values = corr.values.flatten()

    return {
        "mean_correlation": float(np.mean(values)),
        "std_correlation": float(np.std(values)),
        "max_correlation": float(np.max(values)),
        "min_correlation": float(np.min(values)),
        "high_correlation_ratio": float(np.mean(np.abs(values) > 0.8)),
    }


# =========================================================
# SAVE
# =========================================================

def save_correlation_matrix(
    corr: pd.DataFrame,
    path: str | Path,
) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    corr.to_csv(path)

    logger.info(f"Saved correlation matrix: {path}")

    return path