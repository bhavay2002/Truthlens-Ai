"""
File Name: score_normalizer.py
Module: TruthLens AI - Aggregation Score Normalizer
Description:
    Provides normalization utilities for signals produced by different
    TruthLens subsystems before aggregation.

    Since analysis modules may output values on different scales
    (e.g., emotion intensity 0–5, graph centrality 0–100, bias score 0–1),
    this module standardizes them using multiple normalization techniques.

    Supported normalization methods:
        • Min-Max normalization
        • Z-score normalization
        • Robust scaling (median / IQR)
        • Score clipping 

Dependencies:
    logging
    typing
    numpy

Inputs:
    numeric score vectors or iterables

Outputs:
    normalized numpy arrays
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np


logger = logging.getLogger(__name__)


EPS = 1e-12


def _to_array(values: Iterable[float]) -> np.ndarray:
    """Convert input to numpy array with validation."""

    try:
        arr = np.asarray(list(values), dtype=np.float32)
    except Exception as exc:
        raise TypeError("values must be numeric iterable") from exc

    if arr.size == 0:
        raise ValueError("values cannot be empty")

    if not np.isfinite(arr).all():
        raise ValueError("values contain NaN or infinite values")

    return arr


def normalize_minmax(
    values: Iterable[float],
    *,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Apply Min-Max normalization.

    x' = (x - min) / (max - min)
    """

    arr = _to_array(values)

    if (
        not isinstance(feature_range, tuple)
        or len(feature_range) != 2
        or not all(isinstance(x, (int, float)) for x in feature_range)
    ):
        raise TypeError("feature_range must be a tuple of two numeric values")

    a, b = float(feature_range[0]), float(feature_range[1])
    if not np.isfinite([a, b]).all():
        raise ValueError("feature_range values must be finite")
    if not a < b:
        raise ValueError("feature_range must satisfy a < b")

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))

    if abs(vmax - vmin) < EPS:
        logger.warning("Min-max normalization encountered constant values")
        midpoint = (a + b) / 2.0
        return np.full_like(arr, fill_value=midpoint, dtype=np.float32)

    norm = (arr - vmin) / (vmax - vmin)
    norm = norm * (b - a) + a

    return norm.astype(np.float32)


def normalize_zscore(values: Iterable[float]) -> np.ndarray:
    """
    Apply Z-score normalization.

    z = (x - μ) / σ
    """

    arr = _to_array(values)

    mean = float(np.mean(arr))
    std = float(np.std(arr))

    if std < EPS:
        logger.warning("Z-score normalization encountered zero std")
        return np.zeros_like(arr)

    norm = (arr - mean) / std

    return norm.astype(np.float32)


def normalize_robust(values: Iterable[float]) -> np.ndarray:
    """
    Apply robust scaling using median and IQR.

    x' = (x - median) / IQR
    """

    arr = _to_array(values)

    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))

    iqr = q3 - q1

    if abs(iqr) < EPS:
        logger.warning("Robust scaling encountered zero IQR")
        return np.zeros_like(arr)

    norm = (arr - median) / iqr

    return norm.astype(np.float32)


def clip_scores(
    values: Iterable[float],
    *,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
    """
    Clip values to specified range.
    """

    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")

    arr = _to_array(values)

    clipped = np.clip(arr, min_value, max_value)

    return clipped.astype(np.float32)


def normalize_pipeline(
    values: Iterable[float],
    *,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalize values using selected method.

    Supported methods:
        minmax
        zscore
        robust
    """

    if not isinstance(method, str):
        raise TypeError("method must be a string")

    method = method.lower()

    if method == "minmax":
        return normalize_minmax(values)

    if method == "zscore":
        return normalize_zscore(values)

    if method == "robust":
        return normalize_robust(values)

    raise ValueError(
        f"Unsupported normalization method: {method}"
    )