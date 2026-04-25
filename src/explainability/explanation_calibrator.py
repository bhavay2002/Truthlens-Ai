"""
File: explanation_calibrator.py
Module: Explainability Calibration
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_scores(scores: List[float]) -> np.ndarray:
    """
    L1 normalization (probability distribution).
    """

    arr = np.asarray(scores, dtype=float)

    if arr.size == 0:
        return arr

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    arr = np.abs(arr)
    total = float(np.sum(arr))

    if total <= 0:
        return np.zeros_like(arr)

    return arr / (total + EPS)


# =========================================================
# ENTROPY
# =========================================================

def compute_entropy(probs: np.ndarray) -> float:
    if probs.size == 0:
        return 0.0

    probs = np.clip(probs, EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_confidence(probs: np.ndarray) -> float:
    """
    Confidence derived from normalized entropy.
    """

    if probs.size == 0:
        return 0.0

    entropy = compute_entropy(probs)

    max_entropy = np.log(len(probs) + EPS)

    if max_entropy <= 0:
        return 0.0

    normalized_entropy = entropy / (max_entropy + EPS)

    return float(1.0 - normalized_entropy)


# =========================================================
# METHOD-AWARE NORMALIZATION
# =========================================================

def calibrate_by_method(
    scores: List[float],
    method: Optional[str],
) -> np.ndarray:

    # unified normalization for stability
    arr = normalize_scores(scores)

    # optional method-specific shaping
    if method == "lime":
        # slightly sharpen distribution
        arr = np.power(arr, 0.8)
        arr = normalize_scores(arr.tolist())

    elif method == "attention":
        # flatten distribution slightly
        arr = np.power(arr, 1.2)
        arr = normalize_scores(arr.tolist())

    return arr


# =========================================================
# MAIN CALIBRATION PIPELINE
# =========================================================

def calibrate_explanation(
    scores: List[float],
    method: Optional[str] = None,
) -> Dict[str, object]:
    """
    FINAL CONTRACT:

    Returns:
    {
        "scores": np.ndarray,
        "confidence": float,
        "entropy": float,
    }
    """

    if not scores:
        return {
            "scores": np.array([], dtype=float),
            "confidence": 0.0,
            "entropy": 0.0,
        }

    # -------------------------
    # normalize + method calibration
    # -------------------------
    calibrated = calibrate_by_method(scores, method)

    # -------------------------
    # entropy + confidence
    # -------------------------
    ent = compute_entropy(calibrated)
    conf = compute_confidence(calibrated)

    return {
        "scores": calibrated,          #  numpy array
        "confidence": conf,            #  float
        "entropy": ent,                # float
    }