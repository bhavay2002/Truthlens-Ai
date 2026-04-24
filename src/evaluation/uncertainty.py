"""
File: uncertainty.py (FINAL - RESEARCH + INDUSTRY GRADE)
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# VALIDATION
# =========================================================
def _validate_probs(probs: Iterable) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)

    if probs.ndim < 2:
        raise ValueError("probs must be at least 2D")

    if probs.shape[0] == 0:
        raise ValueError("empty probs")

    return probs


# =========================================================
# ENTROPY
# =========================================================
def predictive_entropy(probs: Iterable) -> np.ndarray:
    probs = _validate_probs(probs)
    return -np.sum(probs * np.log(probs + EPS), axis=-1)


def normalized_entropy(probs: Iterable) -> np.ndarray:
    probs = _validate_probs(probs)

    n_classes = probs.shape[-1]
    entropy = predictive_entropy(probs)

    return entropy / (np.log(n_classes) + EPS)


# =========================================================
# CONFIDENCE
# =========================================================
def confidence_scores(probs: Iterable) -> np.ndarray:
    probs = _validate_probs(probs)
    return np.max(probs, axis=-1)


# =========================================================
# MULTILABEL UNCERTAINTY
# =========================================================
def multilabel_uncertainty(probs: Iterable) -> Dict[str, np.ndarray]:
    probs = _validate_probs(probs)

    entropy = -(
        probs * np.log(probs + EPS) +
        (1 - probs) * np.log(1 - probs + EPS)
    )

    return {
        "label_entropy": entropy,
        "mean_entropy": np.mean(entropy, axis=1),
    }


# =========================================================
# VARIANCE (MC DROPOUT / ENSEMBLE)
# =========================================================
def predictive_variance(prob_samples: Iterable) -> np.ndarray:
    """
    prob_samples: (T, N, C)
    """
    prob_samples = np.asarray(prob_samples)

    if prob_samples.ndim != 3:
        raise ValueError("Expected shape (T, N, C)")

    return np.var(prob_samples, axis=0).mean(axis=1)


# =========================================================
# MUTUAL INFORMATION (NEW 🔥)
# =========================================================
def mutual_information(prob_samples: Iterable) -> np.ndarray:
    """
    prob_samples: (T, N, C)

    MI = H(mean_probs) - mean(H(probs))
    """
    prob_samples = np.asarray(prob_samples)

    if prob_samples.ndim != 3:
        raise ValueError("Expected shape (T, N, C)")

    mean_probs = np.mean(prob_samples, axis=0)

    entropy_mean = -np.sum(mean_probs * np.log(mean_probs + EPS), axis=1)
    entropy_expected = -np.mean(
        np.sum(prob_samples * np.log(prob_samples + EPS), axis=2),
        axis=0,
    )

    return entropy_mean - entropy_expected


# =========================================================
# ENERGY-BASED UNCERTAINTY (NEW 🔥)
# =========================================================
def energy_score(logits: Iterable) -> np.ndarray:
    """
    logits: (N, C)

    Energy = -log(sum(exp(logits)))
    Lower = more confident
    """
    logits = np.asarray(logits)

    if logits.ndim != 2:
        raise ValueError("logits must be (N, C)")

    # log-sum-exp trick
    max_logits = np.max(logits, axis=1, keepdims=True)
    stabilized = logits - max_logits

    logsumexp = np.log(np.sum(np.exp(stabilized), axis=1)) + max_logits.squeeze()

    return -logsumexp


# =========================================================
# MAIN STATS
# =========================================================
def uncertainty_statistics(
    probs: Iterable,
    *,
    logits: Optional[Iterable] = None,
    prob_samples: Optional[Iterable] = None,
    multilabel: bool = False,
) -> Dict[str, float]:

    probs = _validate_probs(probs)

    # ---------------------------
    # BASE
    # ---------------------------
    if multilabel:
        ml = multilabel_uncertainty(probs)
        entropy = ml["mean_entropy"]
        confidence = np.mean(probs, axis=1)
    else:
        entropy = predictive_entropy(probs)
        confidence = confidence_scores(probs)

    stats = {
        "mean_entropy": float(np.mean(entropy)),
        "std_entropy": float(np.std(entropy)),
        "mean_confidence": float(np.mean(confidence)),
        "std_confidence": float(np.std(confidence)),
    }

    # ---------------------------
    # ENERGY (NEW)
    # ---------------------------
    if logits is not None:
        energy = energy_score(logits)
        stats["mean_energy"] = float(np.mean(energy))
        stats["std_energy"] = float(np.std(energy))

    # ---------------------------
    # MUTUAL INFORMATION (NEW)
    # ---------------------------
    if prob_samples is not None:
        mi = mutual_information(prob_samples)
        stats["mean_mutual_information"] = float(np.mean(mi))
        stats["std_mutual_information"] = float(np.std(mi))

    return stats


# =========================================================
# SAMPLE-LEVEL OUTPUT
# =========================================================
def uncertainty_per_sample(
    probs: Iterable,
    *,
    logits: Optional[Iterable] = None,
    prob_samples: Optional[Iterable] = None,
    multilabel: bool = False,
) -> Dict[str, np.ndarray]:

    probs = _validate_probs(probs)

    if multilabel:
        ml = multilabel_uncertainty(probs)
        entropy = ml["mean_entropy"]
    else:
        entropy = predictive_entropy(probs)

    confidence = confidence_scores(probs)

    result = {
        "entropy": entropy,
        "confidence": confidence,
    }

    if logits is not None:
        result["energy"] = energy_score(logits)

    if prob_samples is not None:
        result["mutual_information"] = mutual_information(prob_samples)

    return result