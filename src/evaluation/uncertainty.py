from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import numpy as np

from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# VALIDATION
# =========================================================

def _ensure_2d(probs: np.ndarray) -> np.ndarray:
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    return probs


def _validate_probs(probs: Iterable) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)

    if probs.shape[0] == 0:
        raise ValueError("empty probs")

    return _ensure_2d(probs)


# =========================================================
# ENTROPY
# =========================================================

def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(probs + EPS), axis=-1)


def normalized_entropy(probs: np.ndarray) -> np.ndarray:
    n_classes = probs.shape[-1]
    return predictive_entropy(probs) / (np.log(n_classes) + EPS)


# =========================================================
#  CONFIDENCE
# =========================================================

def confidence_scores(probs: np.ndarray) -> np.ndarray:
    return np.max(probs, axis=-1)


def margin_confidence(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]


# =========================================================
#  CONFIDENCE-WEIGHTED ENTROPY (NEW)
# =========================================================

def confidence_weighted_entropy(probs: np.ndarray) -> np.ndarray:
    entropy = predictive_entropy(probs)
    confidence = confidence_scores(probs)
    return entropy * (1.0 - confidence)


# =========================================================
# MULTILABEL
# =========================================================

def multilabel_uncertainty(probs: np.ndarray):

    entropy = -(
        probs * np.log(probs + EPS) +
        (1 - probs) * np.log(1 - probs + EPS)
    )

    return {
        "label_entropy": entropy,
        "mean_entropy": np.mean(entropy, axis=1),
        "confidence": np.max(probs, axis=1),
    }


# =========================================================
# VARIANCE
# =========================================================

def predictive_variance(prob_samples: Iterable) -> np.ndarray:
    prob_samples = np.asarray(prob_samples)

    if prob_samples.ndim != 3:
        raise ValueError("Expected (T, N, C)")

    return np.var(prob_samples, axis=0).mean(axis=1)


# =========================================================
# MUTUAL INFORMATION
# =========================================================

def mutual_information(prob_samples: Iterable) -> np.ndarray:
    prob_samples = np.asarray(prob_samples)

    mean_probs = np.mean(prob_samples, axis=0)

    entropy_mean = predictive_entropy(mean_probs)
    entropy_expected = np.mean(predictive_entropy(prob_samples), axis=0)

    mi = entropy_mean - entropy_expected
    return mi / (np.max(mi) + EPS)


# =========================================================
# ENERGY
# =========================================================

def energy_score(logits: Iterable) -> np.ndarray:
    logits = np.asarray(logits)

    max_logits = np.max(logits, axis=1, keepdims=True)
    stabilized = logits - max_logits

    logsumexp = np.log(np.sum(np.exp(stabilized), axis=1)) + max_logits.squeeze()
    energy = -logsumexp

    return (energy - np.mean(energy)) / (np.std(energy) + EPS)


# =========================================================
#  DRIFT SIGNAL (NEW)
# =========================================================

def uncertainty_drift(entropy: np.ndarray) -> Dict[str, float]:
    return {
        "entropy_shift": float(np.mean(entropy)),
        "entropy_spread": float(np.std(entropy)),
        "high_uncertainty_ratio": float(np.mean(entropy > 0.8)),
    }


# =========================================================
#  MAIN STATS (UPGRADED)
# =========================================================

def uncertainty_statistics(
    probs: Iterable,
    *,
    task: Optional[str] = None,
    logits: Optional[Iterable] = None,
    prob_samples: Optional[Iterable] = None,
    explanation_scores: Optional[Iterable] = None,
) -> Dict[str, float]:

    probs = _validate_probs(probs)
    task_type = get_task_type(task) if task else None

    # ---------------------------
    # BASE
    # ---------------------------
    if task_type == "multilabel":
        ml = multilabel_uncertainty(probs)
        entropy = ml["mean_entropy"]
        confidence = ml["confidence"]
    else:
        entropy = predictive_entropy(probs)
        confidence = confidence_scores(probs)

    weighted_entropy = confidence_weighted_entropy(probs)

    stats = {
        "mean_entropy": float(np.mean(entropy)),
        "std_entropy": float(np.std(entropy)),
        "p95_entropy": float(np.percentile(entropy, 95)),
        "p99_entropy": float(np.percentile(entropy, 99)),

        "mean_confidence": float(np.mean(confidence)),
        "std_confidence": float(np.std(confidence)),

        #  NEW
        "mean_weighted_entropy": float(np.mean(weighted_entropy)),
    }

    # ---------------------------
    # MARGIN
    # ---------------------------
    if probs.shape[1] > 1:
        margin = margin_confidence(probs)
        stats["mean_margin"] = float(np.mean(margin))

    # ---------------------------
    # ENERGY
    # ---------------------------
    if logits is not None:
        energy = energy_score(logits)
        stats["mean_energy"] = float(np.mean(energy))
        stats["std_energy"] = float(np.std(energy))

    # ---------------------------
    # MI
    # ---------------------------
    if prob_samples is not None:
        mi = mutual_information(prob_samples)
        stats["mean_mutual_information"] = float(np.mean(mi))

    # ---------------------------
    #  DRIFT SIGNAL
    # ---------------------------
    stats.update(uncertainty_drift(entropy))

    # ---------------------------
    #  EXPLAINABILITY-AWARE
    # ---------------------------
    if explanation_scores is not None:
        explanation_scores = np.asarray(explanation_scores)
        stats["uncertainty_explanation_corr"] = float(
            np.corrcoef(entropy, explanation_scores)[0, 1]
        )

    return stats


# =========================================================
# PER-SAMPLE
# =========================================================

def uncertainty_per_sample(
    probs: Iterable,
    *,
    task: Optional[str] = None,
    logits: Optional[Iterable] = None,
    prob_samples: Optional[Iterable] = None,
) -> Dict[str, np.ndarray]:

    probs = _validate_probs(probs)
    task_type = get_task_type(task) if task else None

    if task_type == "multilabel":
        entropy = multilabel_uncertainty(probs)["mean_entropy"]
    else:
        entropy = predictive_entropy(probs)

    confidence = confidence_scores(probs)

    result = {
        "entropy": entropy,
        "confidence": confidence,
        "weighted_entropy": confidence_weighted_entropy(probs),  # 🔥 NEW
    }

    if probs.shape[1] > 1:
        result["margin"] = margin_confidence(probs)

    if logits is not None:
        result["energy"] = energy_score(logits)

    if prob_samples is not None:
        result["mutual_information"] = mutual_information(prob_samples)

    return result