"""Group-fairness metrics for evaluation reports.

Provides simple, dependency-free implementations of the most common fairness
metrics so the evaluator can surface disparate-impact signals when sensitive
attributes are available in the dataset.

All functions accept array-like inputs and return floats / dictionaries that
serialize cleanly into JSON reports.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# =========================================================
# UTILITIES
# =========================================================

def _validate(y_true, y_pred, groups) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true).reshape(-1)
    p = np.asarray(y_pred).reshape(-1)
    g = np.asarray(groups).reshape(-1)

    if not (y.shape == p.shape == g.shape):
        raise ValueError(
            f"Shape mismatch: y_true {y.shape}, y_pred {p.shape}, groups {g.shape}"
        )
    if y.size == 0:
        raise ValueError("inputs cannot be empty")
    return y, p, g


# =========================================================
# PER-GROUP METRICS
# =========================================================

def per_group_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
) -> Dict[str, Dict[str, float]]:
    y, p, g = _validate(y_true, y_pred, groups)
    unique_groups = np.unique(g)

    out: Dict[str, Dict[str, float]] = {}
    for group in unique_groups:
        mask = g == group
        if not mask.any():
            continue

        y_g = y[mask]
        p_g = p[mask]

        out[str(group)] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(y_g, p_g)),
            "precision": float(
                precision_score(y_g, p_g, average="binary", pos_label=positive_label, zero_division=0)
                if set(np.unique(y_g)).issubset({0, 1})
                else precision_score(y_g, p_g, average="macro", zero_division=0)
            ),
            "recall": float(
                recall_score(y_g, p_g, average="binary", pos_label=positive_label, zero_division=0)
                if set(np.unique(y_g)).issubset({0, 1})
                else recall_score(y_g, p_g, average="macro", zero_division=0)
            ),
            "f1": float(
                f1_score(y_g, p_g, average="binary", pos_label=positive_label, zero_division=0)
                if set(np.unique(y_g)).issubset({0, 1})
                else f1_score(y_g, p_g, average="macro", zero_division=0)
            ),
            "positive_rate": float(np.mean(p_g == positive_label)),
        }

    return out


# =========================================================
# DEMOGRAPHIC PARITY
# =========================================================

def demographic_parity(
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
) -> Dict[str, float]:
    p = np.asarray(y_pred).reshape(-1)
    g = np.asarray(groups).reshape(-1)
    if p.shape != g.shape:
        raise ValueError("y_pred and groups must have the same shape")

    unique_groups = np.unique(g)
    rates = {
        str(group): float(np.mean(p[g == group] == positive_label))
        for group in unique_groups
        if (g == group).any()
    }

    if not rates:
        return {"max_diff": 0.0, "ratio": 1.0, "rates": {}}

    values = list(rates.values())
    max_diff = float(max(values) - min(values))
    ratio = float(min(values) / max(values)) if max(values) > 0 else 1.0

    return {"rates": rates, "max_diff": max_diff, "ratio": ratio}


# =========================================================
# EQUAL OPPORTUNITY (TPR PARITY) + EQUALIZED ODDS
# =========================================================

def _per_group_rates(y, p, g, positive_label) -> Dict[str, Dict[str, float]]:
    rates: Dict[str, Dict[str, float]] = {}
    for group in np.unique(g):
        mask = g == group
        if not mask.any():
            continue

        y_g = y[mask]
        p_g = p[mask]
        pos = y_g == positive_label
        neg = ~pos

        tpr = float(np.mean(p_g[pos] == positive_label)) if pos.any() else float("nan")
        fpr = float(np.mean(p_g[neg] == positive_label)) if neg.any() else float("nan")
        rates[str(group)] = {"tpr": tpr, "fpr": fpr}
    return rates


def equal_opportunity(
    y_true: Iterable,
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
) -> Dict[str, float]:
    y, p, g = _validate(y_true, y_pred, groups)
    rates = _per_group_rates(y, p, g, positive_label)

    valid = [r["tpr"] for r in rates.values() if not np.isnan(r["tpr"])]
    if not valid:
        return {"per_group_tpr": rates, "max_diff": 0.0}

    return {"per_group_tpr": rates, "max_diff": float(max(valid) - min(valid))}


def equalized_odds(
    y_true: Iterable,
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
) -> Dict[str, Any]:
    y, p, g = _validate(y_true, y_pred, groups)
    rates = _per_group_rates(y, p, g, positive_label)

    tprs = [r["tpr"] for r in rates.values() if not np.isnan(r["tpr"])]
    fprs = [r["fpr"] for r in rates.values() if not np.isnan(r["fpr"])]

    return {
        "per_group": rates,
        "tpr_max_diff": float(max(tprs) - min(tprs)) if tprs else 0.0,
        "fpr_max_diff": float(max(fprs) - min(fprs)) if fprs else 0.0,
    }


# =========================================================
# TOP-LEVEL ENTRY
# =========================================================

def fairness_report(
    y_true: Iterable,
    y_pred: Iterable,
    groups: Iterable,
    *,
    positive_label: int = 1,
    group_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute a full fairness report for a single sensitive attribute."""
    try:
        y, p, g = _validate(y_true, y_pred, groups)
    except ValueError as exc:
        logger.warning("fairness_report aborted: %s", exc)
        return {"error": str(exc)}

    return {
        "attribute": group_name,
        "per_group_metrics": per_group_metrics(y, p, g, positive_label=positive_label),
        "demographic_parity": demographic_parity(p, g, positive_label=positive_label),
        "equal_opportunity": equal_opportunity(y, p, g, positive_label=positive_label),
        "equalized_odds": equalized_odds(y, p, g, positive_label=positive_label),
    }


def fairness_report_multi(
    y_true: Iterable,
    y_pred: Iterable,
    sensitive_attributes: Dict[str, Iterable],
    *,
    positive_label: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """Run :func:`fairness_report` for each sensitive attribute provided."""
    return {
        name: fairness_report(
            y_true=y_true,
            y_pred=y_pred,
            groups=values,
            positive_label=positive_label,
            group_name=name,
        )
        for name, values in sensitive_attributes.items()
    }


__all__ = [
    "demographic_parity",
    "equal_opportunity",
    "equalized_odds",
    "fairness_report",
    "fairness_report_multi",
    "per_group_metrics",
]
