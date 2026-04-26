from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class ClassBalanceConfig:
    imbalance_threshold: float = 0.2   # min class ratio
    compute_weights: bool = True
    normalize_weights: bool = True


# =========================================================
# RESULT
# =========================================================

@dataclass
class ClassBalanceReport:
    task: str
    type: str  # classification | multilabel

    distribution: Dict[str, Any]
    imbalance_detected: bool

    weights: Optional[Dict[Any, float]] = None


# =========================================================
# CLASSIFICATION
# =========================================================

def analyze_classification(
    df: pd.DataFrame,
    label_col: str,
    *,
    config: Optional[ClassBalanceConfig] = None,
) -> ClassBalanceReport:

    config = config or ClassBalanceConfig()

    counts = df[label_col].value_counts().sort_index()
    total = counts.sum()

    dist = (counts / total).to_dict()

    min_ratio = min(dist.values())
    imbalance = min_ratio < config.imbalance_threshold

    weights = None

    if config.compute_weights:
        weights = _compute_class_weights(counts, normalize=config.normalize_weights)

    logger.info(
        "Class balance | %s | dist=%s | imbalance=%s",
        label_col,
        dist,
        imbalance,
    )

    return ClassBalanceReport(
        task=label_col,
        type="classification",
        distribution=dist,
        imbalance_detected=imbalance,
        weights=weights,
    )


# =========================================================
# MULTILABEL
# =========================================================

def analyze_multilabel(
    df: pd.DataFrame,
    label_cols: List[str],
    *,
    config: Optional[ClassBalanceConfig] = None,
) -> ClassBalanceReport:

    config = config or ClassBalanceConfig()

    dist = {}
    weights = {}

    imbalance = False

    for col in label_cols:

        pos = df[col].sum()
        total = len(df)

        ratio = float(pos) / max(total, 1)

        dist[col] = ratio

        if ratio < config.imbalance_threshold:
            imbalance = True

        if config.compute_weights:
            weights[col] = _compute_binary_weight(pos, total)

    logger.info(
        "Multilabel balance | cols=%d | imbalance=%s",
        len(label_cols),
        imbalance,
    )

    return ClassBalanceReport(
        task="multilabel",
        type="multilabel",
        distribution=dist,
        imbalance_detected=imbalance,
        weights=weights if config.compute_weights else None,
    )


# =========================================================
# TASK WRAPPER (YOUR SYSTEM)
# =========================================================

def analyze_task_balance(
    df: pd.DataFrame,
    task: str,
    *,
    config: Optional[ClassBalanceConfig] = None,
) -> ClassBalanceReport:

    if task == "bias":
        return analyze_classification(df, "bias", config=config)

    elif task == "ideology":
        return analyze_classification(df, "ideology", config=config)

    elif task == "propaganda":
        return analyze_classification(df, "propaganda", config=config)

    elif task == "frame":
        return analyze_multilabel(df, ["CO", "EC", "HI", "MO", "RE"], config=config)

    elif task == "narrative":
        return analyze_multilabel(df, ["hero", "villain", "victim"], config=config)

    elif task == "emotion":
        return analyze_multilabel(
            df,
            [f"emotion_{i}" for i in range(20)],
            config=config,
        )

    else:
        raise ValueError(f"Unknown task: {task}")


# =========================================================
# WEIGHT UTILS
# =========================================================

def _compute_class_weights(counts, normalize=True):

    total = counts.sum()
    weights = {cls: total / c for cls, c in counts.items()}

    if normalize:
        s = sum(weights.values())
        weights = {k: v / s for k, v in weights.items()}

    return weights


def _compute_binary_weight(pos, total):

    neg = total - pos

    if pos == 0:
        return 1.0

    return float(neg) / float(pos)