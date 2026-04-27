from __future__ import annotations

import logging
import gc
import time
from typing import Callable, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold

from src.config.task_config import get_task_type
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


# =========================================================
# METRIC RESOLUTION
# =========================================================

def resolve_metric(
    task: str,
    metrics: Dict[str, Any],
    strategy: str = "auto",
    default: float = float("inf"),
) -> float:

    if strategy == "auto":
        task_type = get_task_type(task)

        if task_type == "multilabel":
            keys = ["micro_f1", "eval_micro_f1"]
        elif task_type == "multiclass":
            keys = ["accuracy", "eval_accuracy"]
        else:
            keys = ["f1", "eval_f1"]

        keys += ["val_loss", "eval_loss"]

    else:
        keys = [strategy]

    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except Exception:
                continue

    logger.warning("[%s] metric not found, using default=%s", task, default)
    return default


# =========================================================
# SPLITS (STRATIFIED + SAFE)
# =========================================================

def build_splits(
    df: pd.DataFrame,
    label_column: str,
    n_splits: int,
    seed: int,
):

    if label_column not in df:
        raise ValueError(f"Missing label column: {label_column}")

    y = df[label_column].values

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    return list(splitter.split(df, y))


# =========================================================
# MAIN CV (TRAINER-BASED)
# =========================================================

def cross_validate_task(
    *,
    task: str,
    df: pd.DataFrame,
    create_trainer_fn: Callable,
    params: Dict[str, Any],
    label_column: str = "label",
    n_splits: int = 5,
    seed: int = 42,
    metric_strategy: str = "auto",
    return_fold_details: bool = True,
) -> Dict[str, Any]:

    set_seed(seed)

    splits = build_splits(df, label_column, n_splits, seed)

    fold_results: List[Dict[str, Any]] = []
    scores: List[float] = []

    for fold_id, (train_idx, val_idx) in enumerate(splits, start=1):

        logger.info("CV | task=%s | fold=%d/%d", task, fold_id, n_splits)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        start = time.time()

        try:
            # -------------------------
            # CREATE TRAINER
            # -------------------------
            trainer = create_trainer_fn(
                task=task,
                train_df=train_df,
                val_df=val_df,
                params=params,
            )

            # -------------------------
            # TRAIN
            # -------------------------
            trainer.train()

            # -------------------------
            # EVALUATE
            # -------------------------
            with torch.no_grad():
                metrics = trainer.evaluate()

            score = resolve_metric(task, metrics, metric_strategy)

            duration = time.time() - start

            scores.append(score)

            fold_result = {
                "fold": fold_id,
                "score": score,
                "metrics": metrics,
                "time": duration,
            }

            if return_fold_details:
                fold_results.append(fold_result)

            logger.info(
                "[%s] Fold %d | score=%.4f | time=%.2fs",
                task,
                fold_id,
                score,
                duration,
            )

        except Exception:
            logger.exception("Fold %d failed", fold_id)

        finally:
            # 🔥 MEMORY SAFETY
            try:
                del trainer
            except Exception:
                pass

            torch.cuda.empty_cache()
            gc.collect()

    # -----------------------------------------------------
    # POST-CHECK
    # -----------------------------------------------------

    if not scores:
        raise RuntimeError("All CV folds failed")

    scores_np = np.array(scores, dtype=float)

    return {
        "task": task,
        "folds": fold_results if return_fold_details else None,
        "mean": float(scores_np.mean()),
        "std": float(scores_np.std()),
        "min": float(scores_np.min()),
        "max": float(scores_np.max()),
        "num_successful_folds": len(scores),
        "num_failed_folds": n_splits - len(scores),
    }


# =========================================================
# MULTI-TASK CV
# =========================================================

def cross_validate_all_tasks(
    *,
    datasets: Dict[str, pd.DataFrame],
    create_trainer_fn: Callable,
    params: Dict[str, Any],
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:

    results: Dict[str, Any] = {}

    for task, df in datasets.items():

        logger.info("==== CV: %s ====", task)

        results[task] = cross_validate_task(
            task=task,
            df=df,
            create_trainer_fn=create_trainer_fn,
            params=params,
            n_splits=n_splits,
            seed=seed,
        )

    return results


# =========================================================
# DASHBOARD
# =========================================================

def build_dashboard(results: Dict[str, Any]) -> Dict[str, Any]:

    global_scores = [
        r["mean"]
        for r in results.values()
        if r.get("mean") is not None
    ]

    return {
        "tasks": {
            task: {
                "mean": res.get("mean"),
                "std": res.get("std"),
                "folds": res.get("num_successful_folds"),
            }
            for task, res in results.items()
        },
        "global": {
            "mean": float(np.mean(global_scores)) if global_scores else None,
            "std": float(np.std(global_scores)) if global_scores else None,
        },
    }