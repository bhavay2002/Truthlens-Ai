from __future__ import annotations

import logging
import gc
import os
import json
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold

# Multilabel stratification
try:
    from skmultilearn.model_selection import IterativeStratification
    HAS_ITERATIVE = True
except ImportError:
    HAS_ITERATIVE = False

from src.config.task_config import TASK_CONFIG, get_task_type, is_multilabel
from src.training.checkpointing import save_checkpoint_v2
from src.utils.settings import load_settings

logger = logging.getLogger(__name__)
SETTINGS = load_settings()


# =========================================================
#  METRIC RESOLUTION
# =========================================================

def _resolve_metric(task: str, metrics: Dict[str, Any]) -> float:
    task_type = get_task_type(task)

    if task_type == "multilabel":
        candidates = ["micro_f1", "eval_micro_f1"]
    elif task_type == "multiclass":
        candidates = ["accuracy", "eval_accuracy"]
    else:
        candidates = ["f1", "eval_f1"]

    candidates += ["eval_loss"]

    for key in candidates:
        if key in metrics:
            return float(metrics[key])

    raise KeyError(f"[{task}] metric not found")


# =========================================================
#  STRATIFICATION
# =========================================================

def _build_splits(df, task, label_column, n_splits, seed):

    if is_multilabel(task) and HAS_ITERATIVE:
        y = np.vstack(df[label_column].values)

        stratifier = IterativeStratification(
            n_splits=n_splits,
            order=1,
        )

        return list(stratifier.split(np.zeros(len(y)), y))

    # fallback
    y = df[label_column].values

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    return list(skf.split(df, y))


# =========================================================
#  EARLY STOPPING WRAPPER
# =========================================================

class EarlyStopper:
    def __init__(self, patience=2):
        self.best = -np.inf
        self.counter = 0
        self.patience = patience

    def step(self, score):
        if score > self.best:
            self.best = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


# =========================================================
#  SINGLE TASK CV
# =========================================================

def cross_validate_task(
    *,
    task: str,
    df: pd.DataFrame,
    train_function: Callable[..., Any],
    text_column: str = "text",
    label_column: str = "label",
    n_splits: int | None = None,
    random_state: int | None = None,
    checkpoint_root: str | None = None,
    ddp_rank: int | None = None,
) -> Dict[str, Any]:

    splits = n_splits or SETTINGS.training.cross_validation_splits
    seed = random_state or SETTINGS.training.seed

    splits_idx = _build_splits(df, task, label_column, splits, seed)

    fold_scores = []
    stopper = EarlyStopper(patience=2)

    for fold, (train_idx, val_idx) in enumerate(splits_idx, start=1):

        #  DDP: shard folds
        if ddp_rank is not None and fold % torch.distributed.get_world_size() != ddp_rank:
            continue

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        trainer = train_function(
            train_df=train_df,
            validation_df=val_df,
            task=task,
        )

        with torch.no_grad():
            metrics = trainer.evaluate(task=task)

        score = _resolve_metric(task, metrics)
        fold_scores.append(score)

        logger.info("[Task=%s] Fold %d score=%.4f", task, fold, score)

        # -----------------------------
        # CHECKPOINT
        # -----------------------------
        if checkpoint_root:
            save_checkpoint_v2(
                model=trainer.model,
                checkpoint_dir=os.path.join(
                    checkpoint_root,
                    f"{task}/fold_{fold}"
                ),
                epoch=trainer.state.epoch,
                step=trainer.state.global_step,
                task_schema=TASK_CONFIG,
            )

        # -----------------------------
        # EARLY STOPPING
        # -----------------------------
        if stopper.step(score):
            logger.info("[Task=%s] Early stopping triggered", task)
            break

        # cleanup
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

    return {
        "task": task,
        "fold_scores": fold_scores,
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
    }


# =========================================================
#  DASHBOARD
# =========================================================

def build_dashboard(results: Dict[str, Any], save_path=None):

    dashboard = {
        "tasks": {},
        "global": {}
    }

    all_scores = []

    for task, res in results.items():
        dashboard["tasks"][task] = {
            "mean": res["mean"],
            "std": res["std"],
        }
        all_scores.append(res["mean"])

    dashboard["global"]["mean"] = float(np.mean(all_scores))
    dashboard["global"]["std"] = float(np.std(all_scores))

    if save_path:
        with open(save_path, "w") as f:
            json.dump(dashboard, f, indent=2)

    return dashboard


# =========================================================
#  MULTI-TASK CV
# =========================================================

def cross_validate_all_tasks(
    *,
    datasets: Dict[str, pd.DataFrame],
    train_function: Callable[..., Any],
    checkpoint_root: str | None = None,
    ddp: bool = False,
) -> Dict[str, Any]:

    results = {}

    ddp_rank = None
    if ddp and torch.distributed.is_initialized():
        ddp_rank = torch.distributed.get_rank()

    for task, df in datasets.items():

        logger.info("==== Running CV for %s ====", task)

        results[task] = cross_validate_task(
            task=task,
            df=df,
            train_function=train_function,
            checkpoint_root=checkpoint_root,
            ddp_rank=ddp_rank,
        )

    return results