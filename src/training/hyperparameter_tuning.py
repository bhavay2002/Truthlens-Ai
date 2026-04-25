from __future__ import annotations

import logging
import os
import json
import time
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch

from src.config.task_config import get_task_type
from src.training.cross_validation import cross_validate_task
from src.utils.settings import load_settings
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)
SETTINGS = load_settings()


# =========================================================
#  EXPERIMENT TRACKING (MLflow / W&B)
# =========================================================

def _init_tracking(task: str):
    tracking = {}

    try:
        import mlflow
        mlflow.set_experiment(f"TruthLens_{task}")
        mlflow.start_run(run_name=f"{task}_{int(time.time())}")
        tracking["mlflow"] = mlflow
    except Exception:
        tracking["mlflow"] = None

    try:
        import wandb
        wandb.init(project="TruthLens", name=task)
        tracking["wandb"] = wandb
    except Exception:
        tracking["wandb"] = None

    return tracking


def _log_tracking(tracking, params, metrics):
    if tracking["mlflow"]:
        for k, v in params.items():
            tracking["mlflow"].log_param(k, v)
        for k, v in metrics.items():
            tracking["mlflow"].log_metric(k, v)

    if tracking["wandb"]:
        tracking["wandb"].log({**params, **metrics})


# =========================================================
#  MULTI-OBJECTIVE SUPPORT
# =========================================================

def _build_objective(
    *,
    task: str,
    df: pd.DataFrame,
    train_function: Callable,
    checkpoint_root: str | None,
    multi_objective: bool = False,
):

    def objective(trial):

        params = {
            "learning_rate": trial.suggest_float("lr", 1e-6, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs": trial.suggest_int("epochs", 2, 5),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        }

        def wrapped_train_fn(**kwargs):
            return train_function(**kwargs, params=params)

        cv_result = cross_validate_task(
            task=task,
            df=df,
            train_function=wrapped_train_fn,
            checkpoint_root=checkpoint_root,
        )

        score = cv_result["mean"]
        std = cv_result["std"]

        # multi-objective: maximize score, minimize variance
        if multi_objective:
            return score, -std

        return score

    return objective


# =========================================================
#  PARALLEL STUDY (MULTI-GPU SAFE)
# =========================================================

def _run_parallel_study(
    study,
    objective,
    n_trials,
    n_jobs,
):
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,  #  parallel trials
    )


# =========================================================
#  SINGLE TASK TUNING
# =========================================================

def tune_task_v3(
    *,
    task: str,
    df: pd.DataFrame,
    train_function: Callable,
    n_trials: int,
    checkpoint_root: str | None = None,
    multi_objective: bool = False,
    n_jobs: int = 1,
):

    import optuna

    set_seed(SETTINGS.training.seed)

    tracking = _init_tracking(task)

    if multi_objective:
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.TPESampler(),
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
        )

    objective = _build_objective(
        task=task,
        df=df,
        train_function=train_function,
        checkpoint_root=checkpoint_root,
        multi_objective=multi_objective,
    )

    _run_parallel_study(
        study,
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    # -----------------------------
    # LOG BEST RESULT
    # -----------------------------
    if not multi_objective:
        best_params = study.best_params
        best_score = study.best_value

        _log_tracking(
            tracking,
            best_params,
            {"best_score": best_score},
        )

        return {
            "task": task,
            "best_params": best_params,
            "best_score": float(best_score),
        }

    else:
        pareto = [
            {
                "params": t.params,
                "values": t.values,
            }
            for t in study.best_trials
        ]

        return {
            "task": task,
            "pareto_front": pareto,
        }


# =========================================================
#  DASHBOARD / REPORT
# =========================================================

def generate_report(results: Dict[str, Any], save_path: str):

    leaderboard = []

    for task, res in results.items():
        if "best_score" in res:
            leaderboard.append((task, res["best_score"]))

    leaderboard.sort(key=lambda x: x[1], reverse=True)

    report = {
        "leaderboard": leaderboard,
        "tasks": results,
    }

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# =========================================================
#  MULTI-TASK TUNING
# =========================================================

def tune_all_tasks_v3(
    *,
    datasets: Dict[str, pd.DataFrame],
    train_function: Callable,
    n_trials: int,
    checkpoint_root: str | None = None,
    multi_objective: bool = False,
    n_jobs: int = 1,
):

    results = {}

    for task, df in datasets.items():

        logger.info("🚀 Tuning task: %s", task)

        results[task] = tune_task_v3(
            task=task,
            df=df,
            train_function=train_function,
            n_trials=n_trials,
            checkpoint_root=checkpoint_root,
            multi_objective=multi_objective,
            n_jobs=n_jobs,
        )

    return results