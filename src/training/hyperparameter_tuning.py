from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

import optuna
import pandas as pd

from src.training.cross_validation import cross_validate_task
from src.utils.seed_utils import set_seed
from src.config.task_config import get_task_type

logger = logging.getLogger(__name__)


# =========================================================
# TRACKING
# =========================================================

def init_tracking(task: str) -> Dict[str, Any]:

    tracking: Dict[str, Any] = {
        "mlflow": None,
        "wandb": None,
    }

    # MLflow
    try:
        import mlflow

        mlflow.set_experiment(f"TruthLens_{task}")
        mlflow.start_run(run_name=f"{task}_{int(time.time())}")
        tracking["mlflow"] = mlflow
    except Exception as e:
        logger.warning("MLflow init failed: %s", e)

    # W&B
    try:
        import wandb

        wandb.init(project="TruthLens", name=task)
        tracking["wandb"] = wandb
    except Exception as e:
        logger.warning("W&B init failed: %s", e)

    return tracking


def finalize_tracking(tracking: Dict[str, Any]):

    if tracking.get("mlflow"):
        try:
            tracking["mlflow"].end_run()
        except Exception:
            pass

    if tracking.get("wandb"):
        try:
            tracking["wandb"].finish()
        except Exception:
            pass


def log_trial(
    tracking: Dict[str, Any],
    trial_id: int,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:

    payload = {
        "trial": trial_id,
        **params,
        **metrics,
    }

    # MLflow
    if tracking.get("mlflow"):
        for k, v in payload.items():
            try:
                tracking["mlflow"].log_metric(k, float(v))
            except Exception:
                continue

    # W&B
    if tracking.get("wandb"):
        try:
            tracking["wandb"].log(payload)
        except Exception:
            pass


# =========================================================
# OBJECTIVE
# =========================================================

def build_objective(
    *,
    task: str,
    df: pd.DataFrame,
    create_trainer_fn: Callable,
    multi_objective: bool,
    tracking: Dict[str, Any],
):

    def objective(trial: optuna.Trial):

        # -------------------------
        # SEED
        # -------------------------
        seed = 42 + trial.number
        set_seed(seed)

        # -------------------------
        # SEARCH SPACE
        # -------------------------
        params = {
            "lr": trial.suggest_float("lr", 1e-6, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs": trial.suggest_int("epochs", 2, 6),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        }

        start = time.time()

        try:
            # -------------------------
            # CROSS VALIDATION
            # -------------------------
            cv_result = cross_validate_task(
                task=task,
                df=df,
                create_trainer_fn=create_trainer_fn,
                params=params,
            )

            score = cv_result["mean"]
            std = cv_result["std"]

        except Exception:
            logger.exception("Trial %d failed", trial.number)
            raise optuna.TrialPruned()

        duration = time.time() - start

        # -------------------------
        # REPORT FOR PRUNING
        # -------------------------
        trial.report(score, step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        # -------------------------
        # LOGGING
        # -------------------------
        log_trial(
            tracking,
            trial.number,
            params,
            {
                "score": score,
                "std": std,
                "time": duration,
            },
        )

        return (score, -std) if multi_objective else score

    return objective


# =========================================================
# STUDY
# =========================================================

def _resolve_direction(task: str) -> str:
    # BUG-8: cross_validate_task returns the model's primary metric
    # (accuracy / micro-F1 for classification, MSE for regression).
    # Classification metrics must be MAXIMISED — Optuna's previous
    # default of "minimize" silently selected the worst trials.
    try:
        ttype = str(get_task_type(task)).replace("_", "").lower()
    except Exception:
        ttype = ""

    if ttype in {"multiclass", "multilabel", "binary"}:
        return "maximize"
    return "minimize"  # regression / loss-style metrics


def create_study(
    *,
    multi_objective: bool,
    storage: Optional[str],
    task: str,
):

    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
    )

    pruner = optuna.pruners.MedianPruner()

    score_direction = _resolve_direction(task)

    if multi_objective:
        # objective returns (score, -std) — both should be MAXIMISED
        # when score is a classification metric, otherwise both MINIMISED.
        return optuna.create_study(
            directions=[score_direction, score_direction],
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

    return optuna.create_study(
        direction=score_direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )


# =========================================================
# MAIN
# =========================================================

def tune_task(
    *,
    task: str,
    df: pd.DataFrame,
    create_trainer_fn: Callable,
    n_trials: int,
    multi_objective: bool = False,
    n_jobs: int = 1,
    storage: Optional[str] = None,
):

    tracking = init_tracking(task)

    study = create_study(
        multi_objective=multi_objective,
        storage=storage,
        task=task,
    )

    objective = build_objective(
        task=task,
        df=df,
        create_trainer_fn=create_trainer_fn,
        multi_objective=multi_objective,
        tracking=tracking,
    )

    logger.info(
        "Starting tuning | task=%s | trials=%d",
        task,
        n_trials,
    )

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
        )
    finally:
        finalize_tracking(tracking)

    # -------------------------
    # RESULTS
    # -------------------------

    if not multi_objective:
        return {
            "task": task,
            "best_params": study.best_params,
            "best_score": float(study.best_value),
        }

    return {
        "task": task,
        "pareto_front": [
            {"params": t.params, "values": t.values}
            for t in study.best_trials
        ],
    }


# =========================================================
# MULTI-TASK
# =========================================================

def tune_all_tasks(
    *,
    datasets: Dict[str, pd.DataFrame],
    create_trainer_fn: Callable,
    n_trials: int,
    multi_objective: bool = False,
    n_jobs: int = 1,
    storage: Optional[str] = None,
):

    results: Dict[str, Any] = {}

    for task, df in datasets.items():

        logger.info("🚀 Tuning task: %s", task)

        results[task] = tune_task(
            task=task,
            df=df,
            create_trainer_fn=create_trainer_fn,
            n_trials=n_trials,
            multi_objective=multi_objective,
            n_jobs=n_jobs,
            storage=storage,
        )

    return results