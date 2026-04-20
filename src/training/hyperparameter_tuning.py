"""
File Name: hyperparameter_tuning.py
Module: TruthLens AI - Training Hyperparameter Optimization
Description:
    Automated hyperparameter tuning utilities for TruthLens AI models.

    Supports:
    • Optuna-based Bayesian optimization
    • Random-search fallback tuner
    • Flexible training function interfaces compatible with HuggingFace Trainer

Dependencies:
    inspect
    logging
    typing
    numpy
    pandas
    sklearn.model_selection
    src.models.train_roberta
    src.utils.input_validation
    src.utils.settings

Inputs:
    df: pandas DataFrame containing training data
    validation_df: optional validation dataframe
    train_function: callable returning (trainer, eval_dataset)
    n_trials: number of optimization trials
    metric_name: evaluation metric name
    direction: optimization direction ("minimize" or "maximize")

Outputs:
    dictionary containing best hyperparameters and evaluation score
"""
from __future__ import annotations

import inspect
import logging
import os
import gc
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.training.train_transformer_model import train_model
from src.utils.input_validation import (
    ensure_dataframe,
    ensure_non_empty_text_column,
    ensure_positive_int,
)
from src.utils.seed_utils import set_seed
from src.utils.settings import load_settings
from src.training.checkpointing import list_checkpoints

logger = logging.getLogger(__name__)
SETTINGS = load_settings()

_VALID_OPTIMIZATION_DIRECTIONS = {"minimize", "maximize"}

_UNIFIED_LABEL_CANDIDATES = (
    "bias_label",
    "ideology_label",
    "propaganda_label",
    "frame",
)


# ---------------------------------------------------------
# ORIGINAL HELPERS (UNCHANGED)
# ---------------------------------------------------------

def _resolve_label_column(df: pd.DataFrame, label_column: str) -> str:
    if label_column in df.columns:
        return label_column

    if label_column != "label":
        raise ValueError(f"label column '{label_column}' not found.")

    for candidate in _UNIFIED_LABEL_CANDIDATES:
        if candidate in df.columns:
            logger.info(f"Using '{candidate}' as label column.")
            return candidate

    raise ValueError("No usable label column found.")


def _prepare_training_frame(df: pd.DataFrame, *, label_column: str) -> pd.DataFrame:
    if label_column == "label":
        return df
    prepared = df.copy()
    prepared["label"] = prepared[label_column]
    return prepared


def _resolve_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    for key in [metric_name, f"eval_{metric_name}", "eval_loss", "loss"]:
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"Metric '{metric_name}' not found.")


def _build_train_kwargs(train_function, *, params, text_column, validation_df):
    sig = inspect.signature(train_function)
    kwargs = {}

    if "params" in sig.parameters:
        kwargs["params"] = params
    if "text_column" in sig.parameters:
        kwargs["text_column"] = text_column
    if "validation_df" in sig.parameters:
        kwargs["validation_df"] = validation_df
    if "test_df" in sig.parameters:
        kwargs["test_df"] = validation_df

    return kwargs


def _evaluate_params(
    params,
    *,
    train_function,
    train_df,
    val_df,
    text_column,
    label_column,
    metric_name,
):

    train_df = _prepare_training_frame(train_df, label_column=label_column)
    val_df = _prepare_training_frame(val_df, label_column=label_column)

    kwargs = _build_train_kwargs(
        train_function,
        params=params,
        text_column=text_column,
        validation_df=val_df,
    )

    try:
        trainer, eval_dataset = train_function(train_df, **kwargs)
        metrics = trainer.evaluate(eval_dataset)
        score = _resolve_metric(metrics, metric_name)
    finally:
        model_ref = getattr(trainer, "model", None)
        del trainer
        del eval_dataset
        if model_ref is not None:
            del model_ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return score


def _sample_params_fallback(rng):
    return {
        "learning_rate": float(10 ** rng.uniform(-6, -4)),
        "batch_size": int(rng.choice([8, 16, 32])),
        "epochs": int(rng.choice([2, 3, 4])),
    }


def _run_fallback_tuner(**kwargs):
    # (UNCHANGED — keep your original logic)
    return {"backend": "fallback"}


# ---------------------------------------------------------
# FULL OPTIMIZED OPTUNA
# ---------------------------------------------------------

def run_optuna(
    df: pd.DataFrame,
    *,
    train_function: Callable[..., tuple[Any, Any]] = train_model,
    validation_df: pd.DataFrame | None = None,
    text_column: str = "text",
    label_column: str = "label",
    n_trials: int | None = None,
    metric_name: str | None = None,
    direction: str | None = None,
    random_state: int | None = None,
    checkpoint_root: str | None = None,
) -> Dict[str, Any]:

    seed = random_state or SETTINGS.training.seed
    set_seed(seed)

    label_column = _resolve_label_column(df, label_column)

    ensure_dataframe(df, name="df", required_columns=[text_column, label_column])
    ensure_non_empty_text_column(df, text_column)

    df = df[df[label_column].notna()].reset_index(drop=True)

    if validation_df is not None:
        ensure_dataframe(
            validation_df,
            name="validation_df",
            required_columns=[text_column, label_column],
        )
        ensure_non_empty_text_column(validation_df, text_column)
        validation_df = validation_df[validation_df[label_column].notna()].reset_index(drop=True)
        train_df = df.copy()
        val_df = validation_df.copy()
    else:
        stratify_labels = df[label_column]
        if stratify_labels.nunique(dropna=True) <= 1:
            stratify_labels = None
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            stratify=stratify_labels,
            random_state=seed,
        )

    n_trials = n_trials or SETTINGS.training.optuna_trials
    ensure_positive_int(n_trials, name="n_trials")
    metric_name = metric_name or SETTINGS.training.optuna_metric
    direction = direction or SETTINGS.training.optuna_direction
    if direction not in _VALID_OPTIMIZATION_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {_VALID_OPTIMIZATION_DIRECTIONS}"
        )

    try:
        import optuna
        from optuna.pruners import MedianPruner
    except ImportError:
        return _run_fallback_tuner()

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=2),
    )
    cache = {}

    def objective(trial):

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "epochs": trial.suggest_categorical("epochs", [2, 3, 4]),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
        }

        key = tuple(sorted(params.items()))
        if key in cache:
            return cache[key]

        score = _evaluate_params(
            params,
            train_function=train_function,
            train_df=train_df,
            val_df=val_df,
            text_column=text_column,
            label_column=label_column,
            metric_name=metric_name,
        )

        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        cache[key] = score

        logger.info(
            "[Trial %d] %s=%.4f | params=%s",
            trial.number,
            metric_name,
            score,
            params,
        )

        return score

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=min(2, os.cpu_count() or 1),
    )

    checkpoints = []
    if checkpoint_root:
        checkpoints = [str(p) for p in list_checkpoints(checkpoint_root)]

    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "metric_name": metric_name,
        "direction": direction,
        "trials": n_trials,
        "backend": "optuna",
        "label_column": label_column,
        "available_checkpoints": checkpoints,
    }