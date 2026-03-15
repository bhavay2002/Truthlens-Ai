"""
File: src/training/hyperparameter_tuning.py

Purpose
-------
Automated hyperparameter tuning for TruthLens AI models.

Supports:
• Optuna-based Bayesian optimization
• Random-search fallback tuner
• Flexible training function interfaces

Outputs
-------
Best hyperparameters and evaluation metric score.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.train_roberta import train_model
from src.utils.input_validation import (
    ensure_dataframe,
    ensure_non_empty_text_column,
    ensure_positive_int,
)
from src.utils.settings import load_settings

logger = logging.getLogger(__name__)
SETTINGS = load_settings()

_VALID_OPTIMIZATION_DIRECTIONS = {"minimize", "maximize"}


def _resolve_metric(
    metrics: dict[str, Any],
    metric_name: str,
) -> float:
    if not isinstance(metrics, dict):
        raise TypeError(
            "trainer.evaluate(...) must return a dictionary, "
            f"received: {type(metrics).__name__}."
        )

    candidates = [
        metric_name,
        f"eval_{metric_name}",
        "eval_loss",
        "loss",
    ]

    for key in candidates:
        if key in metrics:
            return float(metrics[key])

    raise KeyError(
        f"Unable to resolve metric '{metric_name}' "
        f"from keys: {sorted(metrics.keys())}"
    )


def _build_train_kwargs(
    train_function: Callable[..., tuple[Any, Any]],
    *,
    params: dict[str, Any],
    text_column: str,
    validation_df: pd.DataFrame,
) -> dict[str, Any]:
    train_sig = inspect.signature(train_function)

    kwargs: dict[str, Any] = {}

    if "params" in train_sig.parameters:
        kwargs["params"] = params
    if "text_column" in train_sig.parameters:
        kwargs["text_column"] = text_column
    if "validation_df" in train_sig.parameters:
        kwargs["validation_df"] = validation_df
    if "test_df" in train_sig.parameters:
        kwargs["test_df"] = validation_df

    return kwargs


def _evaluate_params(
    params: dict[str, Any],
    *,
    train_function: Callable[..., tuple[Any, Any]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_column: str,
    metric_name: str,
) -> float:
    train_kwargs = _build_train_kwargs(
        train_function,
        params=params,
        text_column=text_column,
        validation_df=val_df,
    )

    train_result = train_function(train_df, **train_kwargs)
    if not isinstance(train_result, tuple) or len(train_result) != 2:
        raise TypeError("train_function must return (trainer, eval_dataset).")

    trainer, eval_dataset = train_result
    metrics = trainer.evaluate(eval_dataset)

    return _resolve_metric(metrics, metric_name)


def _sample_params_fallback(
    rng: np.random.Generator,
) -> dict[str, Any]:
    lr_min = np.log10(SETTINGS.training.optuna_learning_rate_min)
    lr_max = np.log10(SETTINGS.training.optuna_learning_rate_max)

    learning_rate = float(10 ** rng.uniform(lr_min, lr_max))
    batch_size = int(rng.choice(SETTINGS.training.optuna_batch_sizes))
    epochs = int(rng.choice(SETTINGS.training.optuna_epoch_choices))

    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
    }


def _run_fallback_tuner(
    *,
    train_function: Callable[..., tuple[Any, Any]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_column: str,
    n_trials: int,
    metric_name: str,
    direction: str,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    best_params: dict[str, Any] | None = None
    best_value: float | None = None

    for trial_idx in range(1, n_trials + 1):
        params = _sample_params_fallback(rng)

        value = _evaluate_params(
            params,
            train_function=train_function,
            train_df=train_df,
            val_df=val_df,
            text_column=text_column,
            metric_name=metric_name,
        )

        logger.info(
            "Fallback trial %s/%s | %s=%.4f | params=%s",
            trial_idx,
            n_trials,
            metric_name,
            value,
            params,
        )

        is_better = (
            best_value is None
            or (direction == "minimize" and value < best_value)
            or (direction == "maximize" and value > best_value)
        )

        if is_better:
            best_value = value
            best_params = params

    if best_value is None or best_params is None:
        raise RuntimeError(
            "Fallback tuner failed to produce any trial results."
        )

    return {
        "best_params": best_params,
        "best_value": float(best_value),
        "metric_name": metric_name,
        "direction": direction,
        "trials": n_trials,
        "backend": "fallback",
    }


def run_optuna(
    df: pd.DataFrame,
    *,
    train_function: Callable[..., tuple[Any, Any]] = train_model,
    validation_df: pd.DataFrame | None = None,
    text_column: str = "text",
    n_trials: int | None = None,
    metric_name: str | None = None,
    direction: str | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Run Optuna-based hyperparameter optimization (with fallback)."""

    ensure_dataframe(
        df,
        name="df",
        required_columns=[text_column, "label"],
        min_rows=10,
    )
    ensure_non_empty_text_column(
        df,
        text_column,
        name="df",
    )

    effective_trials = (
        n_trials if n_trials is not None else SETTINGS.training.optuna_trials
    )
    ensure_positive_int(
        effective_trials,
        name="n_trials",
        min_value=1,
    )

    effective_metric = metric_name or SETTINGS.training.optuna_metric
    effective_direction = direction or SETTINGS.training.optuna_direction
    if effective_direction not in _VALID_OPTIMIZATION_DIRECTIONS:
        raise ValueError(
            "direction must be either 'minimize' or 'maximize', "
            f"received: {effective_direction!r}."
        )

    effective_seed = (
        SETTINGS.training.seed if random_state is None else random_state
    )

    if validation_df is None:
        train_df, val_df = train_test_split(
            df,
            test_size=SETTINGS.training.optuna_validation_split,
            random_state=effective_seed,
            stratify=df["label"],
        )
    else:
        ensure_dataframe(
            validation_df,
            name="validation_df",
            required_columns=[text_column, "label"],
            min_rows=2,
        )
        ensure_non_empty_text_column(
            validation_df,
            text_column,
            name="validation_df",
        )
        train_df = df
        val_df = validation_df

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    try:
        import optuna
    except ImportError:
        logger.warning("Optuna not installed, using fallback tuner")
        return _run_fallback_tuner(
            train_function=train_function,
            train_df=train_df,
            val_df=val_df,
            text_column=text_column,
            n_trials=effective_trials,
            metric_name=effective_metric,
            direction=effective_direction,
            seed=effective_seed,
        )

    sampler = optuna.samplers.TPESampler(seed=effective_seed)

    study = optuna.create_study(
        direction=effective_direction,
        sampler=sampler,
    )

    def objective(trial) -> float:
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                SETTINGS.training.optuna_learning_rate_min,
                SETTINGS.training.optuna_learning_rate_max,
                log=True,
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size",
                list(SETTINGS.training.optuna_batch_sizes),
            ),
            "epochs": trial.suggest_categorical(
                "epochs",
                list(SETTINGS.training.optuna_epoch_choices),
            ),
        }

        score = _evaluate_params(
            params,
            train_function=train_function,
            train_df=train_df,
            val_df=val_df,
            text_column=text_column,
            metric_name=effective_metric,
        )

        logger.info(
            "Optuna trial %s | %s=%.4f | params=%s",
            trial.number,
            effective_metric,
            score,
            params,
        )

        return score

    study.optimize(
        objective,
        n_trials=effective_trials,
    )

    logger.info(
        "Optuna best score %.4f | params=%s",
        study.best_value,
        study.best_params,
    )

    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "metric_name": effective_metric,
        "direction": effective_direction,
        "trials": effective_trials,
        "backend": "optuna",
    }
