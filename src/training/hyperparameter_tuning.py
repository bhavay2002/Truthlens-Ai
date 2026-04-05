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
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.training.train_transformer_model import train_model
from src.utils.input_validation import (
    ensure_dataframe,
    ensure_non_empty_text_column,
    ensure_positive_int,
)
from src.utils.settings import load_settings
from src.models.training.trainer import Trainer as TruthLensTrainer, TrainerConfig
from src.models.training.training_step import TrainingStep, TrainingStepConfig
from src.models.training.training_utils import TrainingMetrics, get_device
from src.models.training.loss_functions import LossConfig, LossFactory


logger = logging.getLogger(__name__)
SETTINGS = load_settings()

_VALID_OPTIMIZATION_DIRECTIONS = {"minimize", "maximize"}

_UNIFIED_LABEL_CANDIDATES = (
    "bias_label",
    "ideology_label",
    "propaganda_label",
    "frame",
)


def _resolve_label_column(
    df: pd.DataFrame,
    label_column: str,
) -> str:

    if label_column in df.columns:
        return label_column

    if label_column != "label":
        raise ValueError(
            f"label column '{label_column}' not found in dataframe columns."
        )

    for candidate in _UNIFIED_LABEL_CANDIDATES:
        if candidate in df.columns:
            logger.info(
                "Column 'label' not found. Using '%s' as label column.",
                candidate,
            )
            return candidate

    raise ValueError(
        "No usable label column found. Expected 'label' or one of "
        f"{list(_UNIFIED_LABEL_CANDIDATES)}."
    )


def _prepare_training_frame(
    df: pd.DataFrame,
    *,
    label_column: str,
) -> pd.DataFrame:

    if label_column == "label":
        return df

    prepared = df.copy()
    prepared["label"] = prepared[label_column]
    return prepared


def _resolve_metric(
    metrics: Dict[str, Any],
    metric_name: str,
) -> float:

    if not isinstance(metrics, dict):
        raise TypeError(
            "trainer.evaluate(...) must return a dictionary."
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
    params: Dict[str, Any],
    text_column: str,
    validation_df: pd.DataFrame,
) -> Dict[str, Any]:

    train_sig = inspect.signature(train_function)

    kwargs: Dict[str, Any] = {}

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
    params: Dict[str, Any],
    *,
    train_function: Callable[..., tuple[Any, Any]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    metric_name: str,
) -> float:

    prepared_train_df = _prepare_training_frame(
        train_df,
        label_column=label_column,
    )

    prepared_val_df = _prepare_training_frame(
        val_df,
        label_column=label_column,
    )

    train_kwargs = _build_train_kwargs(
        train_function,
        params=params,
        text_column=text_column,
        validation_df=prepared_val_df,
    )

    train_result = train_function(prepared_train_df, **train_kwargs)

    if not isinstance(train_result, tuple) or len(train_result) != 2:
        raise TypeError(
            "train_function must return (trainer, eval_dataset)."
        )

    trainer, eval_dataset = train_result

    metrics = trainer.evaluate(eval_dataset)

    return _resolve_metric(metrics, metric_name)


def _sample_params_fallback(
    rng: np.random.Generator,
) -> Dict[str, Any]:

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
    label_column: str,
    n_trials: int,
    metric_name: str,
    direction: str,
    seed: int,
) -> Dict[str, Any]:

    rng = np.random.default_rng(seed)

    best_params: Dict[str, Any] | None = None
    best_value: float | None = None
    trial_metrics = TrainingMetrics()

    for trial_idx in range(1, n_trials + 1):

        params = _sample_params_fallback(rng)

        value = _evaluate_params(
            params,
            train_function=train_function,
            train_df=train_df,
            val_df=val_df,
            text_column=text_column,
            label_column=label_column,
            metric_name=metric_name,
        )

        trial_metrics.update(f"trial_{trial_idx}", value)
        trial_metrics.step = trial_idx

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
            "Fallback tuner failed to produce results."
        )

    return {
        "best_params": best_params,
        "best_value": float(best_value),
        "metric_name": metric_name,
        "direction": direction,
        "trials": n_trials,
        "trial_metrics": trial_metrics.to_dict(),
        "backend": "fallback",
        "label_column": label_column,
    }


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
) -> Dict[str, Any]:

    resolved_label_column = _resolve_label_column(df, label_column)

    ensure_dataframe(
        df,
        name="df",
        required_columns=[text_column, resolved_label_column],
        min_rows=10,
    )

    ensure_non_empty_text_column(df, text_column, name="df")

    working_df = df[df[resolved_label_column].notna()].reset_index(drop=True)

    if working_df.empty:
        raise ValueError(
            f"No non-null labels found in column '{resolved_label_column}'."
        )

    ensure_dataframe(
        working_df,
        name="working_df",
        required_columns=[text_column, resolved_label_column],
        min_rows=10,
    )

    ensure_non_empty_text_column(
        working_df,
        text_column,
        name="working_df",
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
            "direction must be either 'minimize' or 'maximize'."
        )

    effective_seed = (
        SETTINGS.training.seed if random_state is None else random_state
    )

    if validation_df is None:

        train_df, val_df = train_test_split(
            working_df,
            test_size=SETTINGS.training.optuna_validation_split,
            random_state=effective_seed,
            stratify=working_df[resolved_label_column],
        )

    else:

        resolved_validation_label_column = _resolve_label_column(
            validation_df,
            resolved_label_column,
        )

        ensure_dataframe(
            validation_df,
            name="validation_df",
            required_columns=[text_column, resolved_validation_label_column],
            min_rows=2,
        )

        ensure_non_empty_text_column(
            validation_df,
            text_column,
            name="validation_df",
        )

        filtered_validation_df = validation_df[
            validation_df[resolved_validation_label_column].notna()
        ].reset_index(drop=True)

        if filtered_validation_df.empty:
            raise ValueError(
                "validation_df has no valid labels."
            )

        train_df = working_df
        val_df = filtered_validation_df

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
            label_column=resolved_label_column,
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
            label_column=resolved_label_column,
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
        "label_column": resolved_label_column,
    }
