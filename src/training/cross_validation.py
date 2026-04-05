"""
File Name: cross_validation.py
Module: TruthLens AI - Training Cross Validation
Description:
    Stratified cross-validation utilities for TruthLens training pipelines.
    Designed to work with HuggingFace Trainer pipelines or custom training
    functions returning (trainer, eval_dataset).

Dependencies:
    inspect
    logging
    typing
    numpy
    pandas
    sklearn.model_selection
    src.utils.input_validation
    src.utils.settings

Inputs:
    df: pandas DataFrame containing training data
    train_function: callable returning (trainer, eval_dataset)
    n_splits: number of folds
    text_column: text column name
    label_column: label column name
    params: optional training parameters
    metric_name: metric to extract from trainer.evaluate
    random_state: random seed

Outputs:
    dictionary containing cross-validation metrics and statistics
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
    """
    Resolve the label column for cross-validation.
    """

    if label_column in df.columns:
        return label_column

    if label_column != "label":
        raise ValueError(
            f"Label column '{label_column}' not found in dataframe."
        )

    for candidate in _UNIFIED_LABEL_CANDIDATES:
        if candidate in df.columns:
            logger.info(
                "Column 'label' not found. Using '%s' as label column.",
                candidate,
            )
            return candidate

    raise ValueError(
        "No valid label column found. Expected 'label' or one of "
        f"{list(_UNIFIED_LABEL_CANDIDATES)}."
    )


def _prepare_training_frame(
    df: pd.DataFrame,
    *,
    label_column: str,
) -> pd.DataFrame:
    """
    Ensure training frame contains column 'label'.
    """

    if label_column == "label":
        return df

    prepared = df.copy()
    prepared["label"] = prepared[label_column]
    return prepared


def _resolve_metric(
    metrics: Dict[str, Any],
    metric_name: str,
) -> float:
    """
    Resolve desired metric from trainer output dictionary.
    """

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
        f"Metric '{metric_name}' not found in metrics: "
        f"{sorted(metrics.keys())}"
    )


def cross_validate_model(
    df: pd.DataFrame,
    train_function: Callable[..., tuple[Any, Any]],
    n_splits: int | None = None,
    *,
    text_column: str = "text",
    label_column: str = "label",
    params: Dict[str, Any] | None = None,
    metric_name: str | None = None,
    random_state: int | None = None,
) -> Dict[str, Any]:
    """
    Run stratified cross-validation and return summary metrics.
    """

    resolved_label_column = _resolve_label_column(df, label_column)

    ensure_dataframe(
        df,
        name="df",
        required_columns=[text_column, resolved_label_column],
        min_rows=3,
    )

    working_df = df[df[resolved_label_column].notna()].reset_index(drop=True)

    if working_df.empty:
        raise ValueError(
            f"No valid labels found in column '{resolved_label_column}'."
        )

    ensure_dataframe(
        working_df,
        name="working_df",
        required_columns=[text_column, resolved_label_column],
        min_rows=3,
    )

    ensure_non_empty_text_column(
        working_df,
        text_column,
        name="working_df",
    )

    effective_splits = (
        n_splits
        if n_splits is not None
        else SETTINGS.training.cross_validation_splits
    )

    ensure_positive_int(
        effective_splits,
        name="n_splits",
        min_value=2,
    )

    if working_df[resolved_label_column].nunique() < 2:
        raise ValueError(
            "Cross-validation requires at least two classes."
        )

    if len(working_df) < effective_splits:
        raise ValueError(
            "n_splits cannot exceed number of rows."
        )

    minimum_class_size = int(
        working_df[resolved_label_column].value_counts().min()
    )

    if minimum_class_size < effective_splits:
        raise ValueError(
            "Each class must contain at least n_splits samples for "
            "stratified cross-validation."
        )

    effective_metric = (
        metric_name or SETTINGS.training.cross_validation_metric
    )

    effective_seed = (
        SETTINGS.training.seed if random_state is None else random_state
    )

    skf = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=effective_seed,
    )

    train_sig = inspect.signature(train_function)

    supports_params = "params" in train_sig.parameters
    supports_text_column = "text_column" in train_sig.parameters
    supports_validation_df = "validation_df" in train_sig.parameters
    supports_test_df = "test_df" in train_sig.parameters

    fold_scores: List[float] = []
    fold_metrics = TrainingMetrics()

    X = working_df[text_column]
    y = working_df[resolved_label_column]

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y),
        start=1,
    ):

        fold_train_df = working_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = working_df.iloc[val_idx].reset_index(drop=True)

        fold_train_df = _prepare_training_frame(
            fold_train_df,
            label_column=resolved_label_column,
        )

        fold_val_df = _prepare_training_frame(
            fold_val_df,
            label_column=resolved_label_column,
        )

        train_kwargs: Dict[str, Any] = {}

        if supports_params:
            train_kwargs["params"] = params

        if supports_text_column:
            train_kwargs["text_column"] = text_column

        if supports_validation_df:
            train_kwargs["validation_df"] = fold_val_df

        if supports_test_df:
            train_kwargs["test_df"] = fold_val_df

        train_result = train_function(
            fold_train_df,
            **train_kwargs,
        )

        if (
            not isinstance(train_result, tuple)
            or len(train_result) != 2
        ):
            raise TypeError(
                "train_function must return (trainer, eval_dataset)."
            )

        trainer, eval_dataset = train_result

        metrics = trainer.evaluate(eval_dataset)

        score = _resolve_metric(
            metrics,
            effective_metric,
        )

        fold_scores.append(score)
        fold_metrics.update(f"fold_{fold}", score)
        fold_metrics.epoch = fold

        logger.info(
            "CV fold %s/%s - %s: %.4f",
            fold,
            effective_splits,
            effective_metric,
            score,
        )

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    return {
        "metric_name": effective_metric,
        "fold_scores": fold_scores,
        "fold_metrics": fold_metrics.to_dict(),
        "mean_score": mean_score,
        "std_score": std_score,
        "n_splits": effective_splits,
        "label_column": resolved_label_column,
    }