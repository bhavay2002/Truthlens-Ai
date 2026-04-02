"""
File: src/training/cross_validation.py

Purpose
-------
Provide stratified cross-validation utilities for training pipelines.

Supports flexible training functions compatible with HuggingFace
Trainer-based pipelines or custom training functions.

Features
--------
• StratifiedKFold splitting
• Dynamic parameter handling
• Automatic metric extraction
• Robust validation
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils.input_validation import (
    ensure_dataframe,
    ensure_non_empty_text_column,
    ensure_positive_int,
)
from src.utils.settings import load_settings

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
    metrics: dict[str, Any],
    metric_name: str,
) -> float:
    """Resolve metric name from trainer output dictionary."""
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


def cross_validate_model(
    df: pd.DataFrame,
    train_function: Callable[..., tuple[Any, Any]],
    n_splits: int | None = None,
    *,
    text_column: str = "text",
    label_column: str = "label",
    params: dict[str, Any] | None = None,
    metric_name: str | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Run stratified cross-validation and return summary metrics."""

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
            f"No non-null labels found in column '{resolved_label_column}'."
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
        raise ValueError("Cross-validation requires at least 2 classes")
    if len(working_df) < effective_splits:
        raise ValueError("n_splits cannot exceed number of rows")

    minimum_class_size = int(working_df[resolved_label_column].value_counts().min())
    if minimum_class_size < effective_splits:
        raise ValueError(
            "Each class must contain at least n_splits samples for stratified "
            f"cross-validation. Smallest class has {minimum_class_size}, "
            f"n_splits is {effective_splits}."
        )

    effective_metric = metric_name or SETTINGS.training.cross_validation_metric
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

    fold_scores: list[float] = []

    X = working_df[text_column]
    y = working_df[resolved_label_column]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
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

        train_kwargs: dict[str, Any] = {}
        if supports_params:
            train_kwargs["params"] = params
        if supports_text_column:
            train_kwargs["text_column"] = text_column
        if supports_validation_df:
            train_kwargs["validation_df"] = fold_val_df
        if supports_test_df:
            train_kwargs["test_df"] = fold_val_df

        train_result = train_function(fold_train_df, **train_kwargs)
        if not isinstance(train_result, tuple) or len(train_result) != 2:
            raise TypeError(
                "train_function must return (trainer, eval_dataset)."
            )

        trainer, eval_dataset = train_result
        metrics = trainer.evaluate(eval_dataset)
        score = _resolve_metric(metrics, effective_metric)
        fold_scores.append(score)

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
        "mean_score": mean_score,
        "std_score": std_score,
        "n_splits": effective_splits,
        "label_column": resolved_label_column,
    }
