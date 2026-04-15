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

import torch
import gc
import os
from concurrent.futures import ThreadPoolExecutor

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
from src.models.training.training_utils import TrainingMetrics
from src.features.dataset_feature_generator import DatasetFeatureGenerator
from src.features.feature_schema_validator import FeatureSchemaValidator
from src.features.feature_statistics import FeatureStatistics
from src.training.checkpointing import list_checkpoints

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


def _run_feature_diagnostics(texts: list[str], label: str = "") -> None:
    """
    Compute feature statistics and validate schema for a text corpus.

    Uses DatasetFeatureGenerator to extract a feature matrix from the
    provided texts, FeatureStatistics to surface the dataset summary and
    any constant (zero-variance) features, and FeatureSchemaValidator to
    confirm that all feature vectors match the inferred schema.

    Runs non-fatally — any error is captured as a warning so it does not
    interrupt the cross-validation loop.

    Parameters
    ----------
    texts : list[str]
        Raw article texts from the working dataframe.
    label : str
        Descriptive tag used in log messages (e.g. "cross-validation dataset").
    """
    try:
        from src.features.pipelines.feature_pipeline import FeaturePipeline
        from src.features.pipelines.batch_feature_pipeline import BatchFeaturePipeline

        tag = f" [{label}]" if label else ""
        logger.info("Running feature diagnostics%s | samples=%d", tag, len(texts))

        batch_pipeline = BatchFeaturePipeline(pipeline=FeaturePipeline())
        generator = DatasetFeatureGenerator(pipeline=batch_pipeline)
        _, feature_names = generator.generate(texts)

        contexts = generator._build_contexts(texts)
        feature_dicts = batch_pipeline._sequential_extract(contexts)

        stats = FeatureStatistics()
        summary = stats.dataset_summary(feature_dicts)
        logger.info(
            "Feature diagnostics%s | samples=%d features=%d mean_variance=%.6f",
            tag,
            int(summary["num_samples"]),
            int(summary["num_features"]),
            summary["mean_variance"],
        )

        constant = stats.detect_constant_features(feature_dicts)
        if constant:
            logger.warning(
                "Detected %d constant feature(s)%s: %s",
                len(constant),
                tag,
                constant[:10],
            )

        validator = FeatureSchemaValidator(
            expected_features=feature_names,
            strict=False,
            allow_missing=True,
            allow_extra=True,
        )
        validator.validate_batch(feature_dicts[:min(5, len(feature_dicts))])
        logger.info(
            "Feature schema validated%s | schema_features=%d",
            tag,
            validator.schema_summary()["num_features"],
        )

    except Exception as _diag_exc:
        logger.warning("Feature diagnostics skipped (non-fatal): %s", _diag_exc)

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
    checkpoint_root: str | None = None,
    use_parallel: bool = False,  # 🔥 NEW
) -> Dict[str, Any]:

    resolved_label_column = _resolve_label_column(df, label_column)

    ensure_dataframe(
        df,
        name="df",
        required_columns=[text_column, resolved_label_column],
        min_rows=3,
    )

    working_df = df[df[resolved_label_column].notna()].reset_index(drop=True)

    ensure_non_empty_text_column(working_df, text_column, name="working_df")

    effective_splits = (
        n_splits
        if n_splits is not None
        else SETTINGS.training.cross_validation_splits
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

    _run_feature_diagnostics(
        working_df[text_column].tolist(),
        label="cross-validation dataset",
    )

    train_sig = inspect.signature(train_function)

    supports_params = "params" in train_sig.parameters
    supports_text_column = "text_column" in train_sig.parameters
    supports_validation_df = "validation_df" in train_sig.parameters
    supports_test_df = "test_df" in train_sig.parameters

    fold_scores: List[float] = []
    fold_metrics = TrainingMetrics()

    cache: Dict[Any, float] = {}

    def run_fold(fold, train_idx, val_idx):

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

        cache_key = (fold, tuple(sorted((params or {}).items())))
        if cache_key in cache:
            return cache[cache_key]

        trainer, eval_dataset = train_function(
            fold_train_df,
            **train_kwargs,
        )

        with torch.no_grad():  
            metrics = trainer.evaluate(eval_dataset)

        score = _resolve_metric(metrics, effective_metric)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info(
            "[Fold %d/%d] %s=%.4f",
            fold,
            effective_splits,
            effective_metric,
            score,
        )

        cache[cache_key] = score
        return score

    splits = list(skf.split(working_df[text_column], working_df[resolved_label_column]))

    if use_parallel:
        with ThreadPoolExecutor(max_workers=min(2, os.cpu_count() or 1)) as executor:
            results = list(
                executor.map(
                    lambda x: run_fold(x[0] + 1, x[1][0], x[1][1]),
                    enumerate(splits),
                )
            )
        fold_scores.extend(results)
    else:
        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            score = run_fold(fold, train_idx, val_idx)
            fold_scores.append(score)

    for i, score in enumerate(fold_scores, start=1):
        fold_metrics.update(f"fold_{i}", score)

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    available_checkpoints: list[str] = []
    if isinstance(checkpoint_root, str) and checkpoint_root.strip():
        available_checkpoints = [
            str(p) for p in list_checkpoints(checkpoint_root)
        ]

    return {
        "metric_name": effective_metric,
        "fold_scores": fold_scores,
        "fold_metrics": fold_metrics.to_dict(),
        "mean_score": mean_score,
        "std_score": std_score,
        "n_splits": effective_splits,
        "label_column": resolved_label_column,
        "available_checkpoints": available_checkpoints,
    }