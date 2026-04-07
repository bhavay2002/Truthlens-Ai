"""
File: settings.py

Purpose
-------
Central settings system for TruthLens AI.

This module loads configuration from config.yaml and converts
it into structured dataclasses used across the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.config_loader import (
    get_config_value,
    get_path,
    load_config,
)


# ---------------------------------------------------------
# Model Settings
# ---------------------------------------------------------

@dataclass(frozen=True)
class ModelSettings:
    name: str
    max_length: int
    path: Path


# ---------------------------------------------------------
# Feature Settings
# ---------------------------------------------------------

@dataclass(frozen=True)
class FeaturesSettings:
    tfidf_max_features: int
    tfidf_top_terms_per_doc: int


# ---------------------------------------------------------
# Data Settings
# ---------------------------------------------------------

@dataclass(frozen=True)
class DataSettings:
    raw_dir: Path
    interim_dir: Path
    augmentation_multiplier: float
    cleaned_dataset_path: Path
    merged_dataset_path: Path
    test_set_path: Path


# ---------------------------------------------------------
# Paths Settings
# ---------------------------------------------------------

@dataclass(frozen=True)
class PathsSettings:
    models_dir: Path
    logs_dir: Path
    reports_dir: Path
    training_log_path: Path
    evaluation_results_path: Path
    confusion_matrix_path: Path
    cleaning_report_path: Path
    tfidf_vectorizer_path: Path


# ---------------------------------------------------------
# API Settings
# ---------------------------------------------------------

@dataclass(frozen=True)
class ApiSettings:
    title: str
    description: str
    version: str
    text_preview_chars: int


# ---------------------------------------------------------
# Inference Settings
# ---------------------------------------------------------

@dataclass(frozen=True)
class InferenceSettings:
    batch_size: int
    device: str
    allow_raw_text_fallback: bool


# ---------------------------------------------------------
# Training Settings
# ---------------------------------------------------------

@dataclass(frozen=True)
class TrainingSettings:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    resume_from_checkpoint: bool
    validation_size: float
    test_size: float
    text_column: str
    run_cross_validation: bool
    cross_validation_splits: int
    cross_validation_metric: str
    run_hyperparameter_tuning: bool
    optuna_trials: int
    optuna_direction: str
    optuna_metric: str
    optuna_learning_rate_min: float
    optuna_learning_rate_max: float
    optuna_batch_sizes: tuple[int, ...]
    optuna_epoch_choices: tuple[int, ...]
    optuna_validation_split: float


# ---------------------------------------------------------
# Root Settings Object
# ---------------------------------------------------------

@dataclass(frozen=True)
class AppSettings:
    model: ModelSettings
    features: FeaturesSettings
    data: DataSettings
    paths: PathsSettings
    training: TrainingSettings
    api: ApiSettings
    inference: InferenceSettings


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def _as_int_tuple(value: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        return fallback
    return tuple(int(v) for v in value)


def _first_defined(
    config: dict[str, Any],
    key_paths: tuple[tuple[str, ...], ...],
    default: Any,
) -> Any:
    """Return the first configured value across multiple key paths."""
    sentinel = object()
    for key_path in key_paths:
        value = get_config_value(config, *key_path, default=sentinel)
        if value is not sentinel:
            return value
    return default


# ---------------------------------------------------------
# Load Settings
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    """
    Load application settings from config.yaml.
    """

    config = load_config()

    model = ModelSettings(
        name=str(
            _first_defined(
                config,
                (
                    ("model", "encoder", "name"),
                    ("model", "name"),
                ),
                "roberta-base",
            )
        ),
        max_length=int(
            _first_defined(
                config,
                (
                    ("model", "encoder", "max_length"),
                    ("model", "max_length"),
                ),
                512,
            )
        ),
        path=get_path(
            config,
            "model",
            "path",
            default="models/roberta_model",
        ),
    )

    features = FeaturesSettings(
        tfidf_max_features=int(
            _first_defined(
                config,
                (
                    ("features", "tfidf", "max_features"),
                    ("features", "tfidf_max_features"),
                ),
                5000,
            )
        ),
        tfidf_top_terms_per_doc=int(
            _first_defined(
                config,
                (
                    ("features", "tfidf", "top_terms_per_doc"),
                    ("features", "tfidf_top_terms_per_doc"),
                ),
                4,
            )
        ),
    )

    data = DataSettings(
        raw_dir=get_path(config, "data", "raw_dir", default="data/raw"),
        interim_dir=get_path(
            config,
            "data",
            "interim_dir",
            default="data/interim",
        ),
        augmentation_multiplier=float(
            get_config_value(
                config,
                "data",
                "augmentation_multiplier",
                default=2,
            )
        ),
        cleaned_dataset_path=get_path(
            config,
            "data",
            "cleaned_dataset_path",
            default="data/processed/cleaned_dataset.csv",
        ),
        merged_dataset_path=get_path(
            config,
            "data",
            "merged_dataset_path",
            default="data/interim/merged_dataset.csv",
        ),
        test_set_path=get_path(
            config,
            "data",
            "test_set_path",
            default="data/processed/test_set.csv",
        ),
    )

    paths = PathsSettings(
        models_dir=get_path(config, "paths", "models_dir", default="models"),
        logs_dir=get_path(config, "paths", "logs_dir", default="logs"),
        reports_dir=get_path(
            config,
            "paths",
            "reports_dir",
            default="reports",
        ),
        training_log_path=get_path(
            config,
            "paths",
            "training_log_path",
            default="logs/training.log",
        ),
        evaluation_results_path=get_path(
            config,
            "paths",
            "evaluation_results_path",
            default="reports/evaluation_results.json",
        ),
        confusion_matrix_path=get_path(
            config,
            "paths",
            "confusion_matrix_path",
            default="reports/confusion_matrix.png",
        ),
        cleaning_report_path=get_path(
            config,
            "paths",
            "cleaning_report_path",
            default="reports/data_cleaning_report.json",
        ),
        tfidf_vectorizer_path=get_path(
            config,
            "paths",
            "tfidf_vectorizer_path",
            default="models/tfidf_vectorizer.joblib",
        ),
    )

    run_cross_validation = bool(
        _first_defined(
            config,
            (
                ("training", "run_cross_validation"),
                ("cross_validation", "enabled"),
            ),
            False,
        )
    )
    cross_validation_splits = int(
        _first_defined(
            config,
            (
                ("training", "cross_validation_splits"),
                ("cross_validation", "splits"),
            ),
            5,
        )
    )
    cross_validation_metric = str(
        _first_defined(
            config,
            (
                ("training", "cross_validation_metric"),
                ("cross_validation", "metric"),
            ),
            "eval_loss",
        )
    )

    run_hyperparameter_tuning = bool(
        _first_defined(
            config,
            (
                ("training", "run_hyperparameter_tuning"),
                ("hyperparameter_tuning", "enabled"),
            ),
            False,
        )
    )
    optuna_trials = int(
        _first_defined(
            config,
            (
                ("training", "optuna_trials"),
                ("hyperparameter_tuning", "trials"),
            ),
            10,
        )
    )
    optuna_direction = str(
        _first_defined(
            config,
            (
                ("training", "optuna_direction"),
                ("hyperparameter_tuning", "direction"),
            ),
            "minimize",
        )
    )
    optuna_metric = str(
        _first_defined(
            config,
            (
                ("training", "optuna_metric"),
                ("hyperparameter_tuning", "metric"),
            ),
            "eval_loss",
        )
    )
    optuna_batch_sizes = _as_int_tuple(
        _first_defined(
            config,
            (
                ("training", "optuna_batch_sizes"),
                ("hyperparameter_tuning", "batch_sizes"),
            ),
            [8, 16],
        ),
        (8, 16),
    )
    optuna_epoch_choices = _as_int_tuple(
        _first_defined(
            config,
            (
                ("training", "optuna_epoch_choices"),
                ("hyperparameter_tuning", "epoch_choices"),
            ),
            [2, 3],
        ),
        (2, 3),
    )
    optuna_validation_split = float(
        _first_defined(
            config,
            (
                ("training", "optuna_validation_split"),
                ("hyperparameter_tuning", "validation_split"),
            ),
            0.2,
        )
    )

    sentinel = object()
    optuna_lr_min = get_config_value(
        config,
        "training",
        "optuna_learning_rate_min",
        default=sentinel,
    )
    optuna_lr_max = get_config_value(
        config,
        "training",
        "optuna_learning_rate_max",
        default=sentinel,
    )
    if optuna_lr_min is sentinel or optuna_lr_max is sentinel:
        optuna_lr_range = _first_defined(
            config,
            (
                ("training", "optuna_learning_rate_range"),
                ("hyperparameter_tuning", "learning_rate_range"),
            ),
            [1e-6, 5e-5],
        )
        if (
            not isinstance(optuna_lr_range, list)
            or len(optuna_lr_range) < 2
        ):
            optuna_lr_range = [1e-6, 5e-5]
        if optuna_lr_min is sentinel:
            optuna_lr_min = optuna_lr_range[0]
        if optuna_lr_max is sentinel:
            optuna_lr_max = optuna_lr_range[1]

    training = TrainingSettings(
        seed=int(get_config_value(config, "training", "seed", default=42)),
        epochs=int(get_config_value(config, "training", "epochs", default=3)),
        batch_size=int(
            get_config_value(config, "training", "batch_size", default=8)
        ),
        learning_rate=float(
            get_config_value(config, "training", "learning_rate", default=2e-5)
        ),
        resume_from_checkpoint=bool(
            get_config_value(
                config,
                "training",
                "resume_from_checkpoint",
                default=False,
            )
        ),
        validation_size=float(
            get_config_value(
                config,
                "training",
                "validation_size",
                default=0.15,
            )
        ),
        test_size=float(
            get_config_value(config, "training", "test_size", default=0.15)
        ),
        text_column=str(
            get_config_value(
                config,
                "training",
                "text_column",
                default="engineered_text",
            )
        ),
        run_cross_validation=run_cross_validation,
        cross_validation_splits=cross_validation_splits,
        cross_validation_metric=cross_validation_metric,
        run_hyperparameter_tuning=run_hyperparameter_tuning,
        optuna_trials=optuna_trials,
        optuna_direction=optuna_direction,
        optuna_metric=optuna_metric,
        optuna_learning_rate_min=float(optuna_lr_min),
        optuna_learning_rate_max=float(optuna_lr_max),
        optuna_batch_sizes=optuna_batch_sizes,
        optuna_epoch_choices=optuna_epoch_choices,
        optuna_validation_split=optuna_validation_split,
    )

    api = ApiSettings(
        title=str(
            get_config_value(
                config,
                "api",
                "title",
                default="TruthLens AI - Fake News Detection API",
            )
        ),
        description=str(
            get_config_value(
                config,
                "api",
                "description",
                default="Detect fake news using RoBERTa-based NLP model",
            )
        ),
        version=str(
            get_config_value(
                config,
                "api",
                "version",
                default="1.0.0",
            )
        ),
        text_preview_chars=int(
            get_config_value(
                config,
                "api",
                "text_preview_chars",
                default=100,
            )
        ),
    )

    inference = InferenceSettings(
        batch_size=int(
            get_config_value(
                config,
                "inference",
                "batch_size",
                default=16,
            )
        ),
        device=str(
            get_config_value(
                config,
                "inference",
                "device",
                default="auto",
            )
        ),
        allow_raw_text_fallback=bool(
            get_config_value(
                config,
                "inference",
                "allow_raw_text_fallback",
                default=True,
            )
        ),
    )

    return AppSettings(
        model=model,
        features=features,
        data=data,
        paths=paths,
        training=training,
        api=api,
        inference=inference,
    )
