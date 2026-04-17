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
# Dataclasses
# ---------------------------------------------------------

@dataclass(frozen=True)
class ModelSettings:
    name: str
    max_length: int
    path: Path


@dataclass(frozen=True)
class FeaturesSettings:
    tfidf_max_features: int
    tfidf_top_terms_per_doc: int


@dataclass(frozen=True)
class DataSettings:
    raw_dir: Path
    interim_dir: Path
    augmentation_multiplier: float
    cleaned_dataset_path: Path
    merged_dataset_path: Path
    test_set_path: Path


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


@dataclass(frozen=True)
class ApiSettings:
    title: str
    description: str
    version: str
    text_preview_chars: int


@dataclass(frozen=True)
class InferenceSettings:
    batch_size: int
    device: str
    allow_raw_text_fallback: bool


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
# Helpers
# ---------------------------------------------------------

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_int_tuple(value: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        return fallback
    return tuple(int(v) for v in value)


def _first_defined(config, key_paths, default):
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

    config = load_config()

    # ---------------- MODEL ----------------

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
        path=get_path(config, "model", "path", default="models/roberta_model"),
    )

    # ---------------- FEATURES ----------------

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

    # ---------------- DATA ----------------

    data = DataSettings(
        raw_dir=_ensure_dir(get_path(config, "data", "raw_dir", default="data/raw")),
        interim_dir=_ensure_dir(
            get_path(config, "data", "interim_dir", default="data/interim")
        ),
        augmentation_multiplier=float(
            get_config_value(config, "data", "augmentation_multiplier", default=2)
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

    # ---------------- PATHS ----------------

    models_dir = _ensure_dir(get_path(config, "paths", "models_dir", default="models"))
    logs_dir = _ensure_dir(get_path(config, "paths", "logs_dir", default="logs"))
    reports_dir = _ensure_dir(get_path(config, "paths", "reports_dir", default="reports"))

    paths = PathsSettings(
        models_dir=models_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        training_log_path=logs_dir / "training.log",
        evaluation_results_path=reports_dir / "evaluation_results.json",
        confusion_matrix_path=reports_dir / "confusion_matrix.png",
        cleaning_report_path=reports_dir / "data_cleaning_report.json",
        tfidf_vectorizer_path=models_dir / "tfidf_vectorizer.joblib",
    )

    # ---------------- TRAINING ----------------

    validation_size = float(get_config_value(config, "training", "validation_size", default=0.15))
    test_size = float(get_config_value(config, "training", "test_size", default=0.15))

    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be < 1")

    cv_config = config.get("cross_validation", {})
    hpt_config = config.get("hyperparameter_tuning", {})

    lr_range = hpt_config.get("learning_rate_range")
    if not isinstance(lr_range, list) or len(lr_range) < 2:
        lr_range = [1e-6, 5e-5]

    valid_metrics = {"eval_loss", "accuracy", "f1"}
    cv_metric = str(cv_config.get("metric", "eval_loss"))
    if cv_metric not in valid_metrics:
        cv_metric = "eval_loss"

    training = TrainingSettings(
        seed=int(get_config_value(config, "training", "seed", default=42)),
        epochs=int(get_config_value(config, "training", "epochs", default=3)),
        batch_size=max(1, int(get_config_value(config, "training", "batch_size", default=8))),
        learning_rate=float(get_config_value(config, "training", "learning_rate", default=2e-5)),
        resume_from_checkpoint=bool(
            get_config_value(config, "training", "resume_from_checkpoint", default=False)
        ),
        validation_size=validation_size,
        test_size=test_size,
        text_column=str(get_config_value(config, "training", "text_column", default="text")),

        # ===============================
        # Cross Validation
        # ===============================
        run_cross_validation=bool(cv_config.get("enabled", False)),
        cross_validation_splits=max(2, int(cv_config.get("splits", 5))),
        cross_validation_metric=cv_metric,

        # ===============================
        # Hyperparameter Tuning (Optuna)
        # ===============================
        run_hyperparameter_tuning=bool(hpt_config.get("enabled", False)),
        optuna_trials=max(1, int(hpt_config.get("trials", 10))),
        optuna_direction=str(hpt_config.get("direction", "minimize")).lower(),
        optuna_metric=str(hpt_config.get("metric", "eval_loss")),

        optuna_learning_rate_min=float(lr_range[0]),
        optuna_learning_rate_max=float(lr_range[-1]),

        optuna_batch_sizes=_as_int_tuple(
            hpt_config.get("batch_sizes"), (8, 16)
        ),
        optuna_epoch_choices=_as_int_tuple(
            hpt_config.get("epoch_choices"), (2, 3)
        ),

        optuna_validation_split=float(
            hpt_config.get("validation_split", 0.2)
        ),
    )

    # ---------------- API ----------------

    api = ApiSettings(
        title=str(get_config_value(config, "api", "title", default="TruthLens API")),
        description=str(
            get_config_value(
                config,
                "api",
                "description",
                default="Fake news detection using transformers",
            )
        ),
        version=str(get_config_value(config, "api", "version", default="1.0")),
        text_preview_chars=int(get_config_value(config, "api", "text_preview_chars", default=100)),
    )

    # ---------------- INFERENCE ----------------

    inference = InferenceSettings(
        batch_size=max(1, int(get_config_value(config, "inference", "batch_size", default=16))),
        device=str(get_config_value(config, "inference", "device", default="auto")),
        allow_raw_text_fallback=bool(
            get_config_value(config, "inference", "allow_raw_text_fallback", default=True)
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