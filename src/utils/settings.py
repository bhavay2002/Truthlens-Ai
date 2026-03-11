from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.utils.config_loader import get_config_value, get_path, load_config


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
    augmentation_multiplier: int
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


def _as_int_tuple(value: Any, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        return fallback
    return tuple(int(v) for v in value)


@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    config = load_config()

    model = ModelSettings(
        name=str(get_config_value(config, "model", "name", default="roberta-base")),
        max_length=int(get_config_value(config, "model", "max_length", default=256)),
        path=get_path(config, "model", "path", default="models/roberta_model"),
    )

    features = FeaturesSettings(
        tfidf_max_features=int(
            get_config_value(config, "features", "tfidf_max_features", default=5000)
        ),
        tfidf_top_terms_per_doc=int(
            get_config_value(config, "features", "tfidf_top_terms_per_doc", default=4)
        ),
    )

    data = DataSettings(
        raw_dir=get_path(config, "data", "raw_dir", default="data/raw"),
        interim_dir=get_path(config, "data", "interim_dir", default="data/interim"),
        augmentation_multiplier=int(
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

    paths = PathsSettings(
        models_dir=get_path(config, "paths", "models_dir", default="models"),
        logs_dir=get_path(config, "paths", "logs_dir", default="logs"),
        reports_dir=get_path(config, "paths", "reports_dir", default="reports"),
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

    training = TrainingSettings(
        seed=int(get_config_value(config, "training", "seed", default=42)),
        epochs=int(get_config_value(config, "training", "epochs", default=3)),
        batch_size=int(get_config_value(config, "training", "batch_size", default=8)),
        learning_rate=float(
            get_config_value(config, "training", "learning_rate", default=2e-5)
        ),
        resume_from_checkpoint=bool(
            get_config_value(config, "training", "resume_from_checkpoint", default=False)
        ),
        validation_size=float(
            get_config_value(config, "training", "validation_size", default=0.15)
        ),
        test_size=float(
            get_config_value(config, "training", "test_size", default=0.15)
        ),
        text_column=str(
            get_config_value(config, "training", "text_column", default="engineered_text")
        ),
        run_cross_validation=bool(
            get_config_value(config, "training", "run_cross_validation", default=False)
        ),
        cross_validation_splits=int(
            get_config_value(config, "training", "cross_validation_splits", default=5)
        ),
        cross_validation_metric=str(
            get_config_value(config, "training", "cross_validation_metric", default="eval_loss")
        ),
        run_hyperparameter_tuning=bool(
            get_config_value(config, "training", "run_hyperparameter_tuning", default=False)
        ),
        optuna_trials=int(get_config_value(config, "training", "optuna_trials", default=10)),
        optuna_direction=str(
            get_config_value(config, "training", "optuna_direction", default="minimize")
        ),
        optuna_metric=str(
            get_config_value(config, "training", "optuna_metric", default="eval_loss")
        ),
        optuna_learning_rate_min=float(
            get_config_value(config, "training", "optuna_learning_rate_min", default=1e-6)
        ),
        optuna_learning_rate_max=float(
            get_config_value(config, "training", "optuna_learning_rate_max", default=5e-5)
        ),
        optuna_batch_sizes=_as_int_tuple(
            get_config_value(config, "training", "optuna_batch_sizes", default=[8, 16]),
            (8, 16),
        ),
        optuna_epoch_choices=_as_int_tuple(
            get_config_value(config, "training", "optuna_epoch_choices", default=[2, 3]),
            (2, 3),
        ),
        optuna_validation_split=float(
            get_config_value(config, "training", "optuna_validation_split", default=0.2)
        ),
    )

    return AppSettings(
        model=model,
        features=features,
        data=data,
        paths=paths,
        training=training,
    )
