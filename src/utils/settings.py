from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict

from src.utils.config_loader import load_app_config, AppConfig, TaskConfig


# =========================================================
# RUNTIME SETTINGS (TASK-AWARE)
# =========================================================

@dataclass(frozen=True)
class TaskRuntimeSettings:
    name: str
    train_path: Path
    val_path: Path | None
    test_path: Path | None
    num_labels: int
    task_type: str


@dataclass(frozen=True)
class TrainingRuntimeSettings:
    batch_size: int
    epochs: int
    learning_rate: float
    device: str


@dataclass(frozen=True)
class ModelRuntimeSettings:
    encoder_name: str
    hidden_size: int


@dataclass(frozen=True)
class AppSettings:
    model: ModelRuntimeSettings
    tasks: Dict[str, TaskRuntimeSettings]
    training: TrainingRuntimeSettings
    output_dir: Path
    seed: int


# =========================================================
# LOAD SETTINGS (SINGLE SOURCE OF TRUTH)
# =========================================================

@lru_cache(maxsize=1)
def load_settings() -> AppSettings:

    config: AppConfig = load_app_config()

    # ---------------- MODEL ----------------
    model = ModelRuntimeSettings(
        encoder_name=config.model.encoder.name,
        hidden_size=config.model.encoder.hidden_size,
    )

    # ---------------- TASKS ----------------
    tasks: Dict[str, TaskRuntimeSettings] = {}

    for name, task in config.tasks.items():

        tasks[name] = TaskRuntimeSettings(
            name=name,
            train_path=task.dataset.train_path,
            val_path=task.dataset.validation_path,
            test_path=task.dataset.test_path,
            num_labels=task.num_labels,
            task_type=task.task_type,
        )

    # ---------------- TRAINING ----------------
    training = TrainingRuntimeSettings(
        batch_size=config.training.batch_size,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        device=config.training.device,
    )

    return AppSettings(
        model=model,
        tasks=tasks,
        training=training,
        output_dir=config.experiment.output_dir,
        seed=config.experiment.seed,
    )