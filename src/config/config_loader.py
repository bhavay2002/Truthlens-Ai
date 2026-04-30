from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

logger = logging.getLogger(__name__)


# =========================================================
# SECTIONS (TYPED)
# =========================================================

@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    log_every: int
    eval_every: int
    checkpoint_every: int
    gradient_accumulation_steps: int
    early_stopping_patience: int


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float


@dataclass(frozen=True)
class SchedulerConfig:
    name: str
    step_mode: str  # step | epoch | metric
    warmup_steps: int


@dataclass(frozen=True)
class PrecisionConfig:
    use_amp: bool
    amp_dtype: str  # bf16 | fp16
    allow_tf32: bool


@dataclass(frozen=True)
class ModelConfig:
    encoder: str
    hidden_dim: int
    dropout: float
    gradient_checkpointing: bool = False
    flash_attention: bool = True
    torch_compile: bool = False
    compile_mode: str = "default"


@dataclass(frozen=True)
class LossConfig:
    ignore_index: int
    validate_loss: bool


@dataclass(frozen=True)
class DataConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool
    shuffle: bool


@dataclass(frozen=True)
class DistributedConfig:
    use_ddp: bool
    backend: str
    find_unused_parameters: bool


@dataclass(frozen=True)
class MonitoringConfig:
    spike_threshold: float
    ema_alpha: float
    health_threshold: float
    grad_monitor_interval: int


@dataclass(frozen=True)
class CheckpointConfig:
    dir: str
    max_checkpoints: int
    monitor_metric: str
    mode: str  # min | max


@dataclass(frozen=True)
class EvaluationConfig:
    device: Optional[str]


@dataclass(frozen=True)
class TrackingConfig:
    backend: str
    project_name: str
    run_name: Optional[str]
    tags: Dict[str, str]


# =========================================================
# ROOT CONFIG
# =========================================================

@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    precision: PrecisionConfig
    model: ModelConfig
    tasks: Dict[str, str]
    task_weights: Dict[str, float]
    loss: LossConfig
    data: DataConfig
    distributed: DistributedConfig
    monitoring: MonitoringConfig
    checkpoint: CheckpointConfig
    evaluation: EvaluationConfig
    tracking: TrackingConfig


# =========================================================
# VALIDATION
# =========================================================

def _validate(cfg: Dict[str, Any]) -> None:
    required_top = [
        "project", "training", "optimizer", "scheduler", "precision",
        "model", "tasks", "task_weights", "loss", "data",
        "distributed", "monitoring", "checkpoint", "evaluation", "tracking"
    ]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Missing top-level config keys: {missing}")

    # simple invariants
    if cfg["checkpoint"]["mode"] not in {"min", "max"}:
        raise ValueError("checkpoint.mode must be 'min' or 'max'")

    if cfg["scheduler"]["step_mode"] not in {"step", "epoch", "metric"}:
        raise ValueError("scheduler.step_mode must be step|epoch|metric")

    if cfg["precision"]["amp_dtype"] not in {"bf16", "fp16"}:
        raise ValueError("precision.amp_dtype must be bf16|fp16")

    # tasks vs weights alignment
    tasks = set(cfg["tasks"].keys())
    weights = set(cfg["task_weights"].keys())
    if tasks != weights:
        raise ValueError(f"tasks and task_weights mismatch: {tasks ^ weights}")


# =========================================================
# LOADER
# =========================================================

def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    _validate(raw)

    cfg = Config(
        project=ProjectConfig(**raw["project"]),
        training=TrainingConfig(**raw["training"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
        scheduler=SchedulerConfig(**raw["scheduler"]),
        precision=PrecisionConfig(**raw["precision"]),
        model=ModelConfig(**raw["model"]),
        tasks=dict(raw["tasks"]),
        task_weights=dict(raw["task_weights"]),
        loss=LossConfig(**raw["loss"]),
        data=DataConfig(**raw["data"]),
        distributed=DistributedConfig(**raw["distributed"]),
        monitoring=MonitoringConfig(**raw["monitoring"]),
        checkpoint=CheckpointConfig(**raw["checkpoint"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        tracking=TrackingConfig(**raw["tracking"]),
    )

    logger.info("Config loaded: %s | model=%s | epochs=%d",
                cfg.project.name, cfg.model.encoder, cfg.training.epochs)

    return cfg