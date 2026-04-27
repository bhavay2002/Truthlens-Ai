from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

from src.models.registry.model_factory import build_model
from src.models.optimization.optimizer_factory import build_optimizer
from src.models.optimization.lr_scheduler import build_scheduler
from src.data_processing.dataloader_factory import build_dataloader

from .training_step import TrainingStep, TrainingStepConfig
from .monitor_engine import MonitoringEngine
from .experiment_tracker import ExperimentTracker
from .task_scheduler import TaskScheduler
from .loss_engine import LossEngine, LossEngineConfig
from .evaluation_engine import EvaluationEngine
from .trainer import Trainer

from src.config.task_config import get_task_type
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


# =========================================================
# HELPERS
# =========================================================

def _validate_params(params: Dict[str, Any]):

    required = ["lr", "batch_size"]

    for k in required:
        if k not in params:
            raise ValueError(f"Missing required param: {k}")


def _resolve_device(params: Dict[str, Any]) -> str:
    if "device" in params:
        return params["device"]
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# FACTORY
# =========================================================

def create_trainer_fn(
    *,
    task: str,
    train_df,
    val_df,
    params: Dict[str, Any],
):

    # =====================================================
    # VALIDATION + SEED
    # =====================================================

    _validate_params(params)

    seed = params.get("seed", 42)
    set_seed(seed)

    device = _resolve_device(params)

    # =====================================================
    # MODEL
    # =====================================================

    model = build_model(
        task=task,
        config=params,
    )

    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader: DataLoader = build_dataloader(
        df=train_df,
        task=task,
        batch_size=params["batch_size"],
        shuffle=True,
    )

    val_loader: DataLoader = build_dataloader(
        df=val_df,
        task=task,
        batch_size=params["batch_size"],
        shuffle=False,
    )

    # =====================================================
    # OPTIMIZER + SCHEDULER
    # =====================================================

    optimizer = build_optimizer(
        model=model,
        lr=params["lr"],
        weight_decay=params.get("weight_decay", 0.0),
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        config=params,
    )

    # =====================================================
    # LOSS ENGINE (DYNAMIC )
    # =====================================================

    task_type = get_task_type(task)

    loss_engine = LossEngine(
        LossEngineConfig(
            task_types={task: task_type},
        )
    )

    # =====================================================
    # MONITORING
    # =====================================================

    monitor = MonitoringEngine(
        params.get("monitor_config")
    )

    # =====================================================
    # TASK SCHEDULER
    # =====================================================

    task_scheduler = TaskScheduler(
        tasks=[task],
        config=params.get("task_scheduler_config"),
    )

    # =====================================================
    # TRACKER
    # =====================================================

    tracker = ExperimentTracker(
        params.get("tracker_config")
    )

    # =====================================================
    # TRAINING STEP
    # =====================================================

    training_step = TrainingStep(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_engine=loss_engine,
        monitor=monitor,
        tracker=tracker,
        task_scheduler=task_scheduler,
        instrumentation=params.get("instrumentation"),
        config=TrainingStepConfig(
            gradient_accumulation_steps=params.get("grad_accum", 1),
            max_grad_norm=params.get("max_grad_norm", 1.0),
            use_mixed_precision=params.get("amp", True),
        ),
        device=device,
    )

    # =====================================================
    # EVALUATION
    # =====================================================

    evaluator = EvaluationEngine()

    # =====================================================
    # OPTIONAL COMPONENTS
    # =====================================================

    checkpoint = params.get("checkpoint")      # plug your CheckpointEngine
    distributed = params.get("distributed")    # plug DistributedEngine

    # =====================================================
    # TRAINER
    # =====================================================

    trainer = Trainer(
        config_path=params.get("config_path", ""),
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_step=training_step,
        evaluator=evaluator,
        checkpoint=checkpoint,
        distributed=distributed,
        tracker=tracker,
        monitor_metric=params.get("monitor_metric", "val_loss"),
        maximize_metric=params.get("maximize_metric", False),
    )

    logger.info(
        "Trainer created | task=%s | device=%s | batch_size=%s",
        task,
        device,
        params["batch_size"],
    )

    return trainer