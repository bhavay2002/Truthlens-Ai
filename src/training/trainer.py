from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.models.config.model_config import ModelConfigLoader

from src.training.training_setup import (
    TrainingSetupConfig,
    setup_runtime,
    optimize_model,
    run_sanity_check,
)

from .training_step import TrainingStep
from .evaluation_engine import EvaluationEngine
from ..models.checkpointing.checkpoint_manager import CheckpointEngine
from .distributed_engine import DistributedEngine
from .experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


# =========================================================
# TRAINER (PRODUCTION-GRADE)
# =========================================================

class Trainer:

    def __init__(
        self,
        config_path: str,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        training_step: TrainingStep,
        evaluator: EvaluationEngine,
        checkpoint: Optional[CheckpointEngine] = None,
        distributed: Optional[DistributedEngine] = None,
        tracker: Optional[ExperimentTracker] = None,
        monitor_metric: str = "val_loss",
        maximize_metric: bool = False,
    ):

        # -------------------------------------------------
        # CONFIG
        # -------------------------------------------------
        self.cfg = ModelConfigLoader.load_multitask_config(config_path)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.training_step = training_step
        self.evaluator = evaluator
        self.checkpoint = checkpoint
        self.distributed = distributed
        self.tracker = tracker

        self.monitor_metric = monitor_metric
        self.maximize_metric = maximize_metric

        # -------------------------------------------------
        # TRAINING SETUP (🔥 NEW)
        # -------------------------------------------------
        self.setup_cfg = TrainingSetupConfig()

        self.device = setup_runtime(self.setup_cfg)

        self.model = optimize_model(self.model)
        self.model.to(self.device)

        # -------------------------------------------------
        # TRAINING PARAMS
        # -------------------------------------------------
        self.epochs = self.cfg.training.num_epochs
        self.early_patience = self.cfg.training.early_stopping_patience

        self.global_step = 0
        self.best_metric = None
        self.no_improve_epochs = 0

        # -------------------------------------------------
        # DISTRIBUTED
        # -------------------------------------------------
        if self.distributed:
            self.distributed.initialize()
            self.model = self.distributed.wrap_model(self.model)

        # -------------------------------------------------
        # LOG CONFIG
        # -------------------------------------------------
        if self.tracker and self._is_main():
            self.tracker.log_params(asdict(self.cfg))

        logger.info("Trainer initialized (PRODUCTION-GRADE)")

    # =====================================================
    # TRAIN ENTRY
    # =====================================================

    def train(self):

        # 🔥 SANITY CHECK (CRITICAL)
        if self.setup_cfg.run_sanity_check:
            self._run_sanity_check()

        for epoch in range(self.epochs):

            if self._is_main():
                logger.info("Epoch %d/%d", epoch + 1, self.epochs)

            # DDP sampler sync
            if self.distributed and self.distributed.initialized:
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch()

            # -------------------------
            # VALIDATION
            # -------------------------
            if self.val_loader:

                if self.distributed and self.distributed.initialized:
                    self.distributed.barrier()

                val_metrics = self.evaluate()

                metric_value = val_metrics.get(self.monitor_metric)

                if metric_value is not None:
                    self._update_early_stopping(metric_value)

                # LOGGING
                if self.tracker and self._is_main():
                    self.tracker.log_metrics(val_metrics, step=self.global_step)

                # CHECKPOINT
                if self.checkpoint and self._is_main():
                    self._save_checkpoint(val_metrics)

                # EARLY STOP
                if self.no_improve_epochs >= self.early_patience:
                    if self._is_main():
                        logger.warning("Early stopping triggered")
                    break

        # -------------------------------------------------
        # CLEANUP
        # -------------------------------------------------
        if self.tracker and self._is_main():
            self.tracker.finish()

        if self.distributed:
            self.distributed.cleanup()

    # =====================================================
    # TRAIN EPOCH
    # =====================================================

    def _train_epoch(self):

        for batch in self.train_loader:

            outputs = self.training_step.run(batch, self.global_step)

            self.global_step += 1

            # Skip failed step
            if outputs.get("skipped"):
                continue

            # -------------------------
            # LOGGING
            # -------------------------
            if self.global_step % 50 == 0 and self._is_main():

                log_data = {
                    "train/loss": float(outputs.get("raw_loss", 0.0)),
                    "train/grad_norm": outputs.get("grad_norm"),
                    "train/throughput": outputs.get("throughput"),
                }

                logger.info("Step %d | %s", self.global_step, log_data)

                if self.tracker:
                    self.tracker.log_metrics(log_data, step=self.global_step)

            # -------------------------
            # CHECKPOINT
            # -------------------------
            if (
                self.checkpoint
                and self.global_step % 500 == 0
                and self._is_main()
            ):
                self.checkpoint.save(
                    step=self.global_step,
                    model=self._unwrap_model(),
                )

    # =====================================================
    # SANITY CHECK
    # =====================================================

    def _run_sanity_check(self):

        if not self.train_loader:
            return

        logger.info("Running sanity check...")

        batch = next(iter(self.train_loader))

        run_sanity_check(
            model=self._unwrap_model(),
            batch=batch,
            training_step=self.training_step,
            device=self.device,
        )

        logger.info("Sanity check passed")

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    def _update_early_stopping(self, metric_value: float):

        improved = False

        if self.best_metric is None:
            improved = True
        elif self.maximize_metric:
            improved = metric_value > self.best_metric
        else:
            improved = metric_value < self.best_metric

        if improved:
            self.best_metric = metric_value
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1

    # =====================================================
    # CHECKPOINT LOGIC
    # =====================================================

    def _save_checkpoint(self, metrics: Dict[str, Any]):

        self.checkpoint.save(
            step=self.global_step,
            model=self._unwrap_model(),
            metrics=metrics,
            tag="epoch",
        )

        metric_value = metrics.get(self.monitor_metric)

        if metric_value is None:
            return

        if (
            self.best_metric is not None
            and metric_value == self.best_metric
        ):
            self.checkpoint.save(
                step=self.global_step,
                model=self._unwrap_model(),
                metrics=metrics,
                tag="best",
            )

    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(self) -> Dict[str, Any]:

        if not self.val_loader:
            return {}

        model = self._unwrap_model()
        results = self.evaluator.evaluate(model, self.val_loader)

        if self._is_main():
            logger.info("Validation: %s", results)

        return results

    # =====================================================
    # HELPERS
    # =====================================================

    def _unwrap_model(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _is_main(self):
        return not self.distributed or self.distributed.is_main_process()

# =========================================================
# COMPAT: lightweight TrainerConfig dataclass
# =========================================================

from dataclasses import dataclass as _dataclass, field as _field
from typing import Any as _Any, Dict as _Dict, Optional as _Optional


@_dataclass
class TrainerConfig:
    config_path: str = ""
    monitor_metric: str = "val_loss"
    maximize_metric: bool = False
    extras: _Dict[str, _Any] = _field(default_factory=dict)
