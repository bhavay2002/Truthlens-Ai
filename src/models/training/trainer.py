from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.models.config.model_config import ModelConfigLoader
from src.models.training.training_utils import get_device, move_batch_to_device

from .training_step import TrainingStep
from .lossengine import LossEngine
from .monitorengine import MonitoringEngine
from .evaluationengine import EvaluationEngine
from .checkpointengine import CheckpointEngine
from .taskscheduler import TaskScheduler
from .distributedengine import DistributedEngine
from .experimenttracker import ExperimentTracker

logger = logging.getLogger(__name__)


# =========================================================
# TRAINER
# =========================================================

class Trainer:

    def __init__(
        self,
        config_path: str,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        training_step: TrainingStep,
        loss_engine: LossEngine,
        monitoring_engine: MonitoringEngine,
        evaluation_engine: EvaluationEngine,
        checkpoint_engine: Optional[CheckpointEngine] = None,
        task_scheduler: Optional[TaskScheduler] = None,
        distributed_engine: Optional[DistributedEngine] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):

        # =============================
        # CONFIG
        # =============================
        self.cfg = ModelConfigLoader.load_multitask_config(config_path)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.training_step = training_step
        self.loss_engine = loss_engine
        self.monitor = monitoring_engine
        self.evaluator = evaluation_engine
        self.checkpoint = checkpoint_engine
        self.scheduler = task_scheduler
        self.distributed = distributed_engine
        self.tracker = experiment_tracker

        # =============================
        # DEVICE
        # =============================
        self.device = get_device(self.cfg.encoder.device)
        self.model.to(self.device)

        self.global_step = 0

        # =============================
        # TRAINING CONFIG
        # =============================
        self.epochs = self.cfg.training.num_epochs
        self.grad_accum = self.cfg.training.gradient_accumulation_steps
        self.max_grad_norm = self.cfg.training.max_grad_norm
        self.early_patience = self.cfg.training.early_stopping_patience

        # =============================
        # EARLY STOPPING
        # =============================
        self.best_metric = None
        self.no_improve_epochs = 0

        # =============================
        # DISTRIBUTED INIT
        # =============================
        if self.distributed:
            self.distributed.initialize()
            self.model = self.distributed.wrap_model(self.model)

        # =============================
        # LOG CONFIG
        # =============================
        if self.tracker and self._is_main():
            self.tracker.log_params(asdict(self.cfg))

        logger.info("Trainer initialized (FINAL PRODUCTION VERSION)")

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self):

        for epoch in range(self.epochs):

            if self._is_main():
                logger.info(f"Epoch {epoch+1}/{self.epochs}")

            # ---- DDP sync ----
            if self.distributed and self.distributed.initialized:
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch()

            # =========================
            # VALIDATION
            # =========================
            if self.val_loader:

                if self.distributed and self.distributed.initialized:
                    self.distributed.barrier()

                val_metrics = self.evaluate()
                val_loss = val_metrics.get("val_loss")

                # ---- Early stopping ----
                if val_loss is not None:

                    if self.best_metric is None or val_loss < self.best_metric:
                        self.best_metric = val_loss
                        self.no_improve_epochs = 0
                    else:
                        self.no_improve_epochs += 1

                    if self.no_improve_epochs >= self.early_patience:
                        if self._is_main():
                            logger.warning("Early stopping triggered")
                        break

                # ---- Logging ----
                if self.tracker and self._is_main():
                    self.tracker.log_metrics(val_metrics, step=self.global_step)

                # ---- Checkpoint ----
                if self.checkpoint and self._is_main():
                    self.checkpoint.save(
                        step=self.global_step,
                        model=self._unwrap_model(),
                        metrics=val_metrics,
                        tag="epoch",
                    )

        # =============================
        # CLEANUP
        # =============================
        if self.tracker and self._is_main():
            self.tracker.finish()

        if self.distributed:
            self.distributed.cleanup()

    # =====================================================
    # TRAIN EPOCH
    # =====================================================

    def _train_epoch(self):

        self.model.train()

        for batch in self.train_loader:

            # -------------------------
            # TASK SCHEDULER
            # -------------------------
            if self.scheduler:
                task = self.scheduler.next_task()
                batch = self._filter_batch(batch, task)

            batch = move_batch_to_device(batch, self.device)

            # -------------------------
            # FORWARD
            # -------------------------
            with torch.cuda.amp.autocast(enabled=self.training_step.use_amp):

                outputs = self.model(**batch)

                # 🔥 BALANCER HOOK (before backward)
                self.loss_engine.on_before_backward(
                    outputs,
                    batch,
                    shared_parameters=self._unwrap_model().parameters(),
                )

                total_loss, task_losses = self.loss_engine.compute(
                    outputs,
                    batch,
                )

            # -------------------------
            # SKIP EMPTY BATCH
            # -------------------------
            if getattr(self.loss_engine.loss_module, "last_active_heads", 1) == 0:
                continue

            # -------------------------
            # GRAD ACCUMULATION
            # -------------------------
            loss = total_loss / self.grad_accum
            self.training_step.backward(loss)

            # -------------------------
            # OPTIMIZER STEP
            # -------------------------
            if (self.global_step + 1) % self.grad_accum == 0:

                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )

                self.training_step.step()
                self.training_step.zero_grad()

                # 🔥 BALANCER STEP HOOK
                self.loss_engine.on_step_end()

            # -------------------------
            # POST BACKWARD HOOK
            # -------------------------
            self.loss_engine.on_after_backward()

            # -------------------------
            # DISTRIBUTED SYNC
            # -------------------------
            if self.distributed and self.distributed.initialized:
                loss_value = self.distributed.all_reduce(total_loss.detach()).item()
            else:
                loss_value = float(total_loss.detach())

            self.global_step += 1

            # -------------------------
            # MONITORING
            # -------------------------
            monitor_metrics = self.monitor.update(
                {
                    "loss": loss_value,
                    "task_losses": outputs.get("task_losses", {}),
                },
                model=self._unwrap_model(),
            )

            # -------------------------
            # HEALTH CONTROL
            # -------------------------
            if monitor_metrics.get("action") == "reduce_lr":
                self._reduce_lr()

            if monitor_metrics.get("health", 1.0) < 0.2:
                if self._is_main():
                    logger.error("Training unstable → stopping")
                raise RuntimeError("Training collapsed")

            # -------------------------
            # LOGGING
            # -------------------------
            if self.global_step % 50 == 0 and self._is_main():

                log_data = {
                    "train_loss": loss_value,
                    **monitor_metrics,
                }

                logger.info(f"Step {self.global_step} | {log_data}")

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
    # EVALUATE
    # =====================================================

    def evaluate(self) -> Dict[str, Any]:

        if not self.val_loader:
            return {}

        model = self._unwrap_model()
        results = self.evaluator.evaluate(model, self.val_loader)

        if self._is_main():
            logger.info(f"Validation: {results}")

        return results

    # =====================================================
    # HELPERS
    # =====================================================

    def _unwrap_model(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _reduce_lr(self):
        optimizer = self.training_step.optimizer
        for g in optimizer.param_groups:
            g["lr"] *= 0.5

        if self._is_main():
            logger.warning("LR reduced due to instability")

    def _filter_batch(self, batch: Dict[str, Any], task: str):

        if "labels" not in batch:
            return batch

        labels = batch["labels"]

        if isinstance(labels, dict) and task in labels:
            return {
                **batch,
                "labels": {task: labels[task]},
            }

        return batch

    def _is_main(self):
        return not self.distributed or self.distributed.is_main_process()