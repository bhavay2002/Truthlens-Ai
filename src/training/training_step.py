# src/models/training/training_step.py

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch
import torch.nn as nn

from src.training.training_utils import compute_grad_norm, get_current_lr

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class TrainingStepConfig:
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    skip_nan_loss: bool = True


# =========================================================
# ACTION ENUM (NEW 🔥)
# =========================================================

class TrainAction:
    NONE = "none"
    REDUCE_LR = "reduce_lr"
    STOP = "stop_training"
    CHECK_DATALOADER = "check_dataloader"


# =========================================================
# CORE ENGINE
# =========================================================

class TrainingStep:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_engine,
        monitor,
        tracker=None,
        task_scheduler=None,
        instrumentation=None,
        config: TrainingStepConfig = TrainingStepConfig(),
        device: Optional[str] = None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_engine = loss_engine
        self.monitor = monitor
        self.tracker = tracker
        self.task_scheduler = task_scheduler
        self.instrumentation = instrumentation
        self.config = config

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        self.use_amp = config.use_mixed_precision and self.device.type == "cuda"

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self._last_time = time.time()

        logger.info("TrainingStep initialized | AMP=%s", self.use_amp)

    # =====================================================
    # RUN STEP
    # =====================================================

    def run(self, batch: Dict[str, Any], step: int) -> Dict[str, Any]:

        self.model.train()
        batch = self._move_batch(batch)

        # -------------------------
        # TASK SCHEDULING
        # -------------------------

        task = None
        if self.task_scheduler:
            task = self.task_scheduler.next_task()
            batch = self._filter_batch(batch, task)

        # -------------------------
        # FORWARD + LOSS
        # -------------------------

        with torch.cuda.amp.autocast(enabled=self.use_amp):

            outputs = self.model(**batch)

            total_loss, task_losses = self.loss_engine.compute(
                outputs,
                batch,
                shared_parameters=self.model.parameters(),
            )

        # -------------------------
        # LOSS VALIDATION
        # -------------------------

        if not torch.isfinite(total_loss):

            if self.config.skip_nan_loss:
                logger.warning("Skipping step due to NaN loss")
                self.optimizer.zero_grad(set_to_none=True)
                return {"loss": None, "skipped": True}

            raise RuntimeError(f"Non-finite loss: {total_loss.item()}")

        # -------------------------
        # TASK SCHEDULER UPDATE
        # -------------------------

        if self.task_scheduler and task_losses:
            self.task_scheduler.update_losses(
                {k: float(v.detach()) for k, v in task_losses.items()}
            )

        # -------------------------
        # BACKWARD
        # -------------------------

        loss = total_loss / self.config.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # -------------------------
        # GRAD NORM
        # -------------------------

        grad_norm = compute_grad_norm(self.model)

        # -------------------------
        # OPTIMIZER STEP
        # -------------------------

        should_step = (
            (step + 1) % self.config.gradient_accumulation_steps == 0
        )

        if should_step:

            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            if self.config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            # AMP SAFE STEP
            if self.use_amp:
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Detect overflow
                if self.scaler.get_scale() < prev_scale:
                    logger.warning("Gradient overflow detected, step skipped")
            else:
                self.optimizer.step()

            # Scheduler
            if self.scheduler:
                try:
                    self.scheduler.step()
                except TypeError:
                    self.scheduler.step(float(total_loss.detach()))

            self.optimizer.zero_grad(set_to_none=True)

            # hooks
            self.loss_engine.on_after_backward()
            self.loss_engine.on_step_end()

        # -------------------------
        # THROUGHPUT
        # -------------------------

        now = time.time()
        duration = now - self._last_time
        self._last_time = now

        batch_size = self._infer_batch_size(batch)
        throughput = batch_size / duration if duration > 0 else None

        # -------------------------
        # MONITORING
        # -------------------------

        monitor_metrics = self.monitor.update(
            {"loss": float(total_loss.detach())},
            model=self.model,
            batch_size=batch_size,
        )

        # -------------------------
        # DEBUG ENGINE
        # -------------------------

        debug_info = {}

        if self.instrumentation:
            debug_info = self.instrumentation.step(
                losses=task_losses,
                total_loss=total_loss,
                model=self.model,
                shared_params=self.model.parameters(),
                logits=outputs.get("logits") if isinstance(outputs, dict) else None,
                throughput=throughput,
            )

        # -------------------------
        # ACTION HANDLING
        # -------------------------

        action = debug_info.get("debug/action", TrainAction.NONE)

        if action == TrainAction.STOP:
            raise RuntimeError("Training stopped by AutoDebugEngine")

        elif action == TrainAction.REDUCE_LR:
            self._reduce_lr()

        elif action == TrainAction.CHECK_DATALOADER:
            logger.warning("Potential dataloader bottleneck detected")

        # fallback monitor
        if monitor_metrics.get("monitor/action") == TrainAction.REDUCE_LR:
            self._reduce_lr()

        # -------------------------
        # LOGGING
        # -------------------------

        log_data = {
            "train/loss": float(total_loss.detach()),
            "train/grad_norm": grad_norm,
            "train/lr": get_current_lr(self.optimizer),
            "train/throughput": throughput,
            **monitor_metrics,
            **debug_info,
        }

        if self.tracker:
            self.tracker.log_metrics(log_data, step=step)

        # -------------------------
        # OUTPUT
        # -------------------------

        return {
            "loss": loss.detach(),
            "raw_loss": total_loss.detach(),
            "task_losses": task_losses,
            "grad_norm": grad_norm,
            "throughput": throughput,
            "skipped": False,
            **monitor_metrics,
            **debug_info,
        }

    # =====================================================
    # UTILS
    # =====================================================

    def _move_batch(self, batch):
        return {
            k: v.to(self.device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }

    def _infer_batch_size(self, batch):
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.size(0)
        return 1

    def _filter_batch(self, batch, task):
        if "labels" not in batch:
            return batch

        labels = batch["labels"]

        if isinstance(labels, dict) and task in labels:
            return {
                **batch,
                "labels": {task: labels[task]},
            }

        return batch

    def _reduce_lr(self):
        for g in self.optimizer.param_groups:
            g["lr"] *= 0.5
        logger.warning("LR reduced due to instability")