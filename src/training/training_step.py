from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch
import torch.nn as nn

from src.training.training_utils import compute_grad_norm, get_current_lr

# ✅ NEW: observability
from src.monitoring.feature_logger import (
    log_feature_stats,
    log_feature_summary,
)

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
# ACTION ENUM
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
    # FEATURE HELPER (NEW)
    # =====================================================

    def _tensor_to_feature_dict(self, batch: Dict[str, Any], max_items: int = 50):
        """
        Convert tensor batch into small numeric feature dict for logging.
        Prevents huge logs.
        """
        feature_dict = {}

        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dtype in (torch.float32, torch.float64):
                flat = v.detach().flatten()
                for i in range(min(len(flat), max_items)):
                    feature_dict[f"{k}_{i}"] = float(flat[i])

        return feature_dict

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
        # 🔍 FEATURE OBSERVABILITY (NEW)
        # -------------------------

        if step % 50 == 0:  # avoid slowdown
            try:
                feature_dict = self._tensor_to_feature_dict(batch)

                if feature_dict:
                    log_feature_stats(
                        feature_dict,
                        task=task or "default",
                        step=step,
                    )

                    log_feature_summary(
                        feature_dict,
                        task=task or "default",
                        step=step,
                    )

            except Exception as e:
                logger.warning("Feature logging failed: %s", e)

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
        # OPTIMIZER STEP  (BUG-5: unscale BEFORE measuring grad_norm,
        # otherwise the AMP loss-scale (~6.5e4) is baked into every
        # logged grad_norm and instrumentation flags every step as
        # 'exploding gradients'. Also gated on the accumulation
        # boundary so partial micro-batches don't unscale prematurely.)
        # -------------------------

        should_step = (
            (step + 1) % self.config.gradient_accumulation_steps == 0
        )

        grad_norm: Optional[float] = None
        scaler_stepped_ok = True  # tracks whether the scaler actually stepped

        if should_step:

            if self.use_amp:
                self.scaler.unscale_(self.optimizer)

            grad_norm = compute_grad_norm(self.model)

            if self.config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

            if self.use_amp:
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scaler.get_scale() < prev_scale:
                    scaler_stepped_ok = False
                    logger.warning("Gradient overflow detected, step skipped")
            else:
                self.optimizer.step()

            # BUG-6 (partial fix): only advance the scheduler when the
            # optimizer actually stepped, so AMP overflow doesn't drift
            # the LR schedule.
            if self.scheduler and scaler_stepped_ok:
                try:
                    self.scheduler.step()
                except TypeError:
                    self.scheduler.step(float(total_loss.detach()))

            self.optimizer.zero_grad(set_to_none=True)

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

    def _reduce_lr(self, factor: float = 0.5):
        # BUG-6: a LambdaLR (and most functional schedulers) compute
        # ``g["lr"] = base_lr * lambda(step)`` on every ``scheduler.step()``.
        # Mutating only ``g["lr"]`` is therefore overwritten on the very
        # next scheduler step and the spike-recovery action becomes a
        # no-op. We must reduce the scheduler's ``base_lrs`` so the new
        # rate persists across subsequent scheduler steps.
        for g in self.optimizer.param_groups:
            g["lr"] *= factor

        if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [
                b * factor for b in self.scheduler.base_lrs
            ]

        logger.warning(
            "LR reduced (factor=%.3f) due to instability", float(factor),
        )