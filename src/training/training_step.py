from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Any, Dict

import torch
import torch.nn as nn

from src.training.training_utils import (
    compute_grad_norm,
    get_current_lr,
    move_batch_to_device,
)

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

    # CFG-2: factor used by ``_reduce_lr`` when the spike / health detectors
    # (instrumentation engine OR monitor engine) raise ``REDUCE_LR``. Was
    # previously hardcoded as ``0.5`` in two separate sites; centralising
    # here makes it tunable from the config layer (and matches
    # ``LRSchedulerConfig.spike_lr_scale`` semantics).
    spike_lr_scale: float = 0.5


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

        # GPU-1: the model is moved to its final device ONCE in
        # ``create_trainer_fn`` (BEFORE ``build_optimizer``), so the
        # optimizer holds parameters that already live on the correct
        # device. The previous ``self.model.to(self.device)`` here was the
        # SECOND of three moves (Trainer.__init__ also did one, and
        # DistributedEngine.wrap_model does a third) — and crucially it
        # happened AFTER the optimizer was constructed, leaving the
        # optimizer with stale parameter references on the original device.
        # That's the classic "expected all tensors to be on the same
        # device" failure at first ``optimizer.step()``. Validate that the
        # model is already on the expected device and surface a clear
        # error if not, instead of silently re-moving it.
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device

        if model_device != self.device:
            logger.warning(
                "GPU-1: TrainingStep received model on %s but expected %s; "
                "falling back to in-place move (optimizer may hold stale "
                "parameter refs — prefer moving the model BEFORE building "
                "the optimizer in create_trainer_fn).",
                model_device,
                self.device,
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

        PERF-5: Original implementation called ``float(flat[i])`` inside a
        Python loop, which forces *one host-device sync per element* (up to
        ``max_items × num_keys`` syncs per logging step). The new
        implementation slices on-device first and then performs **a single**
        ``.cpu().tolist()`` per tensor — at most one sync per key.
        """
        feature_dict: Dict[str, float] = {}

        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                continue
            if not v.dtype.is_floating_point:
                continue

            flat = v.detach().flatten()[:max_items].cpu().tolist()
            feature_dict.update({f"{k}_{i}": float(x) for i, x in enumerate(flat)})

        return feature_dict

    # =====================================================
    # RUN STEP
    # =====================================================

    def run(
        self,
        batch: Dict[str, Any],
        step: int,
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        MT-3: ``dry_run=True`` validates forward + loss + backward without
        mutating any persistent training state — specifically:
          * ``task_scheduler.next_task`` is NOT called (round-robin index
            and adaptive EMA stay frozen)
          * ``optimizer.step`` / ``scaler.step`` / ``scaler.update`` /
            ``scheduler.step`` are skipped (no parameter updates, no LR
            decay tick, no AMP loss-scale advance)
          * ``loss_engine.on_after_backward`` / ``on_step_end`` are skipped
            (balancer counters stay frozen)
          * ``monitor.update`` and ``instrumentation.step`` are skipped
            (EMAs / failure memory / spike detector stay frozen)
        Gradients ARE computed (so the backward path is exercised) and
        then immediately zeroed so the next real step starts clean. This
        is the contract the sanity check needs to be both safety-checking
        AND reproducibility-preserving.
        """

        # EDGE-CASE (section 9): the rest of the run loop unpacks ``batch``
        # via ``model(**batch)`` and indexes ``batch["labels"]`` inside
        # ``LossEngine.compute``. A list/tuple batch (e.g. a default
        # ``DataLoader`` collate that doesn't return a dict) would crash
        # at ``model(**batch)`` with a ``TypeError`` whose message points
        # at the model — masking the real cause (a custom dataset that
        # forgot to return a dict). Surface a clear error here at the
        # contract boundary so the dataset author sees the real issue.
        if not isinstance(batch, dict):
            raise TypeError(
                "TrainingStep expects ``batch`` to be a dict (got "
                f"{type(batch).__name__}). Datasets must return dicts; "
                "fix the dataset / collate_fn rather than reshaping here."
            )

        self.model.train()
        batch = self._move_batch(batch)

        # -------------------------
        # TASK SCHEDULING
        # -------------------------
        #
        # LOSS-2: We DELIBERATELY do not call ``_filter_batch`` here.
        # The original code filtered the labels dict to a single task per
        # step, which (a) wasted the joint-encoder forward pass — the model
        # is multi-task and produces logits for every head regardless — and
        # (b) starved the adaptive task scheduler of all but one task's
        # loss signal, collapsing it to round-robin behaviour in disguise.
        # MultiTaskLoss already masks per-task via its label dict, so the
        # full batch can flow through unchanged.
        #
        # MT-3: in dry-run we DO NOT call ``next_task`` because that
        # advances the round-robin index — the real first training step
        # would then start at index 1 instead of 0, silently desyncing
        # the task schedule from any reproducibility seed.

        task = None
        if self.task_scheduler and not dry_run:
            task = self.task_scheduler.next_task()

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

        # EDGE-CASE (section 9, NaN labels): ``LossEngine.compute`` /
        # ``MultiTaskLoss.forward`` raise ``RuntimeError`` on non-finite
        # aggregates. The previous implementation only honoured
        # ``skip_nan_loss`` for the FINAL ``torch.isfinite(total_loss)``
        # check below — meaning NaN labels (which propagate through
        # cross-entropy / BCE before the aggregate is built) escaped the
        # quarantine path and crashed the run despite ``skip_nan_loss=True``.
        # Wrap the whole forward+loss block so the same skip semantics
        # apply uniformly.
        try:
            with torch.cuda.amp.autocast(enabled=self.use_amp):

                outputs = self.model(**batch)

                total_loss, task_losses = self.loss_engine.compute(
                    outputs,
                    batch,
                    shared_parameters=self.model.parameters(),
                )
        except RuntimeError as e:
            if self.config.skip_nan_loss:
                logger.warning(
                    "Skipping step due to RuntimeError in forward/loss "
                    "(likely NaN labels or non-finite logits): %s",
                    e,
                )
                self.optimizer.zero_grad(set_to_none=True)
                return {"loss": None, "skipped": True}
            raise

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
        #
        # MT-3: skip in dry-run so the adaptive EMA isn't poisoned by a
        # one-shot sanity loss before the first real training step.
        # MT-4: ``task_losses`` here is now the RAW per-task loss dict
        # (the second element of MultiTaskLoss.forward's return tuple),
        # which is exactly what the adaptive scheduler's softmax-of-EMA
        # expects — the previous weighted-and-normalized values would
        # have skewed the softmax across tasks with different weights.

        if self.task_scheduler and task_losses and not dry_run:
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

            # REC-3: ``compute_grad_norm`` and ``clip_grad_norm_`` BOTH
            # iterate every parameter and compute the same total L2 norm
            # — and ``instrumentation.step`` calls ``GradTracker.update``
            # which does it a THIRD time (and after ``zero_grad`` clears
            # the gradients, so it would see zeros). Use ``clip_grad_norm_``
            # alone when clipping is enabled (it returns the pre-clip norm
            # — exactly what we want to log) and fall back to
            # ``compute_grad_norm`` only when clipping is disabled. The
            # resulting ``grad_norm`` is then forwarded to instrumentation
            # via ``cached_grad_norm`` so it doesn't redo the work.
            if self.config.max_grad_norm:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                )
            else:
                grad_norm = compute_grad_norm(self.model)

            # MT-3: in dry-run validate forward + loss + backward only.
            # Skip the optimizer / scaler / scheduler / balancer mutations
            # so the persistent training state is preserved. Gradients are
            # zeroed below so the first real step starts clean.
            if not dry_run:

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

                self.loss_engine.on_after_backward()
                self.loss_engine.on_step_end()

            self.optimizer.zero_grad(set_to_none=True)

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
        #
        # MT-3: dry-run skips the monitor entirely so its EMAs / spike
        # detector / health score don't carry sanity-check noise into
        # the first real training step.

        if dry_run:
            monitor_metrics: Dict[str, Any] = {}
        else:
            monitor_metrics = self.monitor.update(
                {"loss": float(total_loss.detach())},
                model=self.model,
                batch_size=batch_size,
            )

        # -------------------------
        # DEBUG ENGINE
        # -------------------------
        #
        # MT-3: skip in dry-run for the same reason as the monitor.
        # REC-3: when we already have ``grad_norm`` from clip_grad_norm_,
        # pass it through as ``cached_grad_norm`` so the instrumentation's
        # GradTracker doesn't re-iterate every parameter to recompute the
        # same value (and on should_step iterations would otherwise see
        # zeroed-out gradients after ``optimizer.zero_grad``).

        debug_info = {}

        if self.instrumentation and not dry_run:
            debug_info = self.instrumentation.step(
                losses=task_losses,
                total_loss=total_loss,
                model=self.model,
                shared_params=self.model.parameters(),
                logits=outputs.get("logits") if isinstance(outputs, dict) else None,
                throughput=throughput,
                cached_grad_norm=grad_norm,
            )

        # -------------------------
        # ACTION HANDLING
        # -------------------------

        action = debug_info.get("debug/action", TrainAction.NONE)

        # LOSS-1: Both the instrumentation engine and the monitor engine can
        # raise REDUCE_LR in the same step. The original code fired
        # ``_reduce_lr`` once per source, halving the LR TWICE on a spike
        # that both detectors caught (a 4× drop instead of the configured
        # 2×). De-duplicate per step.
        lr_reduced_this_step = False

        if action == TrainAction.STOP:
            raise RuntimeError("Training stopped by AutoDebugEngine")

        elif action == TrainAction.REDUCE_LR:
            self._reduce_lr()
            lr_reduced_this_step = True

        elif action == TrainAction.CHECK_DATALOADER:
            logger.warning("Potential dataloader bottleneck detected")

        if (
            not lr_reduced_this_step
            and monitor_metrics.get("monitor/action") == TrainAction.REDUCE_LR
        ):
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

        # MT-3: dry-run does not pollute the experiment tracker with a
        # one-shot sanity row that would shift step-indexed plots by 1.
        if self.tracker and not dry_run:
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
        # GPU-2: ``non_blocking=True`` is silently a no-op unless the
        # source tensor is in pinned host memory. The previous inline
        # comprehension passed ``non_blocking=True`` unconditionally,
        # advertising async H2D copies that never actually happened on
        # un-pinned tensors (e.g. CPU-only runs, or any DataLoader built
        # with ``pin_memory=False``). Delegate to the shared utility that
        # gates ``non_blocking`` on the per-tensor ``is_pinned()`` check.
        return move_batch_to_device(batch, self.device, non_blocking=True)

    def _infer_batch_size(self, batch):
        for v in batch.values():
            if isinstance(v, torch.Tensor):
                return v.size(0)
        return 1

    # NOTE: ``_filter_batch`` was removed (LOSS-2). The model is multi-task
    # and produces logits for every head from a single forward pass; the
    # MultiTaskLoss orchestrator masks per-task via the labels dict, so
    # there is no value in pre-filtering the batch.

    def _reduce_lr(self, factor: Optional[float] = None):
        # CFG-2: factor is sourced from ``TrainingStepConfig.spike_lr_scale``
        # by default rather than the previous hardcoded ``0.5``. Callers may
        # still pass an explicit override.
        if factor is None:
            factor = float(self.config.spike_lr_scale)

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