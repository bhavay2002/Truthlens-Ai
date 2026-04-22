"""
File Name: trainer.py
Module: models.training
Description:
    Implements the training engine for TruthLens models. This module provides a
    reusable Trainer abstraction responsible for coordinating the full training
    lifecycle including forward passes, backpropagation, gradient accumulation,
    optimizer steps, scheduler updates, checkpointing hooks, and metric logging.

    The trainer is framework-agnostic with respect to the model architecture and
    supports both single-task and multi-task models that return either dictionaries
    or structured outputs.

    Designed for research reproducibility and production ML pipelines.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    torch.optim
Inputs:
    Model
    Training DataLoader
    Validation DataLoader
Outputs:
    Training history and trained model parameters
"""
from __future__ import annotations

import inspect
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..checkpointing.checkpoint_manager import CheckpointManager
from src.training.checkpointing import (
    list_checkpoints as list_training_checkpoints,
    resume_training as resume_training_checkpoint,
)
from src.utils import get_device, move_to_device

logger = logging.getLogger(__name__)


def _configure_tf32() -> None:
    """Enable TF32 + FP16 reduced-precision reduction when CUDA is available.

    Invoked inside Trainer.__init__ so importing this module has no global
    side effects on numerical precision.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        # Tensor Core friendly FP16 reductions (no measurable accuracy loss
        # for transformer training).
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


# ---------------------------------------------------------
# UTIL
# ---------------------------------------------------------

def _get_autocast_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

@dataclass
class TrainerConfig:
    epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: Optional[str] = None
    log_every_steps: int = 100
    checkpoint_dir: Optional[str] = None
    checkpoint_every_steps: int = 0
    use_amp: Optional[bool] = None
    amp_dtype: Optional[str] = None
    # Run validation every N epochs (default 2). Saves 10-20% wall time.
    validate_every_n_epochs: int = 1


# ---------------------------------------------------------
# TRAINER
# ---------------------------------------------------------

class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: TrainerConfig,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError("model must be torch.nn.Module")

        _configure_tf32()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # Device
        self.device = (
            torch.device(config.device)
            if config.device
            else get_device(prefer_gpu=True)
        )

        self.model.to(self.device)

        #  torch.compile (CUDA-only, idempotent — C2)
        # Enabled by default on CUDA: pays off ~10-20% on Ampere+ (A100/L4/H100).
        # Disable on T4 (or for debugging) with TRUTHLENS_TORCH_COMPILE=0.
        if (
            os.environ.get("TRUTHLENS_TORCH_COMPILE", "1") == "1"
            and hasattr(torch, "compile")
            and self.device.type == "cuda"
            and not getattr(self.model, "_dynamo_compiled", False)
        ):
            try:
                # dynamic=True keeps recompiles bounded when the bucket
                # sampler emits varying sequence lengths each batch.
                self.model = torch.compile(self.model, dynamic=True)
                try:
                    self.model._dynamo_compiled = True
                except Exception:
                    pass
                logger.info("torch.compile enabled")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # AMP Setup
        if self.config.use_amp is None:
            self.use_amp = self.device.type == "cuda"
        else:
            self.use_amp = bool(self.config.use_amp)

        if self.config.amp_dtype:
            if self.config.amp_dtype.lower() == "bf16":
                self.autocast_dtype = torch.bfloat16
            elif self.config.amp_dtype.lower() == "fp16":
                self.autocast_dtype = torch.float16
            else:
                self.autocast_dtype = _get_autocast_dtype()
        else:
            self.autocast_dtype = _get_autocast_dtype()
        if self.device.type == "cuda":
            self.autocast_device_type = "cuda"
        else:
            self.autocast_device_type = "cpu"
            self.use_amp = False

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.autocast_dtype == torch.float16
        )

        # Forward signature caching
        try:
            sig = inspect.signature(self.model.forward)
            self._forward_params = set(sig.parameters.keys()) - {"self"}
            self._forward_accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
        except Exception:
            self._forward_params = None
            self._forward_accepts_kwargs = True

        self.global_step = 0

        # Checkpoint manager
        self.checkpoint_manager: Optional[CheckpointManager] = None
        if config.checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(Path(config.checkpoint_dir))
            self._attempt_resume()

        logger.info("Trainer initialized on device %s", self.device)

    # ---------------------------------------------------------

    def _validate_batch_labels(self, batch: Dict[str, Any]) -> None:
        """Fail fast on out-of-range multiclass labels before forward/loss."""
        if not isinstance(batch, dict):
            return

        # Default to known TruthLens ranges; model attributes override when present.
        label_specs = (
            ("labels_bias", int(getattr(self.model, "NUM_BIAS", 2))),
            ("labels_ideology", int(getattr(self.model, "NUM_IDEOLOGY", 5))),
            ("labels_propaganda", int(getattr(self.model, "NUM_PROPAGANDA", 2))),
        )
        for key, num_classes in label_specs:
            labels = batch.get(key)
            if not torch.is_tensor(labels):
                continue
            if labels.numel() == 0:
                continue
            lo = int(labels.min().item())
            hi = int(labels.max().item())
            if lo < 0 or hi >= num_classes:
                raise RuntimeError(
                    f"Invalid label range for {key}: min={lo} max={hi} "
                    f"(expected [0,{num_classes - 1}])."
                )

    # ---------------------------------------------------------

    @staticmethod
    def _label_distribution_summary(batch: Dict[str, Any]) -> str:
        """Compact per-batch label distribution for diagnostics."""
        if not isinstance(batch, dict):
            return "labels=unavailable"
        parts: List[str] = []
        for key in ("labels_bias", "labels_ideology", "labels_propaganda"):
            labels = batch.get(key)
            if not torch.is_tensor(labels) or labels.numel() == 0:
                continue
            uniq, counts = torch.unique(labels.detach().cpu(), return_counts=True)
            items = ",".join(
                f"{int(u.item())}:{int(c.item())}" for u, c in zip(uniq, counts)
            )
            parts.append(f"{key}[{items}]")
        return " ".join(parts) if parts else "labels=unavailable"

    # ---------------------------------------------------------

    @staticmethod
    def _task_loss_dominance_summary(task_losses: Dict[str, Any]) -> Optional[str]:
        """Return summary when one task dominates weighted loss."""
        if not isinstance(task_losses, dict) or not task_losses:
            return None
        vals = {
            name: float(loss.detach().item())
            for name, loss in task_losses.items()
            if torch.is_tensor(loss)
        }
        if len(vals) < 2:
            return None
        total = sum(abs(v) for v in vals.values())
        if total <= 1e-12:
            return None
        top_name, top_value = max(vals.items(), key=lambda kv: abs(kv[1]))
        ratio = abs(top_value) / total
        threshold = float(os.environ.get("TRUTHLENS_TASK_DOMINANCE_RATIO", "0.8"))
        if ratio >= threshold:
            parts = " ".join(f"{k}={v:.4f}" for k, v in vals.items())
            return f"dominant={top_name} share={ratio:.2f} threshold={threshold:.2f} | {parts}"
        return None

    # ---------------------------------------------------------

    def _attempt_resume(self):

        checkpoint_root = Path(self.config.checkpoint_dir)
        available = list_training_checkpoints(checkpoint_root)

        if not available:
            return

        latest = available[-1]

        try:
            state = resume_training_checkpoint(
                self.model,
                checkpoint_dir=latest,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )

            self.global_step = int(state.get("start_step", 0) or 0)
            # C4: scheduler state is restored inside resume_training_checkpoint.
            # The previously-duplicated branch here read keys that resume_training
            # never returned and was dead code — removed.
            logger.info("Resumed training from %s", latest)

        except Exception as exc:
            # An on-disk checkpoint exists but failed to load. Silently
            # restarting from scratch (the previous behaviour) hides
            # corruption / schema-drift bugs. Surface them loudly unless
            # the operator opts out via TRUTHLENS_ALLOW_RESUME_FAIL=1.
            logger.error(
                "Checkpoint resume FAILED for %s: %s", latest, exc, exc_info=True,
            )
            if os.environ.get("TRUTHLENS_ALLOW_RESUME_FAIL", "0") != "1":
                raise RuntimeError(
                    f"Refusing to silently restart: checkpoint at {latest} "
                    f"could not be resumed ({exc}). Set "
                    f"TRUTHLENS_ALLOW_RESUME_FAIL=1 to override."
                ) from exc
            logger.warning("Checkpoint resume skipped (override active)")

    # ---------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")

        validate_every = max(1, int(getattr(self.config, "validate_every_n_epochs", 1)))

        # ---------------------------------------------------------
        # Interrupt handling (Lightning AI / preemption / Ctrl+C)
        # Flush a checkpoint before the process dies.
        # ---------------------------------------------------------
        previous_handlers: Dict[int, Any] = {}
        interrupt_state = {"handled": False}

        def _handle_interrupt(signum, _frame):
            if interrupt_state["handled"]:
                return
            interrupt_state["handled"] = True
            logger.warning(
                "Interrupt %s received - saving emergency checkpoint at step %d",
                signum, self.global_step,
            )
            self._save_emergency_checkpoint()
            sys.exit(0)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[sig] = signal.signal(sig, _handle_interrupt)
            except (ValueError, OSError):
                # signal.signal only works in the main thread; skip otherwise.
                pass

        try:
            return self._train_loop(train_loader, val_loader, history, best_val, validate_every)
        finally:
            for sig, prev in previous_handlers.items():
                try:
                    signal.signal(sig, prev)
                except (ValueError, OSError):
                    pass

    def _train_loop(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        history: Dict[str, List[float]],
        best_val: float,
        validate_every: int,
    ) -> Dict[str, List[float]]:

        for epoch in range(self.config.epochs):

            logger.info("Epoch %d/%d", epoch + 1, self.config.epochs)

            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            val_loss: Optional[float] = None
            is_last_epoch = (epoch + 1) == self.config.epochs
            should_validate = (
                val_loader is not None
                and (((epoch + 1) % validate_every == 0) or is_last_epoch)
            )
            if should_validate:
                val_loss = self._validate_epoch(val_loader)
                history["val_loss"].append(val_loss)

            # C3: epoch-level checkpointing + best-model marker
            if self.checkpoint_manager is not None:
                metadata = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                try:
                    self.checkpoint_manager.save_checkpoint(
                        step=self.global_step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        metadata=metadata,
                        save_optimizer=True,
                        save_every=1,
                        deduplicate=False,
                    )
                    if val_loss is not None and val_loss < best_val:
                        best_val = val_loss
                        # Save the best model under a dedicated "best/" directory
                        # instead of injecting a fake step like 10**9+epoch into
                        # the step namespace (that produced the
                        # `checkpoint-1000000001` artifacts seen in earlier runs
                        # and corrupted step-based listing/sorting).
                        try:
                            self._save_best_model(epoch=epoch + 1, metadata=metadata)
                        except Exception as exc:
                            logger.error("Best-model save failed: %s", exc, exc_info=True)
                    self.checkpoint_manager.cleanup_old_checkpoints(max_checkpoints=3)
                except Exception as exc:
                    logger.error(
                        "Checkpoint save failed at epoch %d: %s",
                        epoch + 1, exc, exc_info=True,
                    )

        return history

    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Public checkpoint API (Lightning-AI style external control)
    # ---------------------------------------------------------

    def save_checkpoint(self, tag: Optional[str] = None) -> Optional[Path]:
        """Manually flush a checkpoint at the current global_step.

        ``tag`` is recorded in the checkpoint metadata (e.g. "interrupt",
        "manual", "preempt") so different save reasons are distinguishable
        on disk and during analysis.
        """
        if self.checkpoint_manager is None:
            logger.warning("save_checkpoint() called but no checkpoint_manager configured.")
            return None
        return self.checkpoint_manager.save_checkpoint(
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            metadata={
                "step": self.global_step,
                "epoch": None,
                "type": tag or "manual",
            },
            save_optimizer=True,
            save_every=1,
            deduplicate=False,
        )

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> Dict[str, Any]:
        """Restore model + optimizer + scheduler + global_step from ``path``.

        Accepts either a checkpoint directory (``checkpoint-1234/``) or the
        ``checkpoint.pt`` file directly.
        """
        if self.checkpoint_manager is None:
            self.checkpoint_manager = CheckpointManager(Path(self.config.checkpoint_dir or "."))

        state = self.checkpoint_manager.load_checkpoint(path)

        # Accept both the canonical "model_state_dict" key (audit-mandated)
        # and the legacy "model" key from older CheckpointManager payloads.
        model_state = (
            state.get("model_state_dict")
            or state.get("model")
            or state
        )
        target = self.model
        # torch.compile wraps the module; load into the original module.
        target = getattr(target, "_orig_mod", target)
        missing, unexpected = target.load_state_dict(model_state, strict=strict)
        if missing or unexpected:
            logger.warning(
                "load_checkpoint: missing=%d unexpected=%d", len(missing), len(unexpected),
            )
            # Output-contract audit (Doc 1, root cause #4): silent absence
            # of a task-head's weights is exactly what produced the
            # "no bias logits returned by model" warning in eval. Treat
            # any missing task-head weights as fatal — never let a
            # silent-load scenario reach training/eval.
            _task_head_prefixes = (
                "bias_head", "ideology_head", "propaganda_head",
                "narrative_head", "narrative_frame_head", "emotion_head",
            )
            _missing_heads = sorted({
                k.split(".", 1)[0]
                for k in missing
                if k.split(".", 1)[0] in _task_head_prefixes
            })
            if _missing_heads:
                raise RuntimeError(
                    f"Checkpoint missing task-head weights: {_missing_heads}. "
                    f"Resuming would silently disable these heads. "
                    f"Path: {path}"
                )

        opt_state = state.get("optimizer_state_dict") or state.get("optimizer")
        if opt_state is not None and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception as exc:
                logger.warning("Optimizer state restore failed: %s", exc)

        sch_state = state.get("scheduler_state_dict") or state.get("scheduler")
        if sch_state is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(sch_state)
            except Exception as exc:
                logger.warning("Scheduler state restore failed: %s", exc)

        # Restore AMP loss-scale so the first post-resume step doesn't
        # trigger a scale-search spike.
        scaler_state = state.get("scaler_state_dict") or state.get("scaler")
        if scaler_state is not None and getattr(self, "scaler", None) is not None:
            try:
                self.scaler.load_state_dict(scaler_state)
            except Exception as exc:
                logger.warning("Scaler state restore failed: %s", exc)

        self.global_step = int(state.get("step", self.global_step) or 0)
        logger.info("Loaded checkpoint from %s @ step %d", path, self.global_step)
        return state

    # ---------------------------------------------------------

    def _save_best_model(self, epoch: int, metadata: Dict[str, Any]) -> None:
        """Persist the current model under ``<checkpoint_dir>/best/``.

        Uses a dedicated subdirectory so the best-model marker never collides
        with the step-numbered checkpoint namespace (which previously produced
        `checkpoint-1000000001` artifacts and broke step-based sorting).
        """
        if self.checkpoint_manager is None:
            return
        best_dir = Path(self.config.checkpoint_dir) / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        best_file = best_dir / "checkpoint.pt"
        target = getattr(self.model, "_orig_mod", self.model)
        payload = {
            "step": self.global_step,
            "epoch": epoch,
            "model": {k: v.detach().cpu() for k, v in target.state_dict().items()},
            "metadata": {**metadata, "marker": "best"},
            "pytorch_version": torch.__version__,
        }
        tmp = best_file.with_suffix(best_file.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, best_file)
        logger.info(
            "[Best] Saved best-model checkpoint at epoch %d step %d -> %s",
            epoch, self.global_step, best_file,
        )

    def _save_emergency_checkpoint(self) -> None:
        """Best-effort checkpoint flush triggered by SIGINT / SIGTERM."""
        if self.checkpoint_manager is None:
            logger.warning("[Emergency Checkpoint] No checkpoint_manager configured; skipping.")
            return
        try:
            self.checkpoint_manager.save_checkpoint(
                step=self.global_step,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                metadata={
                    "step": self.global_step,
                    "epoch": None,
                    "type": "interrupt",
                },
                save_optimizer=True,
                save_every=1,
                deduplicate=False,
            )
            logger.info("[Emergency Checkpoint] Saved at step %d", self.global_step)
        except Exception as exc:  # noqa: BLE001 - we are about to exit
            logger.error("[Emergency Checkpoint] Failed: %s", exc, exc_info=True)

    def _train_epoch(self, dataloader: DataLoader) -> float:

        self.model.train()

        # Accumulate loss on-device to avoid per-step GPU→CPU sync.
        loss_accum = torch.zeros((), device=self.device, dtype=torch.float32)
        step_count = 0

        # Spike-batch detector: warn (don't skip) when raw_loss exceeds
        # 5x the running mean. Surfaces the "calm → spike → calm" pattern
        # the loss-stability audit asked us to instrument.
        _spike_ratio = float(os.environ.get("TRUTHLENS_SPIKE_RATIO", "5.0"))
        _hard_loss_cap = float(os.environ.get("TRUTHLENS_HARD_LOSS_CAP", "0"))
        _low_loss_floor = float(os.environ.get("TRUTHLENS_LOW_LOSS_FLOOR", "0.001"))
        _low_loss_tripwire = int(os.environ.get("TRUTHLENS_LOW_LOSS_TRIPWIRE", "8"))
        _single_class_warn_every = int(os.environ.get("TRUTHLENS_SINGLE_CLASS_WARN_EVERY", "20"))
        low_loss_streak = 0

        self.optimizer.zero_grad(set_to_none=True)

        step = -1  # M3: bind step in case dataloader is empty
        for step, batch in enumerate(dataloader):

            batch = self._move_batch_to_device(batch)
            self._validate_batch_labels(batch)
            if _single_class_warn_every > 0 and self.global_step % _single_class_warn_every == 0:
                for key in ("labels_bias", "labels_ideology", "labels_propaganda"):
                    labels = batch.get(key) if isinstance(batch, dict) else None
                    if torch.is_tensor(labels) and labels.numel() > 0:
                        if torch.unique(labels).numel() == 1:
                            logger.warning(
                                "Single-class batch detected at step %d | %s",
                                self.global_step,
                                self._label_distribution_summary(batch),
                            )
                            break

            with torch.autocast(
                device_type=self.autocast_device_type,
                dtype=self.autocast_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(**self._prepare_model_inputs(batch))
                raw_loss = self._extract_loss(outputs)
                loss = raw_loss / self.config.gradient_accumulation_steps

            if not torch.isfinite(raw_loss):
                # M6: poisoned grads from the in-progress accumulation window
                # would otherwise leak into the next optimizer.step. Reset.
                logger.error(
                    "NaN/Inf loss at step %d — resetting accumulation", step
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            # Optional hard fuse for finite-but-abnormal losses.
            # Disabled by default (0). Enable via TRUTHLENS_HARD_LOSS_CAP.
            if _hard_loss_cap > 0.0 and float(raw_loss.detach().item()) > _hard_loss_cap:
                _sig = "?"
                _ids = batch.get("input_ids") if isinstance(batch, dict) else None
                if torch.is_tensor(_ids) and _ids.numel() > 0:
                    _sig = int(_ids[0, :8].sum().item())
                logger.error(
                    "Abnormal loss guard triggered at step %d: raw_loss=%.4f > cap=%.4f "
                    "(batch_sig=%s). Skipping batch.",
                    self.global_step, float(raw_loss.detach().item()), _hard_loss_cap, _sig,
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            raw_loss_value = float(raw_loss.detach().item())
            if _low_loss_floor > 0.0 and raw_loss_value <= _low_loss_floor:
                low_loss_streak += 1
                _task_snapshot = ""
                if isinstance(outputs, dict):
                    _tl = outputs.get("task_losses") or outputs.get("loss_breakdown")
                    if isinstance(_tl, dict) and _tl:
                        _task_snapshot = " | " + " ".join(
                            f"{n}={float(v.detach().item()):.6f}"
                            for n, v in _tl.items()
                            if torch.is_tensor(v)
                        )
                logger.warning(
                    "Near-zero loss detected at step %d: raw_loss=%.6f "
                    "(streak=%d floor=%.6f) | %s%s",
                    self.global_step,
                    raw_loss_value,
                    low_loss_streak,
                    _low_loss_floor,
                    self._label_distribution_summary(batch),
                    _task_snapshot,
                )
                if _low_loss_tripwire > 0 and low_loss_streak >= _low_loss_tripwire:
                    raise RuntimeError(
                        "Consecutive near-zero losses exceeded tripwire "
                        f"({low_loss_streak} >= {_low_loss_tripwire}). "
                        "Potential data leakage, degenerate labels, or loss bypass."
                    )
            else:
                low_loss_streak = 0

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_accum = loss_accum + raw_loss.detach().to(loss_accum.dtype)
            step_count += 1
            self.global_step += 1

            # Spike detection on the raw (un-divided) loss. Compare to the
            # running mean so the threshold scales with the task difficulty.
            if step_count >= 10:
                _running_mean = float((loss_accum / step_count).detach().item())
                _raw = float(raw_loss.detach().item())
                if _running_mean > 0 and _raw > _spike_ratio * _running_mean:
                    # Per-task breakdown + sample input identifier so we can
                    # actually find which head and which batch is causing
                    # the spike (loss-stability audit §10).
                    _per_task = ""
                    if isinstance(outputs, dict):
                        _tl = outputs.get("task_losses") or outputs.get("loss_breakdown")
                        if isinstance(_tl, dict) and _tl:
                            _per_task = " | " + " ".join(
                                f"{n}={float(v.detach().item()):.3f}"
                                for n, v in _tl.items()
                                if torch.is_tensor(v)
                            )
                    _ids = batch.get("input_ids") if isinstance(batch, dict) else None
                    _sample_id = (
                        int(_ids[0, :8].sum().item())
                        if torch.is_tensor(_ids) and _ids.numel() > 0
                        else "?"
                    )
                    logger.warning(
                        "Loss spike at step %d: raw=%.4f vs running_mean=%.4f "
                        "(ratio=%.1fx) batch_sig=%s%s",
                        self.global_step, _raw, _running_mean,
                        _raw / max(_running_mean, 1e-9),
                        _sample_id, _per_task,
                    )
                if isinstance(outputs, dict):
                    _tl_dom = outputs.get("task_losses") or outputs.get("loss_breakdown")
                    _dom_msg = self._task_loss_dominance_summary(_tl_dom)
                    if _dom_msg is not None:
                        logger.warning(
                            "Task loss domination at step %d | %s",
                            self.global_step,
                            _dom_msg,
                        )

            # -------------------------------------------------
            # STEP CHECKPOINTING (every N global steps)
            # -------------------------------------------------
            if (
                self.checkpoint_manager is not None
                and self.config.checkpoint_every_steps > 0
                and self.global_step % self.config.checkpoint_every_steps == 0
            ):
                try:
                    self.checkpoint_manager.save_checkpoint(
                        step=self.global_step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        metadata={
                            "step": self.global_step,
                            "epoch": None,
                            "type": "step",
                        },
                        save_optimizer=True,
                        save_every=1,
                        deduplicate=False,
                    )
                    logger.info("[Checkpoint] Saved at step %d", self.global_step)
                    self.checkpoint_manager.cleanup_old_checkpoints(max_checkpoints=3)
                except Exception as exc:
                    logger.error(
                        "[Checkpoint] Failed at step %d: %s",
                        self.global_step, exc,
                    )

            if (step + 1) % self.config.gradient_accumulation_steps == 0:

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)

                # clip_grad_norm_ returns the pre-clip total norm; checking
                # its finiteness is an O(1) gradient-sanity probe that catches
                # the spike pattern the diagnostic flagged (3.5 → 0.02 swings)
                # without paying the cost of scanning every named parameter.
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                # Periodic grad-norm visibility: surfaces exploding-grad
                # patterns BEFORE they cause loss spikes.
                if self.global_step % max(1, self.config.log_every_steps) == 0:
                    try:
                        logger.info(
                            "step %d | grad_norm(pre-clip)=%.4f",
                            self.global_step, float(total_norm.detach().item()),
                        )
                    except Exception:
                        pass

                if not torch.isfinite(total_norm):
                    logger.warning(
                        "Non-finite grad norm at step %d (norm=%s) — "
                        "skipping optimizer step",
                        self.global_step, total_norm,
                    )
                    if self.scaler.is_enabled():
                        # Tell the scaler the step was skipped so its scale
                        # factor is adjusted instead of left in a stale state.
                        self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

            if (step + 1) % self.config.log_every_steps == 0:
                # Per-task breakdown surfaces multi-task imbalance / collapse
                # (one head dominating, another flat-lining at 0). Falls back
                # to the aggregate loss when the model isn't multi-task.
                task_losses = None
                if isinstance(outputs, dict):
                    task_losses = outputs.get("task_losses") or outputs.get("loss_breakdown")
                else:
                    task_losses = getattr(outputs, "task_losses", None)

                if isinstance(task_losses, dict) and task_losses:
                    parts = " ".join(
                        f"{name}={float(v.detach().item()):.4f}"
                        for name, v in task_losses.items()
                        if torch.is_tensor(v)
                    )
                    logger.info(
                        "step %d | loss %.6f | %s",
                        step + 1, float(raw_loss.detach().item()), parts,
                    )
                else:
                    logger.info(
                        "step %d | loss %.6f",
                        step + 1, float(raw_loss.detach().item()),
                    )

        # Final step fix — flush any partial accumulation window (M3)
        if step >= 0 and (step + 1) % self.config.gradient_accumulation_steps != 0:

            if self.scaler.is_enabled():
                self.scaler.unscale_(self.optimizer)

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            if self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            self.optimizer.zero_grad(set_to_none=True)

        mean_loss = (loss_accum / max(step_count, 1)).detach().item()
        return float(mean_loss)


    # Validation per-task spec: which label key corresponds to which output
    # head and how predictions are derived from logits. Multi-class heads use
    # argmax; multi-label heads use sigmoid > 0.5.
    _VAL_TASKS = (
        # (task_name, label_key, prediction_kind)
        ("bias",            "labels_bias",            "multiclass"),
        ("ideology",        "labels_ideology",        "multiclass"),
        ("propaganda",      "labels_propaganda",      "multiclass"),
        ("narrative",       "labels_narrative",       "multilabel"),
        ("narrative_frame", "labels_narrative_frame", "multilabel"),
        ("emotion",         "labels_emotion",         "multilabel"),
    )

    def _validate_epoch(self, dataloader: DataLoader) -> float:

        self.model.eval()

        loss_accum = torch.zeros((), device=self.device, dtype=torch.float32)
        step_count = 0

        # Per-task buffers (kept on CPU).
        logits_buf: Dict[str, list] = {name: [] for name, _, _ in self._VAL_TASKS}
        labels_buf: Dict[str, list] = {name: [] for name, _, _ in self._VAL_TASKS}

        with torch.no_grad():

            for batch in dataloader:

                batch = self._move_batch_to_device(batch)

                outputs = self.model(**self._prepare_model_inputs(batch))
                loss = self._extract_loss(outputs)

                loss_accum = loss_accum + loss.detach().to(loss_accum.dtype)
                step_count += 1

                # Collect full-epoch logits + labels (never batch-average metrics).
                if isinstance(outputs, dict):
                    for task_name, label_key, kind in self._VAL_TASKS:
                        head_out = outputs.get(task_name)
                        if not isinstance(head_out, dict):
                            continue
                        logits = head_out.get("logits")
                        labels = batch.get(label_key) if isinstance(batch, dict) else None
                        if logits is None or labels is None:
                            continue
                        logits_buf[task_name].append(logits.detach().cpu())
                        if kind == "multilabel":
                            labels_buf[task_name].append((labels > 0.5).int().detach().cpu())
                        else:
                            labels_buf[task_name].append(labels.detach().long().cpu())

        mean_loss = (loss_accum / max(step_count, 1)).detach().item()

        # Compute and log per-task metrics.
        try:
            import numpy as _np
            from sklearn.metrics import (
                accuracy_score as _acc,
                f1_score as _f1,
                roc_auc_score as _auc,
            )

            metric_parts = []
            self.last_val_metrics = {}
            for task_name, _label_key, kind in self._VAL_TASKS:
                if not logits_buf[task_name]:
                    continue
                logits_np = torch.cat(logits_buf[task_name], dim=0).numpy()
                labels_np = torch.cat(labels_buf[task_name], dim=0).numpy()

                task_metrics: Dict[str, float] = {}

                if kind == "multiclass":
                    probs_np = torch.softmax(
                        torch.from_numpy(logits_np).float(), dim=-1
                    ).numpy()
                    preds_np = probs_np.argmax(axis=1)
                    task_metrics["accuracy"] = float(_acc(labels_np, preds_np))
                    task_metrics["f1_macro"] = float(
                        _f1(labels_np, preds_np, average="macro", zero_division=0)
                    )
                    task_metrics["f1_weighted"] = float(
                        _f1(labels_np, preds_np, average="weighted", zero_division=0)
                    )
                    try:
                        if probs_np.shape[1] == 2:
                            task_metrics["auroc"] = float(_auc(labels_np, probs_np[:, 1]))
                        else:
                            task_metrics["auroc"] = float(
                                _auc(labels_np, probs_np, multi_class="ovr")
                            )
                    except ValueError:
                        task_metrics["auroc"] = float("nan")

                else:  # multilabel
                    probs_np = torch.sigmoid(torch.from_numpy(logits_np).float()).numpy()
                    preds_np = (probs_np >= 0.5).astype(_np.int64)
                    # element-wise accuracy for multilabel setting
                    task_metrics["accuracy"] = float((preds_np == labels_np).mean())
                    task_metrics["f1_macro"] = float(
                        _f1(labels_np, preds_np, average="macro", zero_division=0)
                    )
                    task_metrics["f1_weighted"] = float(
                        _f1(labels_np, preds_np, average="weighted", zero_division=0)
                    )
                    try:
                        task_metrics["auroc"] = float(
                            _auc(labels_np, probs_np, average="macro")
                        )
                    except ValueError:
                        task_metrics["auroc"] = float("nan")

                self.last_val_metrics[task_name] = task_metrics
                metric_parts.append(
                    f"{task_name}_acc={task_metrics['accuracy']:.4f} "
                    f"{task_name}_f1m={task_metrics['f1_macro']:.4f} "
                    f"{task_name}_f1w={task_metrics['f1_weighted']:.4f} "
                    f"{task_name}_auc={task_metrics['auroc']:.4f}"
                )

            if metric_parts:
                logger.info("VAL | loss=%.6f | %s", mean_loss, " ".join(metric_parts))
            else:
                logger.info("VAL | loss=%.6f", mean_loss)
        except Exception as exc:
            logger.warning("Per-task validation metrics skipped: %s", exc)

        return float(mean_loss)


    def _extract_loss(self, outputs):

        if isinstance(outputs, dict):
            if "loss" not in outputs:
                raise RuntimeError("Model output must contain 'loss'")
            return outputs["loss"]

        if hasattr(outputs, "loss"):
            return outputs.loss

        raise RuntimeError("Unable to extract loss")


    def _move_batch_to_device(self, batch):

        if isinstance(batch, dict):
            return move_to_device(batch, self.device)

        if isinstance(batch, (list, tuple)):
            return type(batch)(
                move_to_device(x, self.device) for x in batch
            )

        raise TypeError("Unsupported batch format")


    def _prepare_model_inputs(self, batch):

        if not isinstance(batch, dict):
            return batch

        if self._forward_accepts_kwargs:
            return batch

        forward_kwargs = {}

        for key, value in batch.items():

            if key in self._forward_params:
                forward_kwargs[key] = value

        return forward_kwargs