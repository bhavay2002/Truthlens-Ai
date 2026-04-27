from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class TrainingSetupConfig:
    """
    Controls runtime behavior, precision, and safety checks.
    """

    # Precision
    use_amp: bool = True
    amp_dtype: str = "bf16"   # "bf16" | "fp16"
    allow_tf32: bool = True

    # Performance
    cudnn_benchmark: bool = True

    # Safety
    run_sanity_check: bool = True
    detect_anomaly: bool = False

    # Debug
    log_device_info: bool = True


# =========================================================
# DEVICE / RUNTIME SETUP
# =========================================================

def setup_runtime(config: TrainingSetupConfig) -> torch.device:
    """
    Configure global torch runtime.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # TF32 (Ampere+)
    # -------------------------
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32

    # -------------------------
    # cuDNN tuning
    # -------------------------
    torch.backends.cudnn.benchmark = config.cudnn_benchmark

    # -------------------------
    # Debug anomaly detection
    # -------------------------
    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # -------------------------
    # Logging
    # -------------------------
    if config.log_device_info:
        _log_device(device)

    return device


def _log_device(device: torch.device):
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("Using GPU: %s (%.2f GB)", name, total_mem)
    else:
        logger.info("Using CPU")


# =========================================================
# MIXED PRECISION
# =========================================================

def get_autocast(config: TrainingSetupConfig):
    """
    Return autocast context manager.
    """

    if not config.use_amp or not torch.cuda.is_available():
        return torch.cpu.amp.autocast(enabled=False)

    dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16

    return torch.cuda.amp.autocast(dtype=dtype)


def create_grad_scaler(config: TrainingSetupConfig) -> torch.cuda.amp.GradScaler:
    """
    Create AMP scaler.
    """
    return torch.cuda.amp.GradScaler(enabled=config.use_amp)


# =========================================================
# SANITY CHECK (CRITICAL)
# =========================================================

def run_sanity_check(
    *,
    model: torch.nn.Module,
    batch: Dict[str, Any],
    training_step,
    device: torch.device,
    max_batches: int = 1,
) -> None:
    """
    Validate full training pipeline before training starts.

    Checks:
    - forward pass
    - loss validity
    - backward pass
    - optimizer step
    """

    logger.info("Running training sanity check...")

    model.train()

    batch = move_to_device(batch, device)

    try:
        # MT-3: ``dry_run=True`` runs the full forward + loss + backward
        # pipeline so any wiring bug surfaces here, but does NOT mutate
        # any persistent training state — task scheduler index, optimizer
        # parameters, AMP loss scale, LR scheduler tick, monitor EMAs,
        # tracker step counter and balancer counters all stay frozen.
        # Without this flag the sanity check would silently desync every
        # one of these pieces of state by exactly one step versus a
        # reproducibility seed, and the round-robin task scheduler would
        # always start training at task index 1 instead of 0.
        outputs = training_step.run(batch, step=0, dry_run=True)

        # -------------------------
        # LOSS CHECK
        # -------------------------
        loss = outputs.get("raw_loss") or outputs.get("loss")

        if loss is None:
            raise RuntimeError("Sanity check: missing loss")

        if isinstance(loss, torch.Tensor):
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {loss.item()}")

        # -------------------------
        # OPTIONAL GRAD CHECK
        # -------------------------
        if hasattr(model, "parameters"):
            grad_norm = _compute_grad_norm(model)
            logger.info("Sanity grad_norm=%.4f", grad_norm)

    except Exception as e:
        logger.exception("Sanity check failed")
        raise RuntimeError("Sanity check failed") from e

    logger.info("Sanity check passed")


# =========================================================
# UTILITIES
# =========================================================

def move_to_device(batch: Any, device: torch.device) -> Any:
    """
    Recursively move batch to device.
    """

    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}

    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(x, device) for x in batch)

    if hasattr(batch, "to"):
        return batch.to(device)

    return batch


def _compute_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


# =========================================================
# MODEL OPTIMIZATION (OPTIONAL)
# =========================================================

def optimize_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply optional performance optimizations.
    """

    # PERF-3: ``torch.compile`` adds Dynamo tracing overhead with no payoff
    # on CPU (and on the Replit dev container in particular it slows training
    # by ~1.2-2×). The original ``except Exception`` also silently swallowed
    # legitimate compile errors, hiding regressions. Gate on CUDA and surface
    # failures at WARNING level so problems are visible.
    if torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile (mode=reduce-overhead)")
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)
    else:
        logger.info("Skipping torch.compile (CPU device)")

    # gradient checkpointing
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    except Exception:
        logger.debug("Gradient checkpointing skipped")

    return model