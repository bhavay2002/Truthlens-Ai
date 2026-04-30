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

    # COMPILE-OFF: ``torch.compile`` is fully disabled across the
    # codebase (see ``optimize_model`` below and the matching gates in
    # transformer_encoder, model_loader, feature_pipeline,
    # batch_feature_pipeline). The flags below are kept as inert
    # back-compat fields so callers that read them don't break — flipping
    # them back to True will NOT re-enable compilation; remove the
    # explicit no-op gates first if you want it back.
    use_compile: bool = False
    compile_mode: str = "default"   # inert; see COMPILE-OFF above

    # Gradient checkpointing
    use_gradient_checkpointing: bool = True

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

    N-LOW-1: ``torch.cuda.amp.autocast`` and ``torch.cpu.amp.autocast`` were
    deprecated in PyTorch 2.3 in favour of the device-type-aware
    ``torch.amp.autocast(device_type, ...)``. Switching ahead of removal in
    PyTorch 2.6+ avoids a ``DeprecationWarning`` per training step.
    """

    if not config.use_amp or not torch.cuda.is_available():
        return torch.amp.autocast("cpu", enabled=False)

    dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16

    return torch.amp.autocast("cuda", dtype=dtype)


def create_grad_scaler(config: TrainingSetupConfig) -> "torch.amp.GradScaler":
    """
    Create AMP scaler.

    N-LOW-1: ``torch.cuda.amp.GradScaler`` was deprecated in PyTorch 2.3 in
    favour of ``torch.amp.GradScaler("cuda", ...)``.
    """
    return torch.amp.GradScaler("cuda", enabled=config.use_amp)


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
    # N-LOW-8: route through ``training_utils.compute_grad_norm`` instead
    # of a near-identical local copy. Four call sites (training_setup,
    # monitor_engine, instrumentation.GradTracker, training_utils) all
    # implemented the same L2 reduction with subtle differences (.data
    # vs no .data, .item() per param vs sum then sqrt, etc.); consolidating
    # ensures one canonical implementation under test.
    from src.training.training_utils import compute_grad_norm
    return compute_grad_norm(model)


# =========================================================
# MODEL OPTIMIZATION (OPTIONAL)
# =========================================================

def optimize_model(
    model: torch.nn.Module,
    *,
    config: Optional["TrainingSetupConfig"] = None,
) -> torch.nn.Module:
    """
    Apply optional performance optimizations.

    N-LOW-3: ORDERING CONTRACT — ``optimize_model`` MUST be called BEFORE
    ``DistributedEngine.wrap_model``.  ``torch.compile`` walks the model's
    module tree and rewrites forward methods; running it on a DDP-wrapped
    model breaks the rewrite (DDP's ``forward`` interposition prevents
    Dynamo from seeing the underlying module) and produces no speedup.
    The Trainer enforces this order: ``optimize_model`` runs in
    ``__init__`` before ``self.distributed.wrap_model(self.model)`` is
    called. Callers using these helpers directly must do the same.

    Parameters
    ----------
    model:
        The model to optimize.
    config:
        Optional ``TrainingSetupConfig``. When provided, ``use_compile``,
        ``compile_mode``, and ``use_gradient_checkpointing`` are read from
        it. Falls back to safe defaults when ``None``.
    """

    use_gc = config.use_gradient_checkpointing if config is not None else True

    # COMPILE-OFF: model compilation is intentionally disabled. It added
    # instability across the supported environments (Python / CUDA /
    # kernel combinations on Replit and on user machines), surfaced as
    # spurious "Gradient overflow detected" warnings under bf16 AMP, and
    # was not necessary for current training stability. The previous
    # call site has been removed; the ``use_compile`` / ``compile_mode``
    # config fields are kept as inert back-compat plumbing (see
    # ``TrainingSetupConfig`` above).
    logger.info(
        "torch.compile is disabled project-wide (COMPILE-OFF); "
        "running model in eager mode."
    )

    # gradient checkpointing
    if use_gc:
        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        except Exception:
            logger.debug("Gradient checkpointing skipped")
    else:
        logger.info("Gradient checkpointing disabled via config")

    return model