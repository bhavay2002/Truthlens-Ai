"""
File Name: checkpointing.py
Module: TruthLens AI - Training Checkpoint Manager
Description:
    Central checkpoint management utilities for TruthLens AI training pipelines.
    Provides standardized interfaces for saving model checkpoints, loading
    checkpoints, resuming training, and maintaining versioned checkpoints.

Dependencies:
    logging
    pathlib
    typing
    json
    torch

Inputs:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer 
    scheduler: learning rate scheduler 
    epoch: current training epoch
    step: training step
    metadata: additional training metadata

Outputs:
    saved checkpoint files
    restored model/optimizer/scheduler states
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import gzip
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.models.export import (
    ONNXExportConfig,
    ONNXExporter,
    QuantizationConfig,
    QuantizationEngine,
    TorchScriptExportConfig,
    TorchScriptExporter,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.pt"
METADATA_FILE = "metadata.json"

# GPU performance boost (guarded; avoid unsafe import-time behavior on non-CUDA envs)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _quant_backend() -> str:
    supported = list(torch.backends.quantized.supported_engines)

    if "fbgemm" in supported:
        return "fbgemm"
    if "qnnpack" in supported:
        return "qnnpack"

    return supported[0] if supported else "qnnpack"


def _move_to_cpu(obj):
    """Recursively move tensors to CPU (prevents GPU memory spikes)."""
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    return obj


def _atomic_save(data: dict, path: Path) -> None:
    """
    Safe + memory-efficient save.
    """

    tmp_path = path.with_suffix(".tmp")

    cpu_data = _move_to_cpu(data)

    torch.save(cpu_data, tmp_path)

    tmp_path.replace(path)


def _atomic_save_compressed(data: dict, path: Path) -> None:
    """Optional compressed checkpoint."""
    cpu_data = _move_to_cpu(data)
    final_path = Path(str(path) + ".gz")
    tmp_path = Path(str(path) + ".tmp.gz")
    with gzip.open(tmp_path, "wb") as f:
        torch.save(cpu_data, f)
    tmp_path.replace(final_path)


def _copy_to_drive(local_dir: Path, drive_dir: Optional[str | Path]) -> None:

    if drive_dir is None:
        return

    drive_dir = Path(drive_dir)

    try:
        drive_dir.mkdir(parents=True, exist_ok=True)

        dest = drive_dir / local_dir.name

        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(local_dir, dest)

        logger.info("Checkpoint copied to Drive: %s", dest)

    except Exception as exc:
        logger.warning("Drive sync failed: %s", exc)


def _resolve_checkpoint_path(checkpoint_dir: Path) -> Path:
    """
    Resolve checkpoint path supporting both uncompressed and compressed formats.
    """
    pt = checkpoint_dir / CHECKPOINT_FILE
    gz = checkpoint_dir / f"{CHECKPOINT_FILE}.gz"

    if pt.exists():
        return pt
    if gz.exists():
        return gz

    candidates = list_checkpoints(checkpoint_dir)
    if candidates:
        last = candidates[-1]
        pt2 = last / CHECKPOINT_FILE
        gz2 = last / f"{CHECKPOINT_FILE}.gz"
        if pt2.exists():
            return pt2
        if gz2.exists():
            return gz2
    raise FileNotFoundError(f"Checkpoint not found in {checkpoint_dir}")


# ---------------------------------------------------------
# Save Checkpoint
# ---------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    *,
    checkpoint_dir: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    export_formats: Optional[list[str]] = None,
    export_model: Optional[torch.nn.Module] = None,
    export_example_input: Optional[torch.Tensor] = None,
    drive_checkpoint_dir: Optional[str | Path] = None,
    use_compression: bool = False,  # 🔥 NEW
) -> Path:

    checkpoint_dir = Path(checkpoint_dir)
    _ensure_dir(checkpoint_dir)

    checkpoint_path = checkpoint_dir / CHECKPOINT_FILE

    checkpoint_data: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "pytorch_version": torch.__version__,
    }

    if optimizer is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        try:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        except AttributeError:
            logger.warning("Scheduler has no state_dict()")

    saved_path = checkpoint_path
    if use_compression:
        _atomic_save_compressed(checkpoint_data, checkpoint_path)
        saved_path = Path(str(checkpoint_path) + ".gz")
    else:
        _atomic_save(checkpoint_data, checkpoint_path)
    logger.info("Checkpoint saved: %s", saved_path)

    # metadata
    if metadata is not None:
        metadata_path = checkpoint_dir / METADATA_FILE
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    if export_formats:
        target_model = export_model if export_model else model
        _export_artifacts(
            model=target_model,
            checkpoint_dir=checkpoint_dir,
            export_formats=export_formats,
            example_input=export_example_input,
        )

    _copy_to_drive(checkpoint_dir, drive_checkpoint_dir)

    return saved_path


# ---------------------------------------------------------
# Load Checkpoint
# ---------------------------------------------------------

def load_checkpoint(
    model: torch.nn.Module,
    *,
    checkpoint_dir: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str | torch.device] = None,
) -> Dict[str, Any]:

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_dir)

    if str(checkpoint_path).endswith(".gz"):
        with gzip.open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f, map_location=map_location, weights_only=True)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_result.missing_keys:
        logger.warning(
            "load_checkpoint: missing keys in checkpoint (will be randomly initialised): %s",
            load_result.missing_keys,
        )
    if load_result.unexpected_keys:
        logger.warning(
            "load_checkpoint: unexpected keys in checkpoint (will be ignored): %s",
            load_result.unexpected_keys,
        )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            logger.warning("Scheduler restore failed")

    logger.info("Checkpoint loaded: %s", checkpoint_path)

    return {
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
    }


# ---------------------------------------------------------
# Resume Training
# ---------------------------------------------------------

def resume_training(
    model: torch.nn.Module,
    *,
    checkpoint_dir: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str | torch.device] = None,
) -> Dict[str, Any]:

    state = load_checkpoint(
        model,
        checkpoint_dir=checkpoint_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=map_location,
    )

    epoch = state.get("epoch") or 0
    step = state.get("step") or 0

    logger.info("Resuming training from epoch=%s step=%s", epoch, step)

    return {
        "start_epoch": epoch,
        "start_step": step,
    }


# ---------------------------------------------------------
# List Checkpoints
# ---------------------------------------------------------

def list_checkpoints(checkpoint_root: str | Path) -> list[Path]:

    checkpoint_root = Path(checkpoint_root)

    if not checkpoint_root.exists():
        return []

    checkpoints = [
        p for p in checkpoint_root.iterdir()
        if p.is_dir() and (
            (p / CHECKPOINT_FILE).exists()
            or (p / f"{CHECKPOINT_FILE}.gz").exists()
        )
    ]

    def sort_key(p: Path):
        if p.name.startswith("checkpoint-"):
            suffix = p.name.split("-", 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return float("inf")

    return sorted(checkpoints, key=sort_key)


# ---------------------------------------------------------
# Export Artifacts
# ---------------------------------------------------------

def _export_artifacts(
    *,
    model: torch.nn.Module,
    checkpoint_dir: Path,
    export_formats: list[str],
    example_input: Optional[torch.Tensor],
) -> None:

    export_dir = checkpoint_dir / "exports"
    _ensure_dir(export_dir)

    requested = {fmt.lower() for fmt in export_formats}

    model.eval()

    if "torchscript" in requested and example_input is not None:
        try:
            exporter = TorchScriptExporter(
                TorchScriptExportConfig(device="cpu", verify_export=False)
            )
            exporter.export(
                model=model,
                example_input=example_input.detach().cpu(),
                output_path=export_dir / "model.ts.pt",
            )
        except Exception as exc:
            logger.warning("TorchScript export failed: %s", exc)

    if "onnx" in requested and example_input is not None:
        try:
            exporter = ONNXExporter(
                ONNXExportConfig(device="cpu", verify_export=False)
            )
            exporter.export(
                model=model,
                example_input=example_input.detach().cpu(),
                output_path=export_dir / "model.onnx",
            )
        except Exception as exc:
            logger.warning("ONNX export failed: %s", exc)

    if "quantized" in requested:
        try:
            engine = QuantizationEngine(
                QuantizationConfig(method="dynamic", device="cpu", backend=_quant_backend())
            )
            quant_model = engine.apply(model)
            torch.save(
                quant_model.state_dict(),
                export_dir / "model.quantized.pt",
            )
        except Exception as exc:
            logger.warning("Quantized export failed: %s", exc)