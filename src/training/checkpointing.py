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


def _quant_backend() -> str:
    supported = list(torch.backends.quantized.supported_engines)
    if "fbgemm" in supported:
        return "fbgemm"
    if "qnnpack" in supported:
        return "qnnpack"
    return supported[0] if supported else "qnnpack"


def _ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


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
) -> Path:
    """
    Save model checkpoint.
    """

    checkpoint_dir = Path(checkpoint_dir)
    _ensure_dir(checkpoint_dir)

    checkpoint_path = checkpoint_dir / CHECKPOINT_FILE

    checkpoint_data: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
    }

    if optimizer is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        try:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        except AttributeError:
            logger.warning("Scheduler does not support state_dict().")

    torch.save(checkpoint_data, checkpoint_path)

    logger.info("Checkpoint saved: %s", checkpoint_path)

    if metadata is not None:
        metadata_path = checkpoint_dir / METADATA_FILE
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info("Checkpoint metadata saved: %s", metadata_path)

    if export_formats:
        target_model = export_model if export_model is not None else model
        _export_artifacts(
            model=target_model,
            checkpoint_dir=checkpoint_dir,
            export_formats=export_formats,
            example_input=export_example_input,
        )

    return checkpoint_path


def load_checkpoint(
    model: torch.nn.Module,
    *,
    checkpoint_dir: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device | None = None,
) -> Dict[str, Any]:
    """
    Load checkpoint and restore model state.
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / CHECKPOINT_FILE

    if not checkpoint_path.exists():
        candidate_dirs = list_checkpoints(checkpoint_dir)
        if candidate_dirs:
            checkpoint_path = candidate_dirs[-1] / CHECKPOINT_FILE

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing required key 'model_state_dict': {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            logger.warning("Failed to restore scheduler state.")

    logger.info("Checkpoint loaded: %s", checkpoint_path)

    return {
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
    }


def resume_training(
    model: torch.nn.Module,
    *,
    checkpoint_dir: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device | None = None,
) -> Dict[str, Any]:
    """
    Resume training from checkpoint.
    """

    state = load_checkpoint(
        model,
        checkpoint_dir=checkpoint_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        map_location=map_location,
    )

    epoch = state.get("epoch") or 0
    step = state.get("step") or 0

    logger.info(
        "Resuming training from epoch=%s step=%s",
        epoch,
        step,
    )

    return {
        "start_epoch": epoch,
        "start_step": step,
    }


def list_checkpoints(checkpoint_root: str | Path) -> list[Path]:
    """
    List available checkpoint directories.
    """

    checkpoint_root = Path(checkpoint_root)

    if not checkpoint_root.exists():
        return []

    checkpoints = [
        p for p in checkpoint_root.iterdir()
        if p.is_dir() and (p / CHECKPOINT_FILE).exists()
    ]

    def _sort_key(path: Path) -> tuple[int, str]:
        # Prefer numeric ordering for HuggingFace-style checkpoint-<step> dirs.
        if path.name.startswith("checkpoint-"):
            suffix = path.name.split("-", 1)[-1]
            if suffix.isdigit():
                return (int(suffix), path.name)
        return (10**18, path.name)

    return sorted(checkpoints, key=_sort_key)


def _export_artifacts(
    *,
    model: torch.nn.Module,
    checkpoint_dir: Path,
    export_formats: list[str],
    example_input: Optional[torch.Tensor],
) -> None:
    export_dir = checkpoint_dir / "exports"
    _ensure_dir(export_dir)

    requested = {fmt.strip().lower() for fmt in export_formats}

    if "torchscript" in requested:
        if example_input is None:
            logger.warning("Skipping TorchScript export: example_input not provided.")
        else:
            exporter = TorchScriptExporter(
                TorchScriptExportConfig(device="cpu", verify_export=False)
            )
            try:
                exporter.export(
                    model=model,
                    example_input=example_input.detach().cpu(),
                    output_path=export_dir / "model.ts.pt",
                )
                logger.info("TorchScript export saved at %s", export_dir / "model.ts.pt")
            except Exception as exc:  # noqa: BLE001
                logger.warning("TorchScript export failed: %s", exc)

    if "onnx" in requested:
        if example_input is None:
            logger.warning("Skipping ONNX export: example_input not provided.")
        else:
            exporter = ONNXExporter(
                ONNXExportConfig(device="cpu", verify_export=False)
            )
            try:
                exporter.export(
                    model=model,
                    example_input=example_input.detach().cpu(),
                    output_path=export_dir / "model.onnx",
                )
                logger.info("ONNX export saved at %s", export_dir / "model.onnx")
            except Exception as exc:  # noqa: BLE001
                logger.warning("ONNX export failed: %s", exc)

    if "quantized" in requested:
        engine = QuantizationEngine(
            QuantizationConfig(method="dynamic", device="cpu", backend=_quant_backend())
        )
        try:
            quantized_model = engine.apply(model)
            torch.save(
                quantized_model.state_dict(),
                export_dir / "model.quantized.pt",
            )
            logger.info("Quantized export saved at %s", export_dir / "model.quantized.pt")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Quantized export failed: %s", exc)
