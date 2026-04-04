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


logger = logging.getLogger(__name__)


CHECKPOINT_FILE = "checkpoint.pt"
METADATA_FILE = "metadata.json"


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
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

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

    return sorted(checkpoints)