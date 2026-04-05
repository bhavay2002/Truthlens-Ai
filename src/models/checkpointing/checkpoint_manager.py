"""
File Name: checkpoint_manager.py
Module: models.checkpointing
Description:
    Provides checkpoint management utilities for the TruthLens AI training
    system. This module is responsible for saving model checkpoints, detecting
    the latest checkpoint, listing existing checkpoints, and cleaning up old
    checkpoints to control disk usage.

    The implementation follows production ML system practices used in large
    training pipelines where checkpoints are versioned by training step and
    stored in structured directories.

Dependencies:
    logging
    pathlib
    shutil
    typing
    torch
Inputs:
    checkpoint directory paths and model state dictionaries
Outputs:
    Saved checkpoint files and checkpoint metadata
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Utility class for managing training checkpoints.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------

    @staticmethod
    def _checkpoint_step(path: Path) -> Optional[int]:
        """
        Extract step number from checkpoint directory name.
        """

        name = path.name

        if not name.startswith("checkpoint-"):
            return None

        suffix = name.split("-", 1)[-1]

        if not suffix.isdigit():
            return None

        return int(suffix)

    # -------------------------------------------------
    # Save Checkpoint
    # -------------------------------------------------

    def save_checkpoint(
        self,
        step: int,
        model_state_dict: Dict[str, Any],
        optimizer_state_dict: Optional[Dict[str, Any]] = None,
        scheduler_state_dict: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model checkpoint.
        """

        if step < 0:
            raise ValueError("step must be non-negative")

        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_path / "checkpoint.pt"

        checkpoint_data = {
            "step": step,
            "model_state_dict": model_state_dict,
        }

        if optimizer_state_dict is not None:
            checkpoint_data["optimizer_state_dict"] = optimizer_state_dict

        if scheduler_state_dict is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler_state_dict

        if metadata is not None:
            checkpoint_data["metadata"] = metadata

        try:

            torch.save(checkpoint_data, checkpoint_file)

            logger.info("Checkpoint saved: %s", checkpoint_file)

        except Exception as exc:

            logger.exception("Failed to save checkpoint")

            raise RuntimeError("Checkpoint saving failed") from exc

        return checkpoint_path

    # -------------------------------------------------
    # Find Latest Checkpoint
    # -------------------------------------------------

    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Return the most recent checkpoint directory.
        """

        try:

            checkpoints = self.list_checkpoints()

            if not checkpoints:

                logger.info("No checkpoints found")

                return None

            latest = checkpoints[-1]

            logger.info("Latest checkpoint detected: %s", latest)

            return latest

        except Exception as exc:

            logger.exception("Failed to detect latest checkpoint")

            raise RuntimeError("Checkpoint detection failed") from exc

    # -------------------------------------------------
    # List Checkpoints
    # -------------------------------------------------

    def list_checkpoints(self) -> List[Path]:
        """
        Return all checkpoint directories sorted by step.
        """

        checkpoint_pairs: List[tuple[int, Path]] = []

        for checkpoint in self.checkpoint_dir.glob("checkpoint-*"):

            step = self._checkpoint_step(checkpoint)

            if step is not None:
                checkpoint_pairs.append((step, checkpoint))

        checkpoint_pairs.sort(key=lambda item: item[0])

        checkpoints = [checkpoint for _, checkpoint in checkpoint_pairs]

        return checkpoints

    # -------------------------------------------------
    # Cleanup Old Checkpoints
    # -------------------------------------------------

    def cleanup_old_checkpoints(self, max_checkpoints: int = 3) -> None:
        """
        Remove old checkpoints beyond max_checkpoints limit.
        """

        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be >= 1")

        try:

            checkpoints = self.list_checkpoints()

            if len(checkpoints) <= max_checkpoints:
                return

            to_delete = checkpoints[:-max_checkpoints]

            for checkpoint in to_delete:

                logger.info("Removing old checkpoint: %s", checkpoint)

                shutil.rmtree(checkpoint, ignore_errors=False)

        except Exception as exc:

            logger.exception("Checkpoint cleanup failed")

            raise RuntimeError("Checkpoint cleanup failed") from exc

    # -------------------------------------------------
    # Load Checkpoint
    # -------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        """
        Load checkpoint file.
        """

        checkpoint_path = Path(checkpoint_path)

        checkpoint_file = checkpoint_path / "checkpoint.pt"

        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        try:

            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

            logger.info("Checkpoint loaded: %s", checkpoint_file)

            return checkpoint

        except Exception as exc:

            logger.exception("Failed to load checkpoint")

            raise RuntimeError("Checkpoint loading failed") from exc


# ---------------------------------------------------------
# Convenience Helper
# ---------------------------------------------------------


def get_last_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """
    Retrieve latest checkpoint path.
    """

    manager = CheckpointManager(checkpoint_dir)

    return manager.get_latest_checkpoint()
