"""
File: checkpoint_manager.py

Purpose
-------
Checkpoint management utilities for TruthLens AI.

This module provides functionality for saving, detecting,
and cleaning training checkpoints.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional, List

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------


class CheckpointManager:
    """
    Utility class for managing model checkpoints.
    """

    def __init__(self, checkpoint_dir: str | Path):

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Find Latest Checkpoint
    # -------------------------------------------------

    @staticmethod
    def _checkpoint_step(path: Path) -> int | None:
        name = path.name
        if not name.startswith("checkpoint-"):
            return None

        suffix = name.split("-", 1)[-1]
        if not suffix.isdigit():
            return None

        return int(suffix)

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

        except Exception:

            logger.exception("Failed to detect latest checkpoint")

            raise

    # -------------------------------------------------
    # List Checkpoints
    # -------------------------------------------------

    def list_checkpoints(self) -> List[Path]:
        """
        Return all checkpoint directories.
        """

        checkpoint_pairs = []

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

    def cleanup_old_checkpoints(self, max_checkpoints: int = 3):
        """
        Remove old checkpoints beyond max_checkpoints limit.
        """

        try:
            if max_checkpoints < 1:
                raise ValueError("max_checkpoints must be >= 1")

            checkpoints = self.list_checkpoints()

            if len(checkpoints) <= max_checkpoints:

                return

            to_delete = checkpoints[:-max_checkpoints]

            for checkpoint in to_delete:

                logger.info("Removing old checkpoint: %s", checkpoint)

                shutil.rmtree(checkpoint, ignore_errors=False)

        except Exception:

            logger.exception("Checkpoint cleanup failed")

            raise


# ---------------------------------------------------------
# Convenience Helper
# ---------------------------------------------------------


def get_last_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:
    """
    Helper function for retrieving latest checkpoint.
    """

    manager = CheckpointManager(checkpoint_dir)

    return manager.get_latest_checkpoint()
