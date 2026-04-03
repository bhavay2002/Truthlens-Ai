"""Compatibility wrapper for checkpoint manager APIs."""

from src.models.checkpointing.checkpoint_manager import (
    CheckpointManager,
    get_last_checkpoint,
)

__all__ = ["CheckpointManager", "get_last_checkpoint"]
