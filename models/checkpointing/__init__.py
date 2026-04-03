"""Compatibility package for legacy `models.checkpointing.*` imports."""

from .checkpoint_manager import CheckpointManager, get_last_checkpoint

__all__ = ["CheckpointManager", "get_last_checkpoint"]
