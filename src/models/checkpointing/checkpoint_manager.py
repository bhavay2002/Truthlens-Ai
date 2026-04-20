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

import hashlib
import logging
import queue
import shutil
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


# =====================================================
# Async Checkpoint Writer (Non-blocking)
# =====================================================

class AsyncCheckpointWriter:
    def __init__(self, max_queue_size: int = 4) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._closed = False
        self._thread.start()

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            path, obj = item
            tmp_path = path.with_suffix(path.suffix + ".tmp")

            try:
                torch.save(obj, tmp_path, _use_new_zipfile_serialization=True)
                tmp_path.replace(path)
            except Exception as e:
                logger.error("Checkpoint save failed: %s", e)
            finally:
                self._queue.task_done()

    def save(self, path: Path, obj: Any):
        if self._closed:
            raise RuntimeError("Cannot save: AsyncCheckpointWriter is closed")
        try:
            self._queue.put_nowait((path, obj))
        except queue.Full:
            logger.warning("Checkpoint queue full, dropping oldest checkpoint")
            try:
                _ = self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait((path, obj))
            except queue.Empty:
                logger.error(
                    "Checkpoint save dropped due to race condition on full queue: %s",
                    path,
                )

    def flush(self):
        self._queue.join()

    def close(self):
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(None)
        self._thread.join()


# =====================================================
# Checkpoint Manager
# =====================================================

class CheckpointManager:

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._writer = AsyncCheckpointWriter()
        self._last_hash: Optional[str] = None

    # -------------------------------------------------
    # Utils
    # -------------------------------------------------

    @staticmethod
    def should_save(step: int, save_every: int) -> bool:
        return save_every > 0 and step % save_every == 0

    @staticmethod
    def _is_primary() -> bool:
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    @staticmethod
    def _extract_state(model_or_state):
        if isinstance(model_or_state, torch.nn.Module):
            model = model_or_state
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod
            return model.state_dict()
        return model_or_state

    @staticmethod
    def _to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
        new_state: Dict[str, Any] = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                t = v.detach().to("cpu", non_blocking=True)
                new_state[k] = t.pin_memory()
            else:
                new_state[k] = v
        return new_state

    @staticmethod
    def _hash_state(state: Dict[str, Any]) -> str:
        h = hashlib.md5()
        for k, v in state.items():
            h.update(k.encode())
            if torch.is_tensor(v):
                sample = v.detach().cpu().flatten()[:10].contiguous()
                h.update(sample.numpy().tobytes())
        return h.hexdigest()

    @staticmethod
    def _extract_step(path: Path) -> Optional[int]:
        parts = path.name.split("-")
        if len(parts) < 2:
            return None
        try:
            return int(parts[-1])
        except ValueError:
            return None

    # -------------------------------------------------
    # Save Checkpoint
    # -------------------------------------------------

    def save_checkpoint(
        self,
        step: int,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        save_optimizer: bool = False,
        save_every: int = 1000,
        deduplicate: bool = True,
    ) -> Optional[Path]:

        if not self.should_save(step, save_every):
            return None

        if not self._is_primary():
            return None

        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_path / "checkpoint.pt"

        # Extract state
        state_dict = self._extract_state(model)
        state_dict = self._to_cpu(state_dict)

        # Deduplication
        if deduplicate:
            h = self._hash_state(state_dict)
            if h == self._last_hash:
                logger.info("Skipping duplicate checkpoint")
                return None
            self._last_hash = h

        payload: Dict[str, Any] = {
            "step": step,
            "model": state_dict,
        }

        if save_optimizer and optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()

        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()

        if metadata:
            payload["metadata"] = metadata

        # Async save
        self._writer.save(checkpoint_file, payload)

        logger.info("Checkpoint queued: %s", checkpoint_file)
        return checkpoint_path

    # -------------------------------------------------
    # Sharded Save (for large models)
    # -------------------------------------------------

    def save_sharded(self, step: int, model, shards: int = 4):

        if not self._is_primary():
            return []

        state = self._to_cpu(self._extract_state(model))
        items = list(state.items())

        shard_size = max(1, len(items) // shards)
        paths = []

        for i in range(shards):
            start = i * shard_size
            end = (i + 1) * shard_size if i < shards - 1 else len(items)
            shard = dict(items[start:end])
            path = self.checkpoint_dir / f"checkpoint-{step}-shard-{i}.pt"
            self._writer.save(path, shard)
            paths.append(path)

        return paths

    # -------------------------------------------------
    # Load
    # -------------------------------------------------

    def load_checkpoint(self, path: str | Path) -> Dict[str, Any]:
        path = Path(path)

        if path.is_dir():
            path = path / "checkpoint.pt"

        if not path.exists():
            raise FileNotFoundError(path)

        return torch.load(path, map_location="cpu", weights_only=True)

    # -------------------------------------------------
    # List + Latest
    # -------------------------------------------------

    def list_checkpoints(self) -> List[Path]:
        if not self.checkpoint_dir.exists():
            return []

        checkpoints: list[tuple[Path, int]] = []

        for p in self.checkpoint_dir.iterdir():
            if not (p.is_dir() and p.name.startswith("checkpoint-")):
                continue

            step = self._extract_step(p)
            if step is None:
                continue

            checkpoints.append((p, step))

        return [p for p, _ in sorted(checkpoints, key=lambda x: x[1])]

    def get_latest_checkpoint(self) -> Optional[Path]:
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------

    def cleanup_old_checkpoints(self, max_checkpoints: int = 3):
        if max_checkpoints <= 0:
            raise ValueError(
                f"max_checkpoints must be a positive integer, got {max_checkpoints}"
            )

        self._writer.close()

        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= max_checkpoints:
            return

        for p in checkpoints[:-max_checkpoints]:
            try:
                shutil.rmtree(p)
                logger.info("Deleted checkpoint: %s", p)
            except Exception as e:
                logger.warning("Failed to delete checkpoint %s: %s", p, e)

    # -------------------------------------------------
    # Shutdown
    # -------------------------------------------------

    def close(self):
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._writer.close()

    def __del__(self):
        try:
            self._writer.close()
        except Exception:
            pass



# =====================================================
# Module-level convenience function
# =====================================================

def get_last_checkpoint(checkpoint_dir: str | Path) -> "Optional[Path]":
    """Return the path to the most recent checkpoint in *checkpoint_dir*.

    Returns ``None`` if the directory does not exist or contains no valid
    checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    try:
        checkpoints: list[tuple[Path, int]] = []
        for p in checkpoint_dir.iterdir():
            if not (p.is_dir() and p.name.startswith("checkpoint-")):
                continue
            parts = p.name.split("-")
            if len(parts) < 2:
                continue
            try:
                step = int(parts[-1])
            except ValueError:
                continue
            checkpoints.append((p, step))
        return max(checkpoints, key=lambda x: x[1])[0] if checkpoints else None
    except Exception as e:
        logger.warning("Failed to get last checkpoint: %s", e)
        return None
