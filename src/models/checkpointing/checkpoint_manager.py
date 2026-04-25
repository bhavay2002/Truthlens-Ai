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

from .validator import validate_checkpoint

logger = logging.getLogger(__name__)


# =====================================================
# ASYNC WRITER
# =====================================================

class AsyncCheckpointWriter:

    def __init__(self, max_queue_size: int = 4) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._closed = False
        self.last_error: Optional[Exception] = None
        self._thread.start()

    def _worker(self):
        import os
        while True:
            item = self._queue.get()

            if item is None:
                self._queue.task_done()
                break

            path, obj = item
            tmp = path.with_suffix(".tmp")

            try:
                torch.save(obj, tmp)
                os.replace(tmp, path)
            except Exception as e:
                self.last_error = e
                logger.error("Checkpoint save failed", exc_info=True)
            finally:
                self._queue.task_done()

    def save(self, path: Path, obj: Any):

        if self._closed:
            raise RuntimeError("Writer closed")

        try:
            self._queue.put_nowait((path, obj))
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait((path, obj))
            except queue.Empty:
                pass

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
# MANAGER
# =====================================================

class CheckpointManager:

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._writer = AsyncCheckpointWriter()
        self._last_hash: Optional[str] = None

    # -------------------------------------------------
    # UTILS
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
            return model_or_state.state_dict()
        return model_or_state

    @staticmethod
    def _to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.detach().cpu() if torch.is_tensor(v) else v
            for k, v in state.items()
        }

    @staticmethod
    def _validate_finite(state: Dict[str, Any]):
        for k, v in state.items():
            if torch.is_tensor(v) and v.is_floating_point():
                if not torch.isfinite(v).all():
                    raise RuntimeError(f"Non-finite in {k}")

    def _hash(self, state: Dict[str, Any]) -> str:
        h = hashlib.md5()
        for k, v in state.items():
            h.update(k.encode())
            if torch.is_tensor(v):
                h.update(str(v.shape).encode())
                h.update(v.flatten()[:10].cpu().numpy().tobytes())
        return h.hexdigest()

    @staticmethod
    def _extract_step(path: Path) -> Optional[int]:
        try:
            return int(path.name.split("-")[-1])
        except Exception:
            return None

    # -------------------------------------------------
    # SAVE
    # -------------------------------------------------

    def save_checkpoint(
        self,
        step: int,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scaler: Optional[Any] = None,
        save_every: int = 1000,
    ) -> Optional[Path]:

        if not self.should_save(step, save_every):
            return None

        if not self._is_primary():
            return None

        path = self.checkpoint_dir / f"checkpoint-{step}"
        path.mkdir(parents=True, exist_ok=True)

        file = path / "checkpoint.pt"

        state = self._to_cpu(self._extract_state(model))

        self._validate_finite(state)
        validate_checkpoint(state)

        h = self._hash(state)
        if h == self._last_hash:
            return None
        self._last_hash = h

        payload: Dict[str, Any] = {
            "step": step,
            "model_state_dict": state,
            "metadata": metadata,
            "pytorch_version": torch.__version__,
        }

        if optimizer:
            payload["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler:
            try:
                payload["scheduler_state_dict"] = scheduler.state_dict()
            except Exception:
                pass

        if scaler:
            try:
                payload["scaler_state_dict"] = scaler.state_dict()
            except Exception:
                pass

        if self._writer.last_error:
            err = self._writer.last_error
            self._writer.last_error = None
            raise RuntimeError(f"Async failure: {err}")

        self._writer.save(file, payload)

        return path

    # -------------------------------------------------
    # LOAD
    # -------------------------------------------------

    def load_checkpoint(self, path: str | Path) -> Dict[str, Any]:

        path = Path(path)

        if path.is_dir():
            path = path / "checkpoint.pt"

        if not path.exists():
            raise FileNotFoundError(path)

        return torch.load(path, map_location="cpu")

    # -------------------------------------------------
    # LIST
    # -------------------------------------------------

    def list_checkpoints(self) -> List[Path]:

        checkpoints = []

        for p in self.checkpoint_dir.iterdir():
            if p.is_dir() and p.name.startswith("checkpoint-"):
                step = self._extract_step(p)
                if step is not None:
                    checkpoints.append((p, step))

        return [p for p, _ in sorted(checkpoints, key=lambda x: x[1])]

    def latest(self) -> Optional[Path]:

        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None

    # -------------------------------------------------
    # CLEANUP
    # -------------------------------------------------

    def cleanup(self, keep: int = 3):

        self._writer.flush()

        ckpts = self.list_checkpoints()

        if len(ckpts) <= keep:
            return

        for p in ckpts[:-keep]:
            try:
                shutil.rmtree(p)
            except Exception:
                pass

    # -------------------------------------------------
    # CLOSE
    # -------------------------------------------------

    def close(self):
        self._writer.close()

    def __del__(self):
        try:
            self._writer.close()
        except Exception:
            pass


# =====================================================
# HELPER
# =====================================================

def get_last_checkpoint(checkpoint_dir: str | Path) -> Optional[Path]:

    manager = CheckpointManager(checkpoint_dir)
    return manager.latest()