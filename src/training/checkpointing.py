from __future__ import annotations

import json
import logging
import shutil
import gzip
import os
import copy
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


def list_checkpoints(checkpoint_root: Path | str) -> list[Path]:
    """Return a sorted list of checkpoint directories under ``checkpoint_root``.

    A checkpoint directory is one that contains either ``checkpoint.pt`` or
    ``checkpoint.pt.gz``. Missing or empty roots yield an empty list.
    """
    root = Path(checkpoint_root)
    if not root.exists() or not root.is_dir():
        return []

    def _step_key(p: Path) -> tuple[int, int, str]:
        # Sort HF-style ``checkpoint-<N>`` numerically (so checkpoint-100
        # comes after checkpoint-2, not before). Non-numeric directories
        # (e.g. "best/") sort first by a fallback tier so they never get
        # picked as "the latest".
        name = p.name
        if name.startswith("checkpoint-"):
            tail = name[len("checkpoint-"):]
            if tail.isdigit():
                return (1, int(tail), name)
        return (0, 0, name)

    candidates = [
        entry for entry in root.iterdir()
        if entry.is_dir()
        and ((entry / CHECKPOINT_FILE).exists()
             or (entry / f"{CHECKPOINT_FILE}.gz").exists())
    ]
    return sorted(candidates, key=_step_key)


# =========================================================
#  UTILITIES
# =========================================================

def configure_training_precision():
    """Call ONLY in training entrypoint."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _move_to_cpu(obj):
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    return obj


def _fsync_dir(d: Path) -> None:
    """fsync the parent directory so the rename is durable (M8)."""
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        # Some filesystems / platforms don't support directory fsync.
        pass


def _validate_finite(state: dict) -> None:
    """Refuse to serialize NaN/Inf weights (C6)."""
    for k, v in state.items():
        if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
            raise RuntimeError(f"Refusing to save: non-finite values in '{k}'")
        if isinstance(v, dict):
            _validate_finite(v)


def _atomic_save(data: dict, path: Path) -> None:
    tmp_path = path.with_suffix(".tmp")
    cpu_data = _move_to_cpu(data)
    if isinstance(cpu_data, dict) and "model_state_dict" in cpu_data:
        _validate_finite(cpu_data["model_state_dict"])
    torch.save(cpu_data, tmp_path)
    os.replace(tmp_path, path)
    _fsync_dir(path.parent)


def _atomic_save_compressed(data: dict, path: Path) -> None:
    cpu_data = _move_to_cpu(data)
    if isinstance(cpu_data, dict) and "model_state_dict" in cpu_data:
        _validate_finite(cpu_data["model_state_dict"])
    final_path = Path(str(path) + ".gz")
    tmp_path = Path(str(path) + ".tmp.gz")

    try:
        with gzip.open(tmp_path, "wb") as f:
            torch.save(cpu_data, f)
        os.replace(tmp_path, final_path)
        _fsync_dir(final_path.parent)
    except Exception:
        logger.exception("Compressed save failed")
        raise


def _resolve_checkpoint_path(checkpoint_dir: Path) -> Path:
    pt = checkpoint_dir / CHECKPOINT_FILE
    gz = checkpoint_dir / f"{CHECKPOINT_FILE}.gz"

    if pt.exists():
        return pt
    if gz.exists():
        return gz

    # Fallback: caller passed a checkpoint *root* (containing
    # ``checkpoint-<step>/`` subdirs) instead of an individual checkpoint
    # directory. Walk one level down and pick the latest valid one.
    nested = list_checkpoints(checkpoint_dir)
    if nested:
        latest = nested[-1]
        if (latest / CHECKPOINT_FILE).exists():
            return latest / CHECKPOINT_FILE
        if (latest / f"{CHECKPOINT_FILE}.gz").exists():
            return latest / f"{CHECKPOINT_FILE}.gz"

    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}. "
        f"Expected {CHECKPOINT_FILE} or compressed variant."
    )


def _safe_torch_load(path, map_location):
    # PyTorch 2.6+ defaults weights_only=True and refuses non-tensor metadata
    # (e.g. pytorch_version, config dicts). Our checkpoints intentionally carry
    # such fields, so fall back to a full load when the safe load rejects them.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def _copy_to_drive(
    local_dir: Path,
    drive_dir: Optional[str | Path],
    retries: int = 3,
) -> None:
    """Atomic, retried, size-validated drive sync (C7, m4).

    Strategy: copytree → temp dir → atomic rename. Failure raises so the
    trainer can react, instead of being silently downgraded to a warning.
    """
    if drive_dir is None:
        return

    drive_dir = Path(drive_dir)
    drive_dir.mkdir(parents=True, exist_ok=True)

    dest = drive_dir / local_dir.name
    staging = drive_dir / f".{local_dir.name}.tmp"
    backup = drive_dir / f".{local_dir.name}.bak"

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(local_dir, staging)

            # Validate file sizes match
            for src in local_dir.rglob("*"):
                if not src.is_file():
                    continue
                rel = src.relative_to(local_dir)
                tgt = staging / rel
                if not tgt.exists() or tgt.stat().st_size != src.stat().st_size:
                    raise IOError(f"Size mismatch / missing in copy: {rel}")

            # Atomic swap: existing → backup → staging → final
            if dest.exists():
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(dest, backup)
            os.replace(staging, dest)
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)

            logger.info("Checkpoint copied to Drive: %s", dest)
            return
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Drive sync attempt %d/%d failed for %s: %s",
                attempt, retries, dest, exc,
            )
            try:
                if staging.exists():
                    shutil.rmtree(staging)
            except Exception:
                pass

    raise RuntimeError(f"Drive sync failed for {dest}") from last_exc


# =========================================================
#  SAVE CHECKPOINT
# =========================================================

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
    export_example_input: Optional[torch.Tensor] = None,
    drive_checkpoint_dir: Optional[str | Path] = None,
    use_compression: bool = False,
) -> Path:

    checkpoint_dir = Path(checkpoint_dir)
    _ensure_dir(checkpoint_dir)

    checkpoint_path = checkpoint_dir / CHECKPOINT_FILE

    # C5: include audit-mandated keys (loss + config)
    _meta = metadata or {}
    checkpoint_data: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": _meta.get("val_loss") or _meta.get("train_loss") or _meta.get("loss"),
        "config": _meta.get("config"),
        "pytorch_version": torch.__version__,
    }

    if optimizer is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        try:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        except Exception as exc:
            logger.warning("Scheduler save failed: %s", exc)

    if use_compression:
        _atomic_save_compressed(checkpoint_data, checkpoint_path)
        saved_path = Path(str(checkpoint_path) + ".gz")
    else:
        _atomic_save(checkpoint_data, checkpoint_path)
        saved_path = checkpoint_path

    logger.info("Checkpoint saved: %s", saved_path)

    if metadata:
        with (checkpoint_dir / METADATA_FILE).open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    if export_formats:
        _export_artifacts(
            model=model,
            checkpoint_dir=checkpoint_dir,
            export_formats=export_formats,
            example_input=export_example_input,
        )

    _copy_to_drive(checkpoint_dir, drive_checkpoint_dir)

    return saved_path


# =========================================================
#  LOAD CHECKPOINT (STRICT + SAFE)
# =========================================================

def load_checkpoint(
    model: torch.nn.Module,
    *,
    checkpoint_dir: str | Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str | torch.device] = None,
    strict: bool = True,
    allow_missing: tuple[str, ...] = (),
) -> Dict[str, Any]:

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_dir)

    if str(checkpoint_path).endswith(".gz"):
        with gzip.open(checkpoint_path, "rb") as f:
            checkpoint = _safe_torch_load(f, map_location)
    else:
        checkpoint = _safe_torch_load(checkpoint_path, map_location)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Invalid checkpoint format")

    if "model_state_dict" not in checkpoint:
        # KeyError is the audit-mandated signal: callers (e.g. Trainer
        # ._attempt_resume) distinguish "schema drift" from generic load
        # failures by catching this specifically.
        raise KeyError("Checkpoint missing 'model_state_dict'")

    state_dict = checkpoint["model_state_dict"]

    load_result = model.load_state_dict(state_dict, strict=False)

    missing = [
        k for k in load_result.missing_keys
        if not any(k.startswith(p) for p in allow_missing)
    ]

    unexpected = load_result.unexpected_keys

    if strict and (missing or unexpected):
        raise RuntimeError(
            "Strict checkpoint load failed:\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    if missing:
        logger.warning("Missing keys (ignored): %s", missing)

    if unexpected:
        logger.warning("Unexpected keys (ignored): %s", unexpected)

    #  optimizer restore (device safe)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            device = next(model.parameters()).device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        except Exception as exc:
            if strict:
                raise RuntimeError(f"Optimizer restore failed: {exc}") from exc
            logger.warning("Optimizer restore failed: %s", exc)

    #  scheduler restore (strict-aware)
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Scheduler restore failed: {exc}") from exc
            logger.warning("Scheduler restore failed: %s", exc)

    logger.info("Checkpoint loaded: %s", checkpoint_path)

    # C4: surface full payload so callers (Trainer._attempt_resume) can
    # introspect scheduler/optimizer/loss/config without reaching back into
    # the raw checkpoint file.
    return {
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
        "loss": checkpoint.get("loss"),
        "config": checkpoint.get("config"),
        "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
    }


# =========================================================
#  RESUME
# =========================================================

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
        strict=True,
    )

    epoch = state.get("epoch") or 0
    step = state.get("step") or 0

    logger.info("Resuming training from epoch=%s step=%s", epoch, step)

    return {
        "start_epoch": epoch,
        "start_step": step,
    }


# =========================================================
#  EXPORT (SAFE COPY)
# =========================================================

def _export_artifacts(
    *,
    model: torch.nn.Module,
    checkpoint_dir: Path,
    export_formats: list[str],
    example_input: Optional[torch.Tensor],
) -> None:

    export_dir = checkpoint_dir / "exports"
    _ensure_dir(export_dir)

    model_copy = copy.deepcopy(model).cpu().eval()

    requested = {fmt.lower() for fmt in export_formats}

    if "torchscript" in requested and example_input is not None:
        try:
            exporter = TorchScriptExporter(
                TorchScriptExportConfig(device="cpu", verify_export=False)
            )
            exporter.export(
                model=model_copy,
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
                model=model_copy,
                example_input=example_input.detach().cpu(),
                output_path=export_dir / "model.onnx",
            )
        except Exception as exc:
            logger.warning("ONNX export failed: %s", exc)

    if "quantized" in requested:
        try:
            engine = QuantizationEngine(
                QuantizationConfig(method="dynamic", device="cpu", backend="fbgemm")
            )
            quant_model = engine.apply(copy.deepcopy(model).cpu())

            torch.save(
                quant_model.state_dict(),
                export_dir / "model.quantized.pt",
            )

        except Exception as exc:
            logger.warning("Quantized export failed: %s", exc)