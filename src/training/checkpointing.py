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


def _atomic_save(data: dict, path: Path) -> None:
    tmp_path = path.with_suffix(".tmp")
    cpu_data = _move_to_cpu(data)
    torch.save(cpu_data, tmp_path)
    os.replace(tmp_path, path)


def _atomic_save_compressed(data: dict, path: Path) -> None:
    cpu_data = _move_to_cpu(data)
    final_path = Path(str(path) + ".gz")
    tmp_path = Path(str(path) + ".tmp.gz")

    try:
        with gzip.open(tmp_path, "wb") as f:
            torch.save(cpu_data, f)
        os.replace(tmp_path, final_path)
    except Exception as exc:
        logger.exception("Compressed save failed")
        raise


def _resolve_checkpoint_path(checkpoint_dir: Path) -> Path:
    pt = checkpoint_dir / CHECKPOINT_FILE
    gz = checkpoint_dir / f"{CHECKPOINT_FILE}.gz"

    if pt.exists():
        return pt
    if gz.exists():
        return gz

    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}. "
        f"Expected {CHECKPOINT_FILE} or compressed variant."
    )


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _copy_to_drive(local_dir: Path, drive_dir: Optional[str | Path]) -> None:
    if drive_dir is None:
        return

    try:
        drive_dir = Path(drive_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)

        dest = drive_dir / local_dir.name

        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(local_dir, dest)
        logger.info("Checkpoint copied to Drive: %s", dest)

    except Exception as exc:
        logger.warning("Drive sync failed: %s", exc)


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
        raise RuntimeError("Checkpoint missing 'model_state_dict'")

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

    return {
        "epoch": checkpoint.get("epoch"),
        "step": checkpoint.get("step"),
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