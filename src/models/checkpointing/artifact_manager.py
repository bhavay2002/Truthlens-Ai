"""
File Name: artifact_manager.py
Module: models.artifacts
Description:
    Provides artifact management utilities for the TruthLens AI system.
    The ArtifactManager handles saving, loading, versioning, and cleanup
    of model artifacts such as trained model weights, tokenizers, vectorizers,
    configuration files, and metadata.

    This module ensures reproducible experiments by storing artifacts in
    structured directories and providing consistent access patterns for
    training, evaluation, and inference pipelines.

Dependencies:
    logging
    pathlib
    json
    shutil
    typing
    torch
    joblib
Inputs:
    Artifact objects (model state dicts, tokenizers, vectorizers, metadata)
Outputs:
    Persisted artifact files and loaded artifact objects
"""
from __future__ import annotations

import hashlib
import json
import logging
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import torch
import torch.distributed as dist

from src.models.export import (
    ONNXExportConfig,
    ONNXExporter,
    QuantizationConfig,
    QuantizationEngine,
    TorchScriptExportConfig,
    TorchScriptExporter,
)

from src.models.metadata.model_card import ModelCard
from src.models.metadata.model_metadata import ModelMetadata
from src.models.metadata.model_versioning import ModelVersionInfo, ModelVersionRegistry


logger = logging.getLogger(__name__)


class AsyncCheckpointWriter:
    def __init__(self, max_queue_size: int = 4) -> None:
        self._queue: queue.Queue[tuple[Path, Any] | None] = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._closed = False
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, payload = item
                obj, compress = payload
                self._save_atomic(path, obj, compress=compress)
            finally:
                self._queue.task_done()

    @staticmethod
    def _save_atomic(path: Path, obj: Any, compress: bool = True) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp_path, _use_new_zipfile_serialization=compress)
        tmp_path.replace(path)

    def save(self, path: Path, obj: Any, compress: bool = True) -> None:
        if self._closed:
            raise RuntimeError("Cannot enqueue save on closed AsyncCheckpointWriter")
        try:
            self._queue.put_nowait((path, (obj, compress)))
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait((path, (obj, compress)))
            except queue.Empty:
                pass

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(None)
        self._thread.join()


# -------------------------------------------------
# Quantization Backend Helper
# -------------------------------------------------

def _quant_backend() -> str:
    supported = list(torch.backends.quantized.supported_engines)

    if not supported:
        raise RuntimeError("No quantization backend available in this PyTorch build")

    if "fbgemm" in supported:
        return "fbgemm"

    if "qnnpack" in supported:
        return "qnnpack"

    return supported[0]


# -------------------------------------------------
# Artifact Manager
# -------------------------------------------------

class ArtifactManager:
    """
    Manages model artifacts including model weights,
    tokenizers, vectorizers, and metadata.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._async_writer = AsyncCheckpointWriter()
        self._last_hashes: dict[str, str] = {}

    def close(self) -> None:
        self._async_writer.close()

    def flush(self) -> None:
        self._async_writer.flush()

    def __del__(self) -> None:
        try:
            self._async_writer.close()
        except Exception:
            pass

    @staticmethod
    def should_save(step: int, save_every: int) -> bool:
        return save_every > 0 and step % save_every == 0

    @staticmethod
    def should_save_global_step(global_step: int, save_every: int) -> bool:
        return save_every > 0 and global_step % save_every == 0

    @staticmethod
    def _extract_state_dict(model_or_state: torch.nn.Module | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(model_or_state, torch.nn.Module):
            model = model_or_state
            if hasattr(model, "_orig_mod"):
                model = model._orig_mod
            state = model.state_dict()
            return {k: v for k, v in state.items() if "attn_mask" not in k}

        return model_or_state

    @staticmethod
    def _to_cpu_state_dict(
        state_dict: Dict[str, Any],
        *,
        pin_memory: bool = True,
        in_place: bool = True,
    ) -> Dict[str, Any]:
        if in_place:
            for key, value in state_dict.items():
                if torch.is_tensor(value):
                    tensor = value.detach().to("cpu", non_blocking=True)
                    state_dict[key] = tensor.pin_memory() if pin_memory else tensor
            return state_dict

        return {
            key: (
                value.detach().to("cpu", non_blocking=True).pin_memory()
                if torch.is_tensor(value) and pin_memory
                else value.detach().to("cpu", non_blocking=True)
            )
            if torch.is_tensor(value)
            else value
            for key, value in state_dict.items()
        }

    @staticmethod
    def _hash_state(state_dict: Dict[str, Any]) -> str:
        hasher = hashlib.md5()
        for key, value in state_dict.items():
            hasher.update(key.encode("utf-8"))
            if torch.is_tensor(value):
                sample = value.detach().to("cpu").flatten()[:10].contiguous()
                hasher.update(sample.numpy().tobytes())
            else:
                hasher.update(repr(value).encode("utf-8"))

        return hasher.hexdigest()

    @staticmethod
    def _is_primary_process() -> bool:
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    @staticmethod
    def _barrier_before_save() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # -------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------

    def save_model(
        self,
        model: torch.nn.Module | Dict[str, Any],
        model_name: str = "model.pt",
        deduplicate: bool = True,
        compress: bool = True,
    ) -> Path:

        path = self.artifact_dir / model_name

        try:
            if not self._is_primary_process():
                logger.info("Skipping model save on non-primary process: %s", path)
                return path

            state_dict = self._extract_state_dict(model)
            cpu_state_dict = self._to_cpu_state_dict(state_dict)

            if deduplicate:
                state_hash = self._hash_state(cpu_state_dict)
                if self._last_hashes.get(model_name) == state_hash:
                    logger.info("Skipping duplicate model save: %s", path)
                    return path
                self._last_hashes[model_name] = state_hash

            self._barrier_before_save()
            self._async_writer.save(path, cpu_state_dict, compress=compress)
            logger.info("Model artifact queued: %s", path)
        except Exception as exc:
            logger.exception("Failed to save model artifact: %s", path)
            raise RuntimeError("Model saving failed") from exc

        return path

    def save_checkpoint(
        self,
        model_state: torch.nn.Module | Dict[str, Any],
        optimizer_state: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
        save_optimizer: bool = False,
        deduplicate: bool = True,
        compress: bool = True,
        include_metadata: bool = True,
    ) -> Path:

        name = "checkpoint.pt" if step is None else f"checkpoint_{step}.pt"
        path = self.artifact_dir / name

        try:
            if not self._is_primary_process():
                logger.info("Skipping checkpoint save on non-primary process: %s", path)
                return path

            state_dict = self._extract_state_dict(model_state)
            cpu_state_dict = self._to_cpu_state_dict(state_dict)

            if deduplicate:
                state_hash = self._hash_state(cpu_state_dict)
                if self._last_hashes.get(name) == state_hash:
                    logger.info("Skipping duplicate checkpoint save: %s", path)
                    return path
                self._last_hashes[name] = state_hash

            payload: Dict[str, Any] = dict(model=cpu_state_dict)
            if save_optimizer and optimizer_state is not None:
                payload["optimizer"] = optimizer_state
            if include_metadata:
                payload["step"] = step
                payload["timestamp"] = time.time()

            self._barrier_before_save()
            self._async_writer.save(path, payload, compress=compress)
            logger.info("Checkpoint queued: %s", path)

        except Exception as exc:
            logger.exception("Failed to save checkpoint: %s", path)
            raise RuntimeError("Checkpoint saving failed") from exc

        return path

    def save_sharded(
        self,
        model_state: torch.nn.Module | Dict[str, Any],
        shards: int = 4,
        deduplicate: bool = True,
        compress: bool = True,
    ) -> list[Path]:

        if shards <= 1:
            return [self.save_model(model_state, deduplicate=deduplicate)]

        if not self._is_primary_process():
            logger.info("Skipping sharded save on non-primary process")
            return []

        state_dict = self._extract_state_dict(model_state)
        cpu_state_dict = self._to_cpu_state_dict(state_dict)

        if deduplicate:
            state_hash = self._hash_state(cpu_state_dict)
            if self._last_hashes.get("sharded") == state_hash:
                logger.info("Skipping duplicate sharded save")
                return []
            self._last_hashes["sharded"] = state_hash

        items = list(cpu_state_dict.items())
        shard_size = max(1, len(items) // shards)
        paths: list[Path] = []
        self._barrier_before_save()

        for i in range(shards):
            start = i * shard_size
            end = len(items) if i == shards - 1 else (i + 1) * shard_size
            shard = dict(items[start:end])
            path = self.artifact_dir / f"shard_{i}.pt"
            self._async_writer.save(path, shard, compress=compress)
            paths.append(path)

        return paths

    def save_tokenizer(
        self,
        tokenizer: Any,
        directory_name: str = "tokenizer",
    ) -> Path:

        path = self.artifact_dir / directory_name
        path.mkdir(parents=True, exist_ok=True)

        try:
            tokenizer.save_pretrained(path)
            logger.info("Tokenizer saved: %s", path)
        except Exception as exc:
            logger.exception("Failed to save tokenizer: %s", path)
            raise RuntimeError("Tokenizer saving failed") from exc

        return path

    def save_vectorizer(
        self,
        vectorizer: Any,
        file_name: str = "vectorizer.joblib",
    ) -> Path:

        path = self.artifact_dir / file_name

        try:
            joblib.dump(vectorizer, path)
            logger.info("Vectorizer saved: %s", path)
        except Exception as exc:
            logger.exception("Failed to save vectorizer: %s", path)
            raise RuntimeError("Vectorizer saving failed") from exc

        return path

    def save_metadata(
        self,
        metadata: Dict[str, Any],
        file_name: str = "generic_metadata.json",
    ) -> Path:

        path = self.artifact_dir / file_name

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info("Metadata saved: %s", path)

        except Exception as exc:
            logger.exception("Failed to save metadata: %s", path)
            raise RuntimeError("Metadata saving failed") from exc

        return path

    # -------------------------------------------------
    # Export Methods
    # -------------------------------------------------

    def export_onnx(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        file_name: str = "model.onnx",
        config: ONNXExportConfig | None = None,
    ) -> Path:

        path = self.artifact_dir / file_name
        exporter = ONNXExporter(config=config)

        try:
            model.eval()
            exporter.export(
                model=model,
                example_input=example_input,
                output_path=path
            )

            logger.info("ONNX artifact exported: %s", path)

        except Exception as exc:
            logger.exception("Failed to export ONNX artifact: %s", path)
            raise RuntimeError("ONNX export failed") from exc

        return path

    def export_torchscript(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        file_name: str = "model.ts.pt",
        config: TorchScriptExportConfig | None = None,
    ) -> Path:

        path = self.artifact_dir / file_name
        exporter = TorchScriptExporter(config=config)

        try:
            model.eval()
            exporter.export(
                model=model,
                example_input=example_input,
                output_path=path
            )

            logger.info("TorchScript artifact exported: %s", path)

        except Exception as exc:
            logger.exception("Failed to export TorchScript artifact: %s", path)
            raise RuntimeError("TorchScript export failed") from exc

        return path

    def export_quantized_model(
        self,
        model: torch.nn.Module,
        file_name: str = "model.quantized.pt",
        config: QuantizationConfig | None = None,
    ) -> Path:

        path = self.artifact_dir / file_name

        if config is None:
            config = QuantizationConfig(
                method="dynamic",
                device="cpu",
                backend=_quant_backend(),
            )

        engine = QuantizationEngine(config=config)

        try:
            quantized_model = engine.apply(model)

            torch.save(quantized_model, path)

            logger.info("Quantized model artifact saved: %s", path)

        except Exception as exc:
            logger.exception("Failed to export quantized model: %s", path)
            raise RuntimeError("Quantized export failed") from exc

        return path

    # -------------------------------------------------
    # Load Artifacts
    # -------------------------------------------------

    def load_model(
        self,
        file_name: str = "model.pt",
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:

        path = self.artifact_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        try:
            model_state = torch.load(path, map_location=map_location)

            logger.info("Model artifact loaded: %s", path)

            return model_state

        except Exception as exc:
            logger.exception("Failed to load model artifact: %s", path)
            raise RuntimeError("Model loading failed") from exc

    def load_vectorizer(
        self,
        file_name: str = "vectorizer.joblib",
    ) -> Any:

        path = self.artifact_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"Vectorizer artifact not found: {path}")

        try:
            vectorizer = joblib.load(path)

            logger.info("Vectorizer loaded: %s", path)

            return vectorizer

        except Exception as exc:
            logger.exception("Failed to load vectorizer: %s", path)
            raise RuntimeError("Vectorizer loading failed") from exc

    def load_metadata(
        self,
        file_name: str = "generic_metadata.json",
    ) -> Dict[str, Any]:

        path = self.artifact_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            logger.info("Metadata loaded: %s", path)

            return metadata

        except Exception as exc:
            logger.exception("Failed to load metadata: %s", path)
            raise RuntimeError("Metadata loading failed") from exc

    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------

    def delete_artifact(self, name: str) -> None:

        path = self.artifact_dir / name

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        try:

            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

            logger.info("Artifact deleted: %s", path)

        except Exception as exc:
            logger.exception("Failed to delete artifact: %s", path)
            raise RuntimeError("Artifact deletion failed") from exc

    def list_artifacts(self) -> Dict[str, Path]:

        return {p.name: p for p in self.artifact_dir.iterdir()}

    # -------------------------------------------------
    # Model Card
    # -------------------------------------------------

    def save_model_card(
        self,
        card: ModelCard,
        file_name: str = "model_card.json",
        markdown_file_name: str = "model_card.md",
    ) -> Path:

        json_path = self.artifact_dir / file_name
        md_path = self.artifact_dir / markdown_file_name

        try:

            card.save_json(json_path)
            card.save_markdown(md_path)

            logger.info("ModelCard saved: %s , %s", json_path, md_path)

        except Exception as exc:
            logger.exception("Failed to save ModelCard")
            raise RuntimeError("ModelCard saving failed") from exc

        return json_path

    def load_model_card(self, file_name: str = "model_card.json") -> Dict[str, Any]:

        path = self.artifact_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"ModelCard file not found: {path}")

        try:

            with open(path, "r", encoding="utf-8") as f:
                card_dict = json.load(f)

            logger.info("ModelCard loaded: %s", path)

            return card_dict

        except Exception as exc:
            logger.exception("Failed to load ModelCard")
            raise RuntimeError("ModelCard loading failed") from exc

    # -------------------------------------------------
    # Model Metadata
    # -------------------------------------------------

    def save_model_metadata(
        self,
        metadata: ModelMetadata,
        file_name: str = "model_metadata.json",
    ) -> Path:

        path = self.artifact_dir / file_name

        try:

            saved = metadata.save_json(path)

            logger.info("ModelMetadata saved: %s", saved)

            return saved

        except Exception as exc:
            logger.exception("Failed to save ModelMetadata")
            raise RuntimeError("ModelMetadata saving failed") from exc

    def load_model_metadata(
        self,
        file_name: str = "model_metadata.json",
    ) -> ModelMetadata:

        path = self.artifact_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"ModelMetadata file not found: {path}")

        try:

            metadata = ModelMetadata.load_json(path)

            logger.info("ModelMetadata loaded: %s", path)

            return metadata

        except Exception as exc:
            logger.exception("Failed to load ModelMetadata")
            raise RuntimeError("ModelMetadata loading failed") from exc

    # -------------------------------------------------
    # Version Registry
    # -------------------------------------------------

    def register_model_version(
        self,
        version_info: ModelVersionInfo,
        registry_dir: Optional[str | Path] = None,
    ) -> Path:

        target_dir = Path(registry_dir) if registry_dir else self.artifact_dir

        try:

            registry = ModelVersionRegistry(target_dir)

            version_path = registry.register_version(version_info)

            logger.info(
                "Model version registered: %s -> %s",
                version_info.version,
                version_path,
            )

            return version_path

        except Exception as exc:
            logger.exception("Failed to register model version")
            raise RuntimeError("Model version registration failed") from exc