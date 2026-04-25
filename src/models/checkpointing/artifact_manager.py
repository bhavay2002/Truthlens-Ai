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


# =========================================================
# ASYNC WRITER
# =========================================================

class AsyncCheckpointWriter:

    def __init__(self, max_queue_size: int = 4) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
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
                self._save_atomic(path, obj, compress)
            finally:
                self._queue.task_done()

    @staticmethod
    def _save_atomic(path: Path, obj: Any, compress: bool = True) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp, _use_new_zipfile_serialization=compress)
        tmp.replace(path)

    def save(self, path: Path, obj: Any, compress: bool = True) -> None:
        if self._closed:
            raise RuntimeError("Writer closed")

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


# =========================================================
# UTILS
# =========================================================

def _quant_backend() -> str:
    supported = list(torch.backends.quantized.supported_engines)

    if not supported:
        raise RuntimeError("No quant backend")

    if "fbgemm" in supported:
        return "fbgemm"

    if "qnnpack" in supported:
        return "qnnpack"

    return supported[0]


# =========================================================
# MANAGER
# =========================================================

class ArtifactManager:

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self._writer = AsyncCheckpointWriter()
        self._hash_cache: Dict[str, str] = {}

    # =====================================================
    # LIFECYCLE
    # =====================================================

    def close(self) -> None:
        self._writer.close()

    def flush(self) -> None:
        self._writer.flush()

    # =====================================================
    # INTERNAL
    # =====================================================

    @staticmethod
    def _is_primary() -> bool:
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _extract_state(model_or_state):
        if isinstance(model_or_state, torch.nn.Module):
            state = model_or_state.state_dict()
            return {k: v for k, v in state.items() if "attn_mask" not in k}
        return model_or_state

    @staticmethod
    def _to_cpu(state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v.detach().cpu() if torch.is_tensor(v) else v
            for k, v in state.items()
        }

    def _hash(self, state: Dict[str, Any]) -> str:
        h = hashlib.md5()
        for k, v in state.items():
            h.update(k.encode())
            if torch.is_tensor(v):
                h.update(v.flatten()[:10].cpu().numpy().tobytes())
        return h.hexdigest()

    # =====================================================
    # SAVE MODEL
    # =====================================================

    def save_model(
        self,
        model: torch.nn.Module | Dict[str, Any],
        name: str = "model.pt",
        deduplicate: bool = True,
    ) -> Path:

        path = self.artifact_dir / name

        if not self._is_primary():
            return path

        state = self._to_cpu(self._extract_state(model))

        if deduplicate:
            h = self._hash(state)
            if self._hash_cache.get(name) == h:
                return path
            self._hash_cache[name] = h

        self._barrier()
        self._writer.save(path, state)

        return path

    # =====================================================
    # CHECKPOINT
    # =====================================================

    def save_checkpoint(
        self,
        model: torch.nn.Module | Dict[str, Any],
        optimizer: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None,
    ) -> Path:

        name = f"checkpoint_{step}.pt" if step else "checkpoint.pt"
        path = self.artifact_dir / name

        if not self._is_primary():
            return path

        state = self._to_cpu(self._extract_state(model))

        payload = {
            "model": state,
            "optimizer": optimizer,
            "step": step,
            "timestamp": time.time(),
        }

        self._barrier()
        self._writer.save(path, payload)

        return path

    # =====================================================
    # EXPORT
    # =====================================================

    def export_onnx(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        name: str = "model.onnx",
        config: Optional[ONNXExportConfig] = None,
    ) -> Path:

        path = self.artifact_dir / name
        exporter = ONNXExporter(config)

        exporter.export(model, example_input, path)

        return path

    def export_torchscript(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        name: str = "model.ts.pt",
        config: Optional[TorchScriptExportConfig] = None,
    ) -> Path:

        path = self.artifact_dir / name
        exporter = TorchScriptExporter(config)

        exporter.export(model, example_input, path)

        return path

    def export_quantized(
        self,
        model: torch.nn.Module,
        name: str = "model.quantized.pt",
        config: Optional[QuantizationConfig] = None,
    ) -> Path:

        path = self.artifact_dir / name

        config = config or QuantizationConfig(
            method="dynamic",
            backend=_quant_backend(),
        )

        engine = QuantizationEngine(config)
        q_model = engine.apply(model)

        torch.save(q_model, path)

        return path

    # =====================================================
    # SAVE AUX
    # =====================================================

    def save_tokenizer(self, tokenizer: Any, name: str = "tokenizer") -> Path:

        path = self.artifact_dir / name
        path.mkdir(parents=True, exist_ok=True)

        tokenizer.save_pretrained(path)
        return path

    def save_vectorizer(self, vectorizer: Any, name: str = "vectorizer.joblib") -> Path:

        path = self.artifact_dir / name
        joblib.dump(vectorizer, path)

        return path

    def save_metadata(self, metadata: Dict[str, Any], name: str = "metadata.json") -> Path:

        path = self.artifact_dir / name

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

        return path

    # =====================================================
    # LOAD
    # =====================================================

    def load_model(self, name: str = "model.pt") -> Dict[str, Any]:

        path = self.artifact_dir / name

        if not path.exists():
            raise FileNotFoundError(path)

        return torch.load(path)

    def load_vectorizer(self, name: str = "vectorizer.joblib") -> Any:

        path = self.artifact_dir / name

        if not path.exists():
            raise FileNotFoundError(path)

        return joblib.load(path)

    def load_metadata(self, name: str = "metadata.json") -> Dict[str, Any]:

        path = self.artifact_dir / name

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path) as f:
            return json.load(f)

    # =====================================================
    # DELETE / LIST
    # =====================================================

    def delete(self, name: str) -> None:

        path = self.artifact_dir / name

        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    def list(self) -> Dict[str, Path]:
        return {p.name: p for p in self.artifact_dir.iterdir()}

    # =====================================================
    # MODEL CARD
    # =====================================================

    def save_model_card(
        self,
        card: ModelCard,
        json_name: str = "model_card.json",
        md_name: str = "model_card.md",
    ) -> Path:

        json_path = self.artifact_dir / json_name
        md_path = self.artifact_dir / md_name

        card.save_json(json_path)
        card.save_markdown(md_path)

        return json_path

    def load_model_card(self, name: str = "model_card.json") -> Dict[str, Any]:

        path = self.artifact_dir / name

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path) as f:
            return json.load(f)

    # =====================================================
    # METADATA
    # =====================================================

    def save_model_metadata(
        self,
        metadata: ModelMetadata,
        name: str = "model_metadata.json",
    ) -> Path:

        return metadata.save_json(self.artifact_dir / name)

    def load_model_metadata(
        self,
        name: str = "model_metadata.json",
    ) -> ModelMetadata:

        return ModelMetadata.load_json(self.artifact_dir / name)

    # =====================================================
    # VERSIONING
    # =====================================================

    def register_version(
        self,
        info: ModelVersionInfo,
        registry_dir: Optional[str | Path] = None,
    ) -> Path:

        registry = ModelVersionRegistry(
            registry_dir or self.artifact_dir
        )

        return registry.register_version(info)