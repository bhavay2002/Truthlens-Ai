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

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import joblib
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


def _quant_backend() -> str:
    supported = list(torch.backends.quantized.supported_engines)
    if "fbgemm" in supported:
        return "fbgemm"
    if "qnnpack" in supported:
        return "qnnpack"
    return supported[0] if supported else "qnnpack"


class ArtifactManager:
    """
    Manages model artifacts including model weights, tokenizers,
    vectorizers, and metadata.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Save Artifacts
    # -------------------------------------------------

    def save_model(
        self,
        model_state_dict: Dict[str, Any],
        model_name: str = "model.pt",
    ) -> Path:
        """
        Save PyTorch model state dictionary.
        """

        path = self.artifact_dir / model_name

        try:
            torch.save(model_state_dict, path)
            logger.info("Model artifact saved: %s", path)
        except Exception as exc:
            logger.exception("Failed to save model artifact")
            raise RuntimeError("Model saving failed") from exc

        return path

    def save_tokenizer(
        self,
        tokenizer: Any,
        directory_name: str = "tokenizer",
    ) -> Path:
        """
        Save HuggingFace tokenizer.
        """

        path = self.artifact_dir / directory_name

        try:
            tokenizer.save_pretrained(path)
            logger.info("Tokenizer saved: %s", path)
        except Exception as exc:
            logger.exception("Failed to save tokenizer")
            raise RuntimeError("Tokenizer saving failed") from exc

        return path

    def save_vectorizer(
        self,
        vectorizer: Any,
        file_name: str = "vectorizer.joblib",
    ) -> Path:
        """
        Save vectorizer object using joblib.
        """

        path = self.artifact_dir / file_name

        try:
            joblib.dump(vectorizer, path)
            logger.info("Vectorizer saved: %s", path)
        except Exception as exc:
            logger.exception("Failed to save vectorizer")
            raise RuntimeError("Vectorizer saving failed") from exc

        return path

    def save_metadata(
        self,
        metadata: Dict[str, Any],
        file_name: str = "metadata.json",
    ) -> Path:
        """
        Save artifact metadata.
        """

        path = self.artifact_dir / file_name

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Metadata saved: %s", path)
        except Exception as exc:
            logger.exception("Failed to save metadata")
            raise RuntimeError("Metadata saving failed") from exc

        return path

    def export_onnx(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        file_name: str = "model.onnx",
        config: ONNXExportConfig | None = None,
    ) -> Path:
        """
        Export model artifact to ONNX format.
        """

        path = self.artifact_dir / file_name
        exporter = ONNXExporter(config=config)

        try:
            exporter.export(model=model, example_input=example_input, output_path=path)
            logger.info("ONNX artifact exported: %s", path)
        except Exception as exc:
            logger.exception("Failed to export ONNX artifact")
            raise RuntimeError("ONNX export failed") from exc

        return path

    def export_torchscript(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        file_name: str = "model.ts.pt",
        config: TorchScriptExportConfig | None = None,
    ) -> Path:
        """
        Export model artifact to TorchScript format.
        """

        path = self.artifact_dir / file_name
        exporter = TorchScriptExporter(config=config)

        try:
            exporter.export(model=model, example_input=example_input, output_path=path)
            logger.info("TorchScript artifact exported: %s", path)
        except Exception as exc:
            logger.exception("Failed to export TorchScript artifact")
            raise RuntimeError("TorchScript export failed") from exc

        return path

    def export_quantized_model(
        self,
        model: torch.nn.Module,
        file_name: str = "model.quantized.pt",
        config: QuantizationConfig | None = None,
    ) -> Path:
        """
        Quantize model and save quantized state dict.
        """

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
            torch.save(quantized_model.state_dict(), path)
            logger.info("Quantized model artifact saved: %s", path)
        except Exception as exc:
            logger.exception("Failed to export quantized model artifact")
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
        """
        Load PyTorch model state dictionary.
        """

        path = self.artifact_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        try:
            model_state = torch.load(path, map_location=map_location, weights_only=True)
            logger.info("Model artifact loaded: %s", path)
            return model_state
        except Exception as exc:
            logger.exception("Failed to load model artifact")
            raise RuntimeError("Model loading failed") from exc

    def load_vectorizer(
        self,
        file_name: str = "vectorizer.joblib",
    ) -> Any:
        """
        Load vectorizer artifact.
        """

        path = self.artifact_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"Vectorizer artifact not found: {path}")

        try:
            vectorizer = joblib.load(path)
            logger.info("Vectorizer loaded: %s", path)
            return vectorizer
        except Exception as exc:
            logger.exception("Failed to load vectorizer")
            raise RuntimeError("Vectorizer loading failed") from exc

    def load_metadata(
        self,
        file_name: str = "metadata.json",
    ) -> Dict[str, Any]:
        """
        Load metadata artifact.
        """

        path = self.artifact_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            logger.info("Metadata loaded: %s", path)

            return metadata

        except Exception as exc:
            logger.exception("Failed to load metadata")
            raise RuntimeError("Metadata loading failed") from exc

    # -------------------------------------------------
    # Cleanup
    # -------------------------------------------------

    def delete_artifact(self, name: str) -> None:
        """
        Delete a specific artifact.
        """

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
            logger.exception("Failed to delete artifact")
            raise RuntimeError("Artifact deletion failed") from exc

    def list_artifacts(self) -> Dict[str, Path]:
        """
        List all artifacts in the directory.
        """

        artifacts: Dict[str, Path] = {}

        for item in self.artifact_dir.iterdir():
            artifacts[item.name] = item

        return artifacts

    # -------------------------------------------------
    # ModelCard Methods
    # -------------------------------------------------

    def save_model_card(
        self,
        card: ModelCard,
        file_name: str = "model_card.json",
        markdown_file_name: str = "model_card.md",
    ) -> Path:
        """
        Serialize a ModelCard to JSON (and optionally Markdown) in the artifact directory.

        Parameters
        ----------
        card : ModelCard
        file_name : str
            JSON output file name.
        markdown_file_name : str
            Markdown output file name.

        Returns
        -------
        Path
            Path to the saved JSON model card.
        """

        json_path = self.artifact_dir / file_name
        md_path = self.artifact_dir / markdown_file_name

        try:
            card.save_json(json_path)
            card.save_markdown(md_path)
            logger.info("ModelCard saved: %s, %s", json_path, md_path)
        except Exception as exc:
            logger.exception("Failed to save ModelCard")
            raise RuntimeError("ModelCard saving failed") from exc

        return json_path

    def load_model_card(self, file_name: str = "model_card.json") -> Dict[str, Any]:
        """
        Load a ModelCard from a JSON file in the artifact directory.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        Dict[str, Any]
            Raw model card dictionary.
        """

        import json as _json

        path = self.artifact_dir / file_name

        if not path.exists():
            raise FileNotFoundError(f"ModelCard file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                card_dict = _json.load(f)
            logger.info("ModelCard loaded: %s", path)
            return card_dict
        except Exception as exc:
            logger.exception("Failed to load ModelCard")
            raise RuntimeError("ModelCard loading failed") from exc

    # -------------------------------------------------
    # ModelMetadata Methods
    # -------------------------------------------------

    def save_model_metadata(
        self,
        metadata: ModelMetadata,
        file_name: str = "metadata.json",
    ) -> Path:
        """
        Serialize a ModelMetadata object to JSON in the artifact directory.

        Parameters
        ----------
        metadata : ModelMetadata
        file_name : str

        Returns
        -------
        Path
        """

        path = self.artifact_dir / file_name

        try:
            saved = metadata.save_json(path)
            logger.info("ModelMetadata saved: %s", saved)
            return saved
        except Exception as exc:
            logger.exception("Failed to save ModelMetadata")
            raise RuntimeError("ModelMetadata saving failed") from exc

    def load_model_metadata(self, file_name: str = "metadata.json") -> ModelMetadata:
        """
        Load a ModelMetadata object from a JSON file in the artifact directory.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        ModelMetadata
        """

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
    # ModelVersionRegistry Methods
    # -------------------------------------------------

    def register_model_version(
        self,
        version_info: ModelVersionInfo,
        registry_dir: Optional[str | Path] = None,
    ) -> Path:
        """
        Register a model version in the version registry.

        Parameters
        ----------
        version_info : ModelVersionInfo
        registry_dir : str | Path, optional
            Directory for the version registry. Defaults to artifact_dir.

        Returns
        -------
        Path
            Path to the registered version directory.
        """

        target_dir = Path(registry_dir) if registry_dir else self.artifact_dir

        try:
            registry = ModelVersionRegistry(target_dir)
            version_path = registry.register_version(version_info)
            logger.info("Model version registered: %s -> %s", version_info.version, version_path)
            return version_path
        except Exception as exc:
            logger.exception("Failed to register model version")
            raise RuntimeError("Model version registration failed") from exc
