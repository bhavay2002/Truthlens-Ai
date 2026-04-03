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

logger = logging.getLogger(__name__)


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
            model_state = torch.load(path, map_location=map_location)
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