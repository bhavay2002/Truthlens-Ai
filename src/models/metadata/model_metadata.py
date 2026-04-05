"""
File Name: model_metadata.py
Module: model_management
Description:
    Provides structured metadata management for machine learning models.
    This module defines standardized metadata objects used to describe
    model identity, training lineage, experiment provenance, artifact
    tracking, and runtime characteristics.

    The metadata system enables reproducibility, experiment tracking,
    model registry integration, and auditability in large-scale ML systems.
    Metadata objects can be serialized and stored alongside model artifacts
    or registered in experiment tracking systems.

Dependencies:
    dataclasses
    datetime
    json
    logging
    pathlib
    typing
    uuid
Inputs:
    Model identity information, training parameters, dataset lineage,
    artifact paths, and runtime environment information.
Outputs:
    Serialized metadata files (JSON) and in-memory metadata structures.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


def _validate_non_empty(value: str, field_name: str) -> None:
    """Validate that a required string field is non-empty."""
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty.")


def _ensure_directory(path: Path) -> None:
    """Ensure that the parent directory exists."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.exception("Failed to create metadata directory.")
        raise OSError(f"Unable to create directory: {path.parent}") from exc


@dataclass
class ModelIdentity:
    """
    Core identification fields for a model artifact.
    """

    model_name: str
    version: str
    architecture: str
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        _validate_non_empty(self.model_name, "model_name")
        _validate_non_empty(self.version, "version")
        _validate_non_empty(self.architecture, "architecture")


@dataclass
class TrainingProvenance:
    """
    Describes how the model was trained and under which experiment conditions.
    """

    dataset_name: str
    dataset_version: Optional[str]
    experiment_name: Optional[str]
    run_id: Optional[str]
    framework: str
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_non_empty(self.dataset_name, "dataset_name")
        _validate_non_empty(self.framework, "framework")


@dataclass
class ArtifactPaths:
    """
    Tracks file system locations of model-related artifacts.
    """

    model_weights: Optional[str]
    config_file: Optional[str]
    tokenizer_path: Optional[str]
    training_logs: Optional[str]
    checkpoint_directory: Optional[str]

    def validate_paths(self) -> None:
        """Validate that artifact paths exist when provided."""
        paths = [
            self.model_weights,
            self.config_file,
            self.tokenizer_path,
            self.training_logs,
            self.checkpoint_directory,
        ]

        for p in paths:
            if p is None:
                continue

            path_obj = Path(p)
            if not path_obj.exists():
                logger.warning("Artifact path does not exist: %s", p)


@dataclass
class RuntimeEnvironment:
    """
    Describes the runtime environment used during training.
    """

    python_version: str
    framework_version: Optional[str]
    cuda_version: Optional[str]
    hardware: Optional[str]
    device_count: Optional[int]

    def __post_init__(self) -> None:
        _validate_non_empty(self.python_version, "python_version")


@dataclass
class ModelMetadata:
    """
    Aggregates all metadata related to a machine learning model.

    This metadata structure is designed for integration with model registries,
    artifact stores, and experiment tracking systems. It ensures that each
    model artifact has a fully traceable lineage and environment description.
    """

    identity: ModelIdentity
    provenance: TrainingProvenance
    artifacts: ArtifactPaths
    runtime: RuntimeEnvironment
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[Dict[str, str]] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata object to dictionary."""
        try:
            data = asdict(self)

            if self.metrics is not None:
                for key, value in self.metrics.items():
                    if not isinstance(value, (float, int)):
                        raise ValueError(
                            f"Metric '{key}' must be numeric but got {type(value)}"
                        )

            return data

        except Exception as exc:
            logger.exception("Failed to convert ModelMetadata to dict.")
            raise RuntimeError("ModelMetadata serialization failure.") from exc

    def save_json(self, path: str | Path) -> Path:
        """
        Save metadata to a JSON file.

        Parameters
        ----------
        path : str | Path
            Target path for serialized metadata.

        Returns
        -------
        Path
            Path to the saved metadata file.
        """

        output_path = Path(path)

        _ensure_directory(output_path)

        try:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)

        except Exception as exc:
            logger.exception("Failed to write model metadata JSON.")
            raise IOError(f"Could not write metadata to {output_path}") from exc

        logger.info("Model metadata saved to %s", output_path)

        return output_path

    @classmethod
    def load_json(cls, path: str | Path) -> "ModelMetadata":
        """
        Load metadata from a JSON file.

        Parameters
        ----------
        path : str | Path
            Metadata file path.

        Returns
        -------
        ModelMetadata
        """

        metadata_path = Path(path)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception as exc:
            logger.exception("Failed reading metadata file.")
            raise IOError(f"Could not read metadata file: {metadata_path}") from exc

        try:
            identity = ModelIdentity(**data["identity"])
            provenance = TrainingProvenance(**data["provenance"])
            artifacts = ArtifactPaths(**data["artifacts"])
            runtime = RuntimeEnvironment(**data["runtime"])

            metadata = cls(
                identity=identity,
                provenance=provenance,
                artifacts=artifacts,
                runtime=runtime,
                metrics=data.get("metrics"),
                tags=data.get("tags"),
                extra=data.get("extra"),
            )

            artifacts.validate_paths()

            return metadata

        except Exception as exc:
            logger.exception("Metadata structure validation failed.")
            raise ValueError("Invalid metadata structure.") from exc