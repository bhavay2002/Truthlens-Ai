"""
File Name: model_versioning.py
Module: model_management
Description:
    Provides a production-grade model versioning system used to manage
    machine learning model artifacts in large-scale ML systems.

    The module implements structured version control for trained models,
    including version registration, artifact tracking, metadata storage,
    and version history management.

    This system enables reproducibility, auditability, and controlled
    deployment workflows by maintaining a registry of model versions
    and their associated artifacts.

Dependencies:
    dataclasses
    datetime
    json
    logging
    pathlib
    typing
    uuid
Inputs:
    Model artifacts, metadata dictionaries, and version identifiers.
Outputs:
    Versioned model directories, version registry files, and metadata records.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


logger = logging.getLogger(__name__)


def _ensure_directory(path: Path) -> None:
    """Ensure that the specified directory exists."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.exception("Failed to create directory: %s", path)
        raise OSError(f"Unable to create directory: {path}") from exc


def _validate_non_empty(value: str, field_name: str) -> None:
    """Validate that a string value is not empty."""
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty.")


@dataclass
class ModelVersionInfo:
    """
    Metadata describing a specific model version.
    """

    model_name: str
    version: str
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    description: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    artifact_path: Optional[str] = None
    tags: Optional[List[str]] = None

    def __post_init__(self) -> None:
        _validate_non_empty(self.model_name, "model_name")
        _validate_non_empty(self.version, "version")

        if self.metrics is not None:
            for key, value in self.metrics.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Metric '{key}' must be numeric but received {type(value)}"
                    )


class ModelVersionRegistry:
    """
    Manages model version registration and retrieval.

    The registry stores metadata for each model version in a structured
    JSON index file. Each version is stored within its own directory
    containing artifacts and metadata.
    """

    REGISTRY_FILENAME = "model_registry.json"

    def __init__(self, registry_dir: str | Path) -> None:
        """
        Initialize the model version registry.

        Parameters
        ----------
        registry_dir : str | Path
            Directory where model versions and registry metadata are stored.
        """

        self.registry_dir = Path(registry_dir)
        _ensure_directory(self.registry_dir)

        self.registry_file = self.registry_dir / self.REGISTRY_FILENAME

        if not self.registry_file.exists():
            self._initialize_registry()

    def _initialize_registry(self) -> None:
        """Create an empty registry file."""
        try:
            with self.registry_file.open("w", encoding="utf-8") as f:
                json.dump({"models": {}}, f, indent=4)
        except Exception as exc:
            logger.exception("Failed to initialize registry.")
            raise IOError("Could not initialize model registry.") from exc

    def _load_registry(self) -> Dict:
        """Load registry contents."""
        try:
            with self.registry_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.exception("Failed to load registry.")
            raise IOError("Unable to read registry file.") from exc

    def _save_registry(self, registry_data: Dict) -> None:
        """Save registry data."""
        try:
            with self.registry_file.open("w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=4)
        except Exception as exc:
            logger.exception("Failed to save registry.")
            raise IOError("Unable to write registry file.") from exc

    def register_version(self, version_info: ModelVersionInfo) -> Path:
        """
        Register a new model version.

        Parameters
        ----------
        version_info : ModelVersionInfo
            Metadata describing the model version.

        Returns
        -------
        Path
            Path to the created model version directory.
        """

        registry = self._load_registry()

        model_name = version_info.model_name

        if model_name not in registry["models"]:
            registry["models"][model_name] = []

        version_directory = (
            self.registry_dir
            / model_name
            / f"version_{version_info.version}"
        )

        _ensure_directory(version_directory)

        version_info.artifact_path = str(version_directory)

        registry["models"][model_name].append(asdict(version_info))

        self._save_registry(registry)

        metadata_file = version_directory / "version_metadata.json"

        try:
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(asdict(version_info), f, indent=4)
        except Exception as exc:
            logger.exception("Failed to write version metadata.")
            raise IOError("Unable to store version metadata.") from exc

        logger.info(
            "Registered model version: %s (version=%s)",
            version_info.model_name,
            version_info.version,
        )

        return version_directory

    def list_versions(self, model_name: str) -> List[ModelVersionInfo]:
        """
        List all versions of a given model.

        Parameters
        ----------
        model_name : str

        Returns
        -------
        List[ModelVersionInfo]
        """

        _validate_non_empty(model_name, "model_name")

        registry = self._load_registry()

        if model_name not in registry["models"]:
            return []

        versions: List[ModelVersionInfo] = []

        for entry in registry["models"][model_name]:
            versions.append(ModelVersionInfo(**entry))

        return versions

    def get_latest_version(self, model_name: str) -> Optional[ModelVersionInfo]:
        """
        Retrieve the latest registered version of a model.

        Parameters
        ----------
        model_name : str

        Returns
        -------
        Optional[ModelVersionInfo]
        """

        versions = self.list_versions(model_name)

        if not versions:
            return None

        versions_sorted = sorted(
            versions,
            key=lambda v: v.created_at,
            reverse=True,
        )

        return versions_sorted[0]

    def get_version(
        self, model_name: str, version: str
    ) -> Optional[ModelVersionInfo]:
        """
        Retrieve a specific model version.

        Parameters
        ----------
        model_name : str
        version : str

        Returns
        -------
        Optional[ModelVersionInfo]
        """

        versions = self.list_versions(model_name)

        for v in versions:
            if v.version == version:
                return v

        return None