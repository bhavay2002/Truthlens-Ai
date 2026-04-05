"""
File Name: model_card.py
Module: model_management
Description:
    Implements a Model Card system for documenting machine learning models.
    The model card captures essential metadata about a trained model,
    including architecture details, training configuration, evaluation
    metrics, dataset information, ethical considerations, and intended use.

    This module enables reproducible ML workflows by storing structured
    documentation alongside trained artifacts. Model cards can be serialized
    to JSON or Markdown formats for integration with experiment tracking,
    governance pipelines, and model registries.

Dependencies:
    dataclasses
    datetime
    json
    logging
    pathlib
    typing
Inputs:
    Model metadata, training details, evaluation results, and artifact paths.
Outputs:
    Serialized model card files (JSON / Markdown).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _validate_non_empty(value: str, field_name: str) -> None:
    """Validate that a string field is non-empty."""
    if not value or not value.strip():
        raise ValueError(f"{field_name} cannot be empty.")


def _ensure_directory(path: Path) -> None:
    """Ensure directory exists."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.exception("Failed to create directory for model card.")
        raise OSError(f"Could not create directory: {path.parent}") from exc


@dataclass
class ModelDetails:
    """Core metadata describing the model."""

    name: str
    version: str
    architecture: str
    description: str
    author: str
    license: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        _validate_non_empty(self.name, "Model name")
        _validate_non_empty(self.version, "Model version")
        _validate_non_empty(self.architecture, "Architecture")
        _validate_non_empty(self.description, "Description")
        _validate_non_empty(self.author, "Author")


@dataclass
class DatasetInfo:
    """Information about training or evaluation datasets."""

    name: str
    source: Optional[str] = None
    preprocessing: Optional[str] = None
    size: Optional[int] = None
    features: Optional[List[str]] = None

    def __post_init__(self) -> None:
        _validate_non_empty(self.name, "Dataset name")


@dataclass
class TrainingConfig:
    """Training configuration summary."""

    framework: str
    epochs: int
    batch_size: int
    optimizer: str
    learning_rate: float
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    hardware: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_non_empty(self.framework, "Framework")

        if self.epochs <= 0:
            raise ValueError("Epochs must be positive.")

        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")

        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive.")


@dataclass
class EvaluationResults:
    """Evaluation metrics for the trained model."""

    metrics: Dict[str, float]
    validation_dataset: Optional[str] = None
    test_dataset: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, dict) or not self.metrics:
            raise ValueError("Metrics must be a non-empty dictionary.")

        for metric, value in self.metrics.items():
            if not isinstance(value, (float, int)):
                raise ValueError(f"Metric '{metric}' must be numeric.")


@dataclass
class EthicalConsiderations:
    """Ethical considerations and model limitations."""

    intended_use: str
    limitations: Optional[str] = None
    biases: Optional[str] = None
    risks: Optional[str] = None
    mitigation_strategies: Optional[str] = None

    def __post_init__(self) -> None:
        _validate_non_empty(self.intended_use, "Intended use")


@dataclass
class ModelArtifacts:
    """Paths to model artifacts."""

    model_weights: Optional[str] = None
    tokenizer: Optional[str] = None
    config_file: Optional[str] = None
    training_logs: Optional[str] = None
    checkpoint_dir: Optional[str] = None


@dataclass
class ModelCard:
    """
    Structured representation of a machine learning model card.

    This class aggregates metadata and provides serialization utilities
    for storing documentation alongside model artifacts.
    """

    model_details: ModelDetails
    datasets: List[DatasetInfo]
    training: TrainingConfig
    evaluation: EvaluationResults
    ethics: EthicalConsiderations
    artifacts: ModelArtifacts
    tags: Optional[List[str]] = None
    references: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model card to dictionary."""
        try:
            return asdict(self)
        except Exception as exc:
            logger.exception("Failed converting ModelCard to dictionary.")
            raise RuntimeError("ModelCard serialization failure.") from exc

    def save_json(self, path: str | Path) -> Path:
        """Save model card as JSON."""
        output_path = Path(path)

        _ensure_directory(output_path)

        try:
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=4)
        except Exception as exc:
            logger.exception("Failed to write model card JSON.")
            raise IOError(f"Could not write model card to {output_path}") from exc

        logger.info("Model card saved to JSON: %s", output_path)
        return output_path

    def save_markdown(self, path: str | Path) -> Path:
        """Save model card as Markdown."""
        output_path = Path(path)

        _ensure_directory(output_path)

        data = self.to_dict()

        try:
            with output_path.open("w", encoding="utf-8") as f:
                f.write(f"# Model Card: {data['model_details']['name']}\n\n")

                f.write("## Model Details\n")
                for key, value in data["model_details"].items():
                    f.write(f"- **{key}**: {value}\n")

                f.write("\n## Datasets\n")
                for dataset in data["datasets"]:
                    f.write(f"- **{dataset['name']}**\n")
                    for key, value in dataset.items():
                        if key != "name":
                            f.write(f"  - {key}: {value}\n")

                f.write("\n## Training Configuration\n")
                for key, value in data["training"].items():
                    f.write(f"- **{key}**: {value}\n")

                f.write("\n## Evaluation Results\n")
                for metric, value in data["evaluation"]["metrics"].items():
                    f.write(f"- **{metric}**: {value}\n")

                f.write("\n## Ethical Considerations\n")
                for key, value in data["ethics"].items():
                    f.write(f"- **{key}**: {value}\n")

                if data.get("tags"):
                    f.write("\n## Tags\n")
                    for tag in data["tags"]:
                        f.write(f"- {tag}\n")

                if data.get("references"):
                    f.write("\n## References\n")
                    for ref in data["references"]:
                        f.write(f"- {ref}\n")

        except Exception as exc:
            logger.exception("Failed to write model card Markdown.")
            raise IOError(f"Could not write model card to {output_path}") from exc

        logger.info("Model card saved to Markdown: %s", output_path)
        return output_path