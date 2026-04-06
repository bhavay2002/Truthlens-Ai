"""
File Name: model_config.py
Module: models.config
Description:
    Defines configuration structures and utilities for TruthLens models.
    This module centralizes model configuration management including
    architecture parameters, encoder settings, training parameters,
    and artifact metadata.

    Configurations are typically loaded from YAML files and converted
    into strongly-typed dataclasses for safer use across the training,
    evaluation, and inference pipelines.

Dependencies:
    dataclasses
    typing
    pathlib
    yaml
Inputs:
    YAML configuration files or configuration dictionaries
Outputs:
    Structured model configuration dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


# ---------------------------------------------------------
# Encoder Configuration
# ---------------------------------------------------------


@dataclass
class EncoderConfig:
    """
    Configuration for transformer encoder.
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    device: Optional[str] = None


# ---------------------------------------------------------
# Head Configuration
# ---------------------------------------------------------


@dataclass
class HeadConfig:
    """
    Configuration for task-specific heads.
    """

    input_dim: int
    output_dim: int
    dropout: float = 0.1


@dataclass
class RegressionConfig:
    """
    Optional configuration for attaching a regression head to a task.
    """

    enabled: bool = False
    output_dim: int = 1
    hidden_dim: Optional[int] = None
    activation: str = "gelu"
    dropout: float = 0.1


# ---------------------------------------------------------
# Task Configuration
# ---------------------------------------------------------


@dataclass
class TaskConfig:
    """
    Configuration for a single task.
    """

    name: str
    num_labels: int
    task_type: str = "multi_class"
    regression: Optional[RegressionConfig] = None


# ---------------------------------------------------------
# MultiTask Model Configuration
# ---------------------------------------------------------


@dataclass
class MultiTaskModelConfig:
    """
    Configuration for the TruthLens multi-task model.
    """

    encoder: EncoderConfig
    tasks: Dict[str, TaskConfig]
    dropout: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------
# Config Loader
# ---------------------------------------------------------


class ModelConfigLoader:
    """
    Utility class for loading model configurations from YAML files.
    """

    @staticmethod
    def load_yaml(config_path: str | Path) -> Dict[str, Any]:
        """
        Load raw YAML configuration.
        """

        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_multitask_config(config_path: str | Path) -> MultiTaskModelConfig:
        """
        Load and convert YAML config into MultiTaskModelConfig.
        """

        raw_config = ModelConfigLoader.load_yaml(config_path)

        encoder_cfg = EncoderConfig(**raw_config["encoder"])

        tasks_cfg = {
            name: TaskConfig(
                name=name,
                num_labels=task_data["num_labels"],
                task_type=task_data.get("task_type", "multi_class"),
                regression=(
                    RegressionConfig(**task_data["regression"])
                    if isinstance(task_data.get("regression"), dict)
                    else None
                ),
            )
            for name, task_data in raw_config["tasks"].items()
        }

        return MultiTaskModelConfig(
            encoder=encoder_cfg,
            tasks=tasks_cfg,
            dropout=raw_config.get("dropout", 0.1),
            metadata=raw_config.get("metadata", {}),
        )