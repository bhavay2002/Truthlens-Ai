"""
File Name: inference_config.py
Module: Inference Configuration Management
Description:
    Defines the runtime configuration system for the TruthLens inference
    pipeline. This module provides structured configuration objects,
    YAML-based configuration loading, validation, and runtime access.

    The configuration controls operational parameters such as device
    placement, batch size, caching behavior, and optional subsystems
    (e.g., graph analysis).

    Example YAML configuration:

        device: cuda
        batch_size: 32
        cache_predictions: true
        use_graph_analysis: true

    This prevents hardcoded parameters inside inference pipelines and
    ensures reproducibility across deployments.

Author: TruthLens AI
Date: 2026-04-02
Dependencies:
    logging
    typing
    dataclasses
    pathlib
    yaml

Inputs:
    YAML configuration file.

Outputs:
    Validated InferenceConfig dataclass instance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """
    Dataclass representing runtime configuration for inference.
    """

    device: str = "cpu"
    batch_size: int = 32
    cache_predictions: bool = False
    use_graph_analysis: bool = True
    max_sequence_length: int = 512
    enable_explainability: bool = True
    enable_entity_graph: bool = True
    enable_narrative_analysis: bool = True
    prediction_timeout: Optional[int] = None


class InferenceConfigLoader:
    """
    Responsible for loading and validating inference configuration files.
    """

    REQUIRED_FIELDS = {
        "device": str,
        "batch_size": int,
        "cache_predictions": bool,
        "use_graph_analysis": bool,
    }

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Inference config file not found: {self.config_path}")

        logger.info("InferenceConfigLoader initialized with %s", self.config_path)

    def load(self) -> InferenceConfig:
        """
        Load configuration from YAML file.
        """

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

        except Exception as exc:
            logger.exception("Failed to read configuration file")
            raise RuntimeError("Could not load inference configuration") from exc

        if config_dict is None:
            config_dict = {}
        if not isinstance(config_dict, dict):
            raise TypeError("Configuration file must define a dictionary")

        self._validate_config(config_dict)
        allowed = {f.name for f in fields(InferenceConfig)}
        filtered = {k: v for k, v in config_dict.items() if k in allowed}
        unknown = sorted(set(config_dict.keys()) - allowed)
        if unknown:
            logger.warning("Ignoring unknown inference config keys: %s", unknown)
        config = InferenceConfig(**filtered)

        logger.info("Inference configuration loaded successfully")

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate required fields and types.
        """

        if not isinstance(config, dict):
            raise TypeError("Configuration file must define a dictionary")

        # Validate only when explicitly set; defaults live in dataclass.
        for field, expected_type in self.REQUIRED_FIELDS.items():

            if field not in config:
                continue

            if not isinstance(config[field], expected_type):
                raise TypeError(
                    f"Invalid type for '{field}'. Expected {expected_type.__name__}"
                )

        if config["batch_size"] <= 0:
            raise ValueError("batch_size must be greater than 0")

        if config["device"] not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, auto")

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> InferenceConfig:
        """
        Create configuration from dictionary.
        """

        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary")

        return InferenceConfig(**config)


def load_inference_config(path: str | Path) -> InferenceConfig:
    """
    Convenience helper for loading inference configuration.
    """

    loader = InferenceConfigLoader(path)
    return loader.load()