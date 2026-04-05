"""
File Name: config_loader.py
Module: src.utils
Description:
    Production-grade YAML configuration loader for TruthLens AI.

    This module provides utilities for loading, validating, and accessing
    configuration values defined in YAML files. It supports deterministic
    configuration loading, nested key retrieval, path resolution, and
    conversion of configuration dictionaries into structured dataclasses.

Author: TruthLens Engineering
Date: 2026-04-03
Dependencies:
    - Python 3.10+
    - PyYAML

Inputs:
    - YAML configuration file

Outputs:
    - Parsed configuration dictionary
    - Dataclass-based configuration objects
    - Resolved filesystem paths
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Project Root
# ---------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


# ---------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    gradient_accumulation_steps: int = 1
    device: str = "auto"


@dataclass(slots=True)
class DatasetConfig:
    train_path: Path
    validation_path: Path
    test_path: Path
    text_column: str = "text"
    label_column: str = "label"


@dataclass(slots=True)
class ModelConfig:
    name: str
    pretrained_model: Optional[str] = None
    hidden_size: Optional[int] = None
    num_labels: Optional[int] = None


@dataclass(slots=True)
class ExperimentConfig:
    seed: int
    output_dir: Path
    experiment_name: str


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    experiment: ExperimentConfig


# ---------------------------------------------------------
# Path Resolver
# ---------------------------------------------------------


def _resolve_path(path_value: str | Path) -> Path:
    """
    Convert relative paths from configuration files into absolute paths.

    Parameters
    ----------
    path_value : str | Path
        Path value from YAML configuration.

    Returns
    -------
    Path
        Absolute filesystem path.
    """

    path_obj = Path(path_value)

    if path_obj.is_absolute():
        return path_obj

    return (PROJECT_ROOT / path_obj).resolve()


# ---------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------


@lru_cache(maxsize=4)
def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Uses LRU caching to prevent repeated disk reads.

    Parameters
    ----------
    config_path : Optional[str | Path]
        Path to configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If configuration file does not exist.

    yaml.YAMLError
        If YAML parsing fails.
    """

    resolved_path = _resolve_path(config_path or DEFAULT_CONFIG_PATH)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

    logger.info("Loading configuration from %s", resolved_path)

    try:
        with resolved_path.open("r", encoding="utf-8") as config_file:
            config: Dict[str, Any] = yaml.safe_load(config_file) or {}
    except yaml.YAMLError as exc:
        logger.exception("Failed to parse YAML configuration")
        raise RuntimeError("Invalid YAML configuration file") from exc

    return config


# ---------------------------------------------------------
# Nested Config Access
# ---------------------------------------------------------


def get_config_value(
    config: Dict[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """
    Retrieve nested configuration value safely.

    Parameters
    ----------
    config : Dict[str, Any]
        Loaded configuration dictionary
    keys : str
        Nested keys
    default : Any
        Default value if key path is missing

    Returns
    -------
    Any
        Retrieved configuration value
    """

    current: Any = config

    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default

        current = current[key]

    return current


# ---------------------------------------------------------
# Path Retrieval
# ---------------------------------------------------------


def get_path(
    config: Dict[str, Any],
    *keys: str,
    default: str | Path,
) -> Path:
    """
    Retrieve path value from configuration and resolve it.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    keys : str
        Nested config keys
    default : str | Path
        Default path

    Returns
    -------
    Path
        Absolute path
    """

    value = get_config_value(config, *keys, default=default)

    return _resolve_path(value)


# ---------------------------------------------------------
# Config Validation
# ---------------------------------------------------------


def _validate_required_keys(config: Dict[str, Any], required: list[str]) -> None:
    """
    Validate presence of required top-level configuration keys.

    Parameters
    ----------
    config : Dict[str, Any]
        Loaded configuration
    required : list[str]
        Required keys
    """

    missing = [key for key in required if key not in config]

    if missing:
        raise ValueError(f"Missing required config sections: {missing}")


# ---------------------------------------------------------
# Dataclass Conversion
# ---------------------------------------------------------


def load_app_config(config_path: str | Path | None = None) -> AppConfig:
    """
    Load configuration and convert to structured dataclasses.

    Parameters
    ----------
    config_path : Optional[str | Path]

    Returns
    -------
    AppConfig
        Fully structured application configuration
    """

    config = load_config(config_path)

    _validate_required_keys(config, ["model", "data", "training"])

    data_section = config["data"]
    dataset_cfg = DatasetConfig(
        train_path=_resolve_path(data_section.get("train_path", "data/splits/train.csv")),
        validation_path=_resolve_path(data_section.get("validation_path", "data/splits/validation.csv")),
        test_path=_resolve_path(data_section.get("test_path", "data/splits/test.csv")),
        text_column=data_section.get("text_column", "text"),
        label_column=data_section.get("label_column", "label"),
    )

    model_section = config["model"]
    encoder_section = model_section.get("encoder", {})
    model_name = encoder_section.get("name") or model_section.get("name", "roberta-base")
    model_cfg = ModelConfig(
        name=model_name,
        pretrained_model=encoder_section.get("tokenizer_name") or model_section.get("pretrained_model"),
        hidden_size=model_section.get("hidden_size"),
        num_labels=model_section.get("num_labels"),
    )

    training_cfg = TrainingConfig(
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        gradient_accumulation_steps=config["training"].get(
            "gradient_accumulation_steps", 1
        ),
        device=config["training"].get("device", "auto"),
    )

    experiment_section = config.get("experiment", {})
    experiment_cfg = ExperimentConfig(
        seed=experiment_section.get("seed", config["training"].get("seed", 42)),
        output_dir=_resolve_path(experiment_section.get("output_dir", "models")),
        experiment_name=experiment_section.get("experiment_name", "truthlens"),
    )

    logger.info("Configuration successfully loaded and validated")

    return AppConfig(
        model=model_cfg,
        dataset=dataset_cfg,
        training=training_cfg,
        experiment=experiment_cfg,
    )