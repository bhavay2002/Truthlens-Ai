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


# =========================================================
# Encoder Configuration
# =========================================================

@dataclass
class EncoderConfig:
    model_name: str = "roberta-base"
    pooling: str = "cls"
    device: Optional[str] = None

    # -------- Memory / Performance --------
    gradient_checkpointing: bool = False
    enable_fused_attention: bool = True

    # -------- Precision --------
    use_amp: bool = True
    amp_dtype: str = "bf16"  # "fp16" | "bf16"

    # -------- Compilation --------
    use_compile: bool = False
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"


# =========================================================
# Head Configuration
# =========================================================

@dataclass
class HeadConfig:
    input_dim: int
    output_dim: int
    dropout: float = 0.1

    # Optimization
    use_layernorm: bool = False


@dataclass
class RegressionConfig:
    enabled: bool = False
    output_dim: int = 1
    hidden_dim: Optional[int] = None
    activation: str = "gelu"
    dropout: float = 0.1


# =========================================================
# Task Configuration
# =========================================================

@dataclass
class TaskConfig:
    name: str
    num_labels: int
    task_type: str = "multi_class"
    regression: Optional[RegressionConfig] = None

    # Optimization
    use_label_smoothing: bool = False


# =========================================================
# MultiTask Model Configuration
# =========================================================

@dataclass
class MultiTaskModelConfig:

    encoder: EncoderConfig
    tasks: Dict[str, TaskConfig]

    dropout: float = 0.1

    # -------- Shared Optimization --------
    shared_encoder: bool = True  # avoid repeated encoder calls

    # -------- Memory --------
    reduce_intermediate_allocation: bool = True

    # -------- Metadata --------
    metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# Config Loader
# =========================================================

class ModelConfigLoader:

    @staticmethod
    def load_yaml(config_path: str | Path) -> Dict[str, Any]:

        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def load_multitask_config(config_path: str | Path) -> MultiTaskModelConfig:

        raw = ModelConfigLoader.load_yaml(config_path)

        # ---------------- Encoder ----------------

        encoder_cfg = EncoderConfig(**raw.get("encoder", {}))

        # ---------------- Tasks ----------------

        tasks_cfg = {}

        for name, task_data in raw["tasks"].items():

            regression_cfg = None

            if isinstance(task_data.get("regression"), dict):
                regression_cfg = RegressionConfig(**task_data["regression"])

            tasks_cfg[name] = TaskConfig(
                name=name,
                num_labels=task_data["num_labels"],
                task_type=task_data.get("task_type", "multi_class"),
                regression=regression_cfg,
                use_label_smoothing=task_data.get("use_label_smoothing", False),
            )

        # ---------------- Final Config ----------------

        return MultiTaskModelConfig(
            encoder=encoder_cfg,
            tasks=tasks_cfg,
            dropout=raw.get("dropout", 0.1),
            shared_encoder=raw.get("shared_encoder", True),
            reduce_intermediate_allocation=raw.get(
                "reduce_intermediate_allocation", True
            ),
            metadata=raw.get("metadata", {}),
        )