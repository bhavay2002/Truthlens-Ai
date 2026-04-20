"""
File Name: encoder_config.py
Module: models.encoder
Description:
    Defines configuration structures and validation utilities for transformer
    encoders used throughout the TruthLens ML system. The module provides
    dataclass-based configuration objects that encapsulate all parameters
    required to initialize and control transformer encoders, including model
    selection, pooling strategy, device placement, freezing behavior, and
    advanced runtime options.

    These configuration objects integrate with the global YAML configuration
    system and ensure strict validation before encoder instantiation.

Author: TruthLens Engineering
Date: 2026-04-02
Dependencies:
    logging
    dataclasses
    typing
Inputs:
    Configuration parameters typically loaded from YAML files.
Outputs:
    Validated encoder configuration dataclasses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


VALID_POOLING_STRATEGIES = {"cls", "mean"}
VALID_MODEL_TYPES = {"transformer"}


@dataclass
class EncoderConfig:
    """
    Configuration dataclass describing a transformer encoder.
    """

    model_type: str = "transformer"
    model_name: str = "roberta-base"

    pooling: str = "cls"

    device: Optional[str] = None

    freeze_encoder: bool = False

    output_hidden_states: bool = False

    gradient_checkpointing: bool = False

    init_from_config_only: bool = False

    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate encoder configuration parameters.
        """

        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("model_type must be a valid string")

        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type '{self.model_type}'. "
                f"Supported types: {VALID_MODEL_TYPES}"
            )

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if self.pooling not in VALID_POOLING_STRATEGIES:
            raise ValueError(
                f"Invalid pooling strategy '{self.pooling}'. "
                f"Supported strategies: {VALID_POOLING_STRATEGIES}"
            )

        if self.device is not None and not isinstance(self.device, str):
            raise ValueError("device must be a string or None")

        if not isinstance(self.freeze_encoder, bool):
            raise ValueError("freeze_encoder must be a boolean")

        if not isinstance(self.output_hidden_states, bool):
            raise ValueError("output_hidden_states must be a boolean")

        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be a boolean")

        if not isinstance(self.init_from_config_only, bool):
            raise ValueError("init_from_config_only must be a boolean")

        if not isinstance(self.extra_kwargs, dict):
            raise ValueError("extra_kwargs must be a dictionary")

        logger.debug("EncoderConfig validated successfully")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EncoderConfig":
        """
        Construct EncoderConfig from a dictionary.

        Args:
            config_dict:
                Dictionary containing encoder configuration parameters.

        Returns:
            EncoderConfig instance.
        """

        if not isinstance(config_dict, dict):
            raise TypeError("config_dict must be a dictionary")

        config = cls(**config_dict)

        config.validate()

        logger.debug("EncoderConfig created from dictionary")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary representation.

        Returns:
            Dictionary containing configuration parameters.
        """

        config_dict: Dict[str, Any] = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "pooling": self.pooling,
            "device": self.device,
            "freeze_encoder": self.freeze_encoder,
            "output_hidden_states": self.output_hidden_states,
            "gradient_checkpointing": self.gradient_checkpointing,
            "init_from_config_only": self.init_from_config_only,
            "extra_kwargs": self.extra_kwargs,
        }

        return config_dict

    def summary(self) -> Dict[str, Any]:
        """
        Return a lightweight summary of encoder configuration.

        Useful for experiment tracking and logging.

        Returns:
            Dictionary summary of key parameters.
        """

        summary = {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "pooling": self.pooling,
            "freeze_encoder": self.freeze_encoder,
        }

        logger.debug("EncoderConfig summary generated")

        return summary