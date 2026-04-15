"""
File Name: encoder_factory.py
Module: models.encoder
Description:
    Implements a factory responsible for constructing transformer encoders used
    throughout the TruthLens ML system. The factory centralizes encoder creation
    logic, allowing consistent initialization, configuration validation, and
    device placement. It supports HuggingFace transformer backbones and provides
    a clean abstraction for model components that require encoders.

    This module prevents duplicated encoder initialization logic across
    pipelines and ensures standardized behavior across training, evaluation,
    and inference environments.

Dependencies:
    logging
    typing
    dataclasses
    torch
    transformers
    models.encoder.transformer_encoder
Inputs:
    Encoder configuration parameters (model name, pooling strategy, device, etc.)
Outputs:
    Initialized TransformerEncoder instances
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from .transformer_encoder import TransformerEncoder
from .encoder_config import EncoderConfig as FactoryEncoderConfig
from ..config import (
    EncoderConfig as ModelEncoderConfig,
    ModelConfigLoader,
    MultiTaskModelConfig,
)

logger = logging.getLogger(__name__)


# Backward-compatible alias.
EncoderConfig = FactoryEncoderConfig


class EncoderFactory:
    """
    Factory class responsible for constructing encoder modules.

    The factory enables consistent encoder initialization across the system
    and allows easy swapping of backbone models.
    """

    SUPPORTED_ENCODERS = {"transformer"}

    @staticmethod
    def create_transformer_encoder(config: EncoderConfig) -> TransformerEncoder:
        """
        Create a TransformerEncoder instance.

        Args:
            config:
                Encoder configuration dataclass.

        Returns:
            Initialized TransformerEncoder.
        """

        if not isinstance(config, EncoderConfig):
            raise TypeError("config must be an instance of EncoderConfig")

        config.validate()

        logger.info(
            "Creating transformer encoder | model=%s | pooling=%s",
            config.model_name,
            config.pooling,
        )

        encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
            freeze_encoder=config.freeze_encoder,
            gradient_checkpointing=config.gradient_checkpointing,
        )

        return encoder

    @staticmethod
    def create_from_name(
        encoder_type: str,
        config: EncoderConfig,
    ) -> TransformerEncoder:
        """
        Create encoder based on a string identifier.

        Args:
            encoder_type:
                Encoder type name.
            config:
                Encoder configuration.

        Returns:
            Initialized encoder instance.
        """

        if not isinstance(encoder_type, str) or not encoder_type.strip():
            raise ValueError("encoder_type must be a valid string")

        encoder_type = encoder_type.lower()

        if encoder_type not in EncoderFactory.SUPPORTED_ENCODERS:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type}. "
                f"Supported encoders: {EncoderFactory.SUPPORTED_ENCODERS}"
            )

        if encoder_type == "transformer":
            return EncoderFactory.create_transformer_encoder(config)

        raise RuntimeError("Encoder creation failed unexpectedly")

    @staticmethod
    def create_from_model_config(
        model_config: MultiTaskModelConfig,
        freeze_encoder: bool = False,
    ) -> TransformerEncoder:
        """
        Build a TransformerEncoder from the encoder section of a
        ``MultiTaskModelConfig``.

        Parameters
        ----------
        model_config:
            Centralised model configuration loaded via ``ModelConfigLoader``.
        freeze_encoder:
            If ``True`` all encoder parameters are frozen after loading.

        Returns
        -------
        TransformerEncoder
        """
        cfg = EncoderConfig(
            model_type="transformer",
            model_name=model_config.encoder.model_name,
            pooling=model_config.encoder.pooling,
            device=model_config.encoder.device,
            freeze_encoder=freeze_encoder,
            output_hidden_states=False,
            gradient_checkpointing=getattr(
                model_config.encoder, "gradient_checkpointing", False
            ),
            extra_kwargs={},
        )
        logger.info(
            "EncoderFactory.create_from_model_config | model=%s pooling=%s freeze=%s",
            cfg.model_name,
            cfg.pooling,
            freeze_encoder,
        )
        return EncoderFactory.create_transformer_encoder(cfg)

    @staticmethod
    def create_from_encoder_config(
        encoder_config: ModelEncoderConfig,
        freeze_encoder: bool = False,
    ) -> TransformerEncoder:
        cfg = EncoderConfig(
            model_type="transformer",
            model_name=encoder_config.model_name,
            pooling=encoder_config.pooling,
            device=encoder_config.device,
            freeze_encoder=freeze_encoder,
            output_hidden_states=False,
            gradient_checkpointing=getattr(
                encoder_config, "gradient_checkpointing", False
            ),
            extra_kwargs={},
        )
        return EncoderFactory.create_transformer_encoder(cfg)

    @staticmethod
    def create_from_yaml(
        yaml_path: str | Path,
        freeze_encoder: bool = False,
    ) -> TransformerEncoder:
        """
        Load a ``MultiTaskModelConfig`` from a YAML file and build a
        ``TransformerEncoder``.

        Parameters
        ----------
        yaml_path:
            Path to the model YAML configuration file.
        freeze_encoder:
            If ``True`` all encoder parameters are frozen after loading.

        Returns
        -------
        TransformerEncoder
        """
        model_config = ModelConfigLoader.load_multitask_config(yaml_path)
        logger.info("EncoderFactory.create_from_yaml | path=%s", yaml_path)
        return EncoderFactory.create_from_model_config(
            model_config, freeze_encoder=freeze_encoder
        )

    @staticmethod
    def detect_device(device: Optional[str] = None) -> torch.device:
        """
        Determine the device used for encoder execution.

        Args:
            device:
                Optional explicit device string.

        Returns:
            torch.device
        """

        if device:
            return torch.device(device)

        detected = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.debug("Detected device for encoder: %s", detected)

        return detected