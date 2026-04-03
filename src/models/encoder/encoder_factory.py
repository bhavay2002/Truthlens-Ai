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
from dataclasses import dataclass
from typing import Optional

import torch

from .transformer_encoder import TransformerEncoder


logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """
    Configuration object describing a transformer encoder.
    """

    model_name: str
    pooling: str = "cls"
    device: Optional[str] = None
    freeze_encoder: bool = False


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