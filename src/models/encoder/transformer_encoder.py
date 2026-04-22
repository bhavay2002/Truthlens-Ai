"""
File Name: transformer_encoder.py
Module: models.encoder
Description:
    Provides a reusable transformer encoder wrapper used across models in the
    TruthLens AI system. This module abstracts loading pretrained transformer
    backbones from the HuggingFace ecosystem and exposes a clean interface
    for downstream classifier heads and multi-task models.

    The encoder supports configurable pooling strategies, device management,
    hidden state extraction, optional freezing for feature extraction, and
    deterministic behavior for reproducible research.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    transformers
Inputs:
    Tokenized transformer inputs (input_ids, attention_mask)
Outputs:
    Dictionary containing contextual token embeddings and pooled sentence
    representations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..base.base_model import BaseModel
from ..config import (
    EncoderConfig as ModelEncoderConfig,
    MultiTaskModelConfig,
)
from .encoder_config import EncoderConfig as FactoryEncoderConfig

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput:
    """
    Structured output returned by TransformerEncoder.
    """

    sequence_output: torch.Tensor
    pooled_output: torch.Tensor


class TransformerEncoder(BaseModel):
    """
    Generic transformer encoder wrapper for NLP models.

    This abstraction isolates the transformer backbone from downstream models
    and provides standardized outputs for classification heads and multitask
    architectures.
    """

    VALID_POOLING = {"cls", "mean"}

    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",
        device: Optional[str] = None,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
        init_from_config_only: bool = False,
    ) -> None:
        """
        Initialize pretrained transformer encoder.

        Args:
            model_name:
                HuggingFace model identifier (e.g., "roberta-base").
            pooling:
                Pooling strategy ("cls" or "mean").
            device:
                Target device ("cpu" or "cuda").
            freeze_encoder:
                If True, encoder parameters are frozen.
            gradient_checkpointing:
                If True, enable gradient checkpointing when supported.
        """

        super().__init__()

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if pooling not in self.VALID_POOLING:
            raise ValueError(f"pooling must be one of {self.VALID_POOLING}")

        self.model_name = model_name
        self.pooling = pooling

        target_device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            self.config = AutoConfig.from_pretrained(model_name)
            # We never consume HF pooler_output in this codebase; using a
            # random pooler (e.g. RoBERTa) adds noisy unused parameters and
            # triggers misleading initialization warnings.
            if hasattr(self.config, "add_pooling_layer"):
                self.config.add_pooling_layer = False
            if init_from_config_only:
                logger.info(
                    "Initializing encoder from config only: %s", model_name
                )
                self.encoder = AutoModel.from_config(self.config)
            else:
                self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        except Exception as exc:
            logger.exception("Failed to initialize transformer model: %s", model_name)
            raise RuntimeError(
                f"Transformer encoder initialization failed for {model_name}"
            ) from exc

        self.hidden_size: int = self.config.hidden_size
        self.gradient_checkpointing_enabled: bool = False
        # Cached frozen-state flag; updated by freeze() / unfreeze().
        # Avoids an O(n_params) iteration on every forward call.
        self._encoder_frozen: bool = False

        if gradient_checkpointing:
            self.gradient_checkpointing_enable()

        if freeze_encoder:
            self.freeze()

        self.set_device(target_device)

        logger.info(
            "TransformerEncoder initialized | model=%s | pooling=%s | hidden=%d",
            model_name,
            pooling,
            self.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward pass through transformer encoder.

        Args:
            input_ids:
                Token IDs tensor (batch_size, seq_len)
            attention_mask:
                Attention mask tensor (batch_size, seq_len)

        Returns:
            Dictionary containing sequence_output and pooled_output.
        """

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask cannot be None")

        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch_size, seq_len)")

        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids shape")

        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device, non_blocking=True)

        if attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device, non_blocking=True)

        try:
            # Use the cached frozen flag instead of iterating all parameters
            # on every forward call (O(n_params) per inference step).
            if self._encoder_frozen:
                with torch.no_grad():
                    outputs = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
            else:
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        except Exception as exc:
            logger.exception("Transformer forward pass failed")
            raise RuntimeError("Encoder forward pass failed") from exc

        sequence_output = outputs.last_hidden_state

        pooled_output = self._pool(sequence_output, attention_mask)

        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on the underlying HF encoder."""
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
            self.gradient_checkpointing_enabled = True
            logger.info("Transformer gradient checkpointing enabled")
        else:
            logger.warning("Gradient checkpointing is not supported by this encoder")

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on the underlying HF encoder."""
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()
            self.gradient_checkpointing_enabled = False
            logger.info("Transformer gradient checkpointing disabled")

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply pooling strategy to obtain sentence embeddings.

        Args:
            hidden_states:
                Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask:
                Tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, hidden_size)
        """

        if self.pooling == "cls":
            return hidden_states[:, 0]

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = torch.sum(hidden_states * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        raise RuntimeError(f"Invalid pooling strategy: {self.pooling}")

    def freeze(self) -> None:
        """
        Freeze encoder parameters (useful for feature extraction).
        """

        for param in self.encoder.parameters():
            param.requires_grad = False

        self._encoder_frozen = True
        logger.info("Transformer encoder parameters frozen")

    def unfreeze(self) -> None:
        """
        Unfreeze encoder parameters.
        """

        for param in self.encoder.parameters():
            param.requires_grad = True

        self._encoder_frozen = False
        logger.info("Transformer encoder parameters unfrozen")

    def get_hidden_size(self) -> int:
        """
        Returns hidden dimension of the encoder.
        """

        return self.hidden_size

    @classmethod
    def from_config(
        cls,
        config: ModelEncoderConfig | FactoryEncoderConfig,
        freeze_encoder: bool = False,
    ) -> "TransformerEncoder":
        """
        Instantiate a ``TransformerEncoder`` from a central
        ``ModelEncoderConfig``.

        Parameters
        ----------
        config:
            Encoder configuration loaded from ``src.models.config.model_config``.
        freeze_encoder:
            If ``True`` all encoder parameters are frozen after loading.

        Returns
        -------
        TransformerEncoder
        """
        logger.info(
            "TransformerEncoder.from_config | model=%s pooling=%s freeze=%s",
            config.model_name,
            config.pooling,
            freeze_encoder,
        )
        return cls(
            model_name=config.model_name,
            pooling=config.pooling,
            device=config.device,
            freeze_encoder=freeze_encoder,
        )

    @classmethod
    def from_model_config(
        cls,
        model_config: MultiTaskModelConfig,
        freeze_encoder: bool = False,
    ) -> "TransformerEncoder":
        return cls.from_config(model_config.encoder, freeze_encoder=freeze_encoder)