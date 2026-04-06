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
from ..config.model_config import EncoderConfig as ModelEncoderConfig

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
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        except Exception as exc:
            logger.exception("Failed to initialize transformer model: %s", model_name)
            raise RuntimeError(
                f"Transformer encoder initialization failed for {model_name}"
            ) from exc

        self.hidden_size: int = self.config.hidden_size

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

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        try:
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
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            masked_embeddings = hidden_states * mask
            summed = torch.sum(masked_embeddings, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        raise RuntimeError(f"Invalid pooling strategy: {self.pooling}")

    def freeze(self) -> None:
        """
        Freeze encoder parameters (useful for feature extraction).
        """

        for param in self.encoder.parameters():
            param.requires_grad = False

        logger.info("Transformer encoder parameters frozen")

    def unfreeze(self) -> None:
        """
        Unfreeze encoder parameters.
        """

        for param in self.encoder.parameters():
            param.requires_grad = True

        logger.info("Transformer encoder parameters unfrozen")

    def get_hidden_size(self) -> int:
        """
        Returns hidden dimension of the encoder.
        """

        return self.hidden_size

    @classmethod
    def from_config(
        cls,
        config: ModelEncoderConfig,
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