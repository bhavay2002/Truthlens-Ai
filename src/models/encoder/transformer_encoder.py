"""
File Name: transformer_encoder.py
Module: Model Architecture - Transformer Encoder
Description:
    Provides a reusable transformer encoder wrapper used across models in the
    TruthLens AI system. The module abstracts loading pretrained transformer
    backbones from the HuggingFace ecosystem and exposes a clean interface
    for downstream classifier heads. It supports device management, pooling
    strategies, hidden state access, and modular integration into multi-task
    architectures.

Dependencies:
    logging
    typing
    torch
    torch.nn
    transformers

Inputs:
    Tokenized transformer inputs (input_ids, attention_mask)

Outputs:
    Encoded contextual embeddings and pooled representations
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


logger = logging.getLogger(__name__)


class TransformerEncoder(nn.Module):
    """
    Generic transformer encoder wrapper for NLP models.
    """

    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",
        device: Optional[str] = None,
    ) -> None:
        """Initialize pretrained transformer encoder."""

        super().__init__()

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a valid string")

        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'")

        self.pooling = pooling

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        except Exception as exc:
            logger.exception("Failed to load transformer model")
            raise RuntimeError("Transformer encoder initialization failed") from exc

        self.hidden_size = self.config.hidden_size

        self.to(self.device)

        logger.info("TransformerEncoder initialized with model %s", model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass through transformer encoder."""

        if input_ids is None or attention_mask is None:
            raise ValueError("input tensors cannot be None")

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        try:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
        """Apply pooling strategy to obtain sentence embedding."""

        if self.pooling == "cls":
            return hidden_states[:, 0]

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            masked_embeddings = hidden_states * mask
            summed = torch.sum(masked_embeddings, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        raise RuntimeError("Invalid pooling strategy")