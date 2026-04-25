from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..base.base_model import BaseModel

logger = logging.getLogger(__name__)


# =========================================================
# OUTPUT
# =========================================================

@dataclass
class EncoderOutput:
    sequence_output: torch.Tensor
    pooled_output: torch.Tensor


# =========================================================
# ENCODER
# =========================================================

class TransformerEncoder(BaseModel):

    VALID_POOLING = {"cls", "mean", "attention"}

    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",
        device: Optional[str] = None,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
        output_hidden_states: bool = False,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        use_compile: bool = False,
        compile_mode: str = "default",
        max_length: int = 512,
        init_from_config_only: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()

        if pooling not in self.VALID_POOLING:
            raise ValueError(f"Invalid pooling: {pooling}")

        self.model_name = model_name
        self.pooling = pooling
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.max_length = max_length
        self.output_hidden_states = output_hidden_states

        self._encoder_frozen = False

        device_obj = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        try:
            self.config = AutoConfig.from_pretrained(model_name)

            if hasattr(self.config, "add_pooling_layer"):
                self.config.add_pooling_layer = False

            if init_from_config_only:
                self.encoder = AutoModel.from_config(self.config)
            else:
                self.encoder = AutoModel.from_pretrained(
                    model_name,
                    config=self.config,
                )

        except Exception as e:
            logger.exception("Encoder init failed")
            raise RuntimeError from e

        self.hidden_size = self.config.hidden_size

        if gradient_checkpointing:
            self.gradient_checkpointing_enable()

        if freeze_encoder:
            self.freeze()

        if use_compile and hasattr(torch, "compile"):
            try:
                self.encoder = torch.compile(
                    self.encoder,
                    mode=compile_mode,
                )
                logger.info("Encoder compiled")
            except Exception:
                logger.warning("Compile failed")

        self.set_device(device_obj)

        logger.info(
            "Encoder ready | model=%s | hidden=%d",
            model_name,
            self.hidden_size,
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)

        if attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)

        autocast_dtype = (
            torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        )

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.use_amp and self.device.type == "cuda",
            dtype=autocast_dtype,
        ):
            if self._encoder_frozen:
                with torch.no_grad():
                    outputs = self.encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                        output_hidden_states=self.output_hidden_states,
                    )
            else:
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=self.output_hidden_states,
                )

        sequence_output = outputs.last_hidden_state
        pooled_output = self._pool(sequence_output, attention_mask)

        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
        }

    # =====================================================
    # POOLING
    # =====================================================

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        if self.pooling == "cls":
            return hidden_states[:, 0]

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = torch.sum(hidden_states * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        if self.pooling == "attention":
            weights = torch.softmax(hidden_states.mean(dim=-1), dim=1)
            return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

        raise RuntimeError("Invalid pooling")

    # =====================================================
    # GRAD CKPT
    # =====================================================

    def gradient_checkpointing_enable(self):

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):

        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()

    # =====================================================
    # FREEZE
    # =====================================================

    def freeze(self):

        for p in self.encoder.parameters():
            p.requires_grad = False

        self._encoder_frozen = True

    def unfreeze(self):

        for p in self.encoder.parameters():
            p.requires_grad = True

        self._encoder_frozen = False

    # =====================================================
    # UTILS
    # =====================================================

    def get_hidden_size(self) -> int:
        return self.hidden_size