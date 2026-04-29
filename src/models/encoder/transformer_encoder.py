"""Canonical ``TransformerEncoder`` implementation (audit 3.2).

This module used to be a re-export shim that pointed at
``src.models.inference.model_wrapper``. The audit flipped the convention:
the encoder package is the natural home for an encoder, so the real
class lives here and ``model_wrapper.py`` is the back-compat shim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.models.base.base_model import BaseModel

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
        # P2.1: tri-state — None means "auto: on for CUDA, off for CPU".
        # See ``EncoderConfig.use_compile`` for the rationale.
        use_compile: Optional[bool] = None,
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

        # A5.1: route through the centralised detector so MPS is honoured
        # and the resolution rules cannot drift from the EncoderFactory /
        # benchmark / model_utils path.
        from src.models._device import detect_device

        device_obj = detect_device(device)

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

        # P2.1: resolve tri-state ``use_compile`` against the resolved
        # device. ``None`` (auto) maps to True on CUDA and False on CPU.
        if use_compile is None:
            resolved_use_compile = device_obj.type == "cuda"
        else:
            resolved_use_compile = bool(use_compile)

        if resolved_use_compile and hasattr(torch, "compile"):
            try:
                self.encoder = torch.compile(
                    self.encoder,
                    mode=compile_mode,
                )
                logger.info("Encoder compiled (mode=%s)", compile_mode)
            except Exception:
                logger.warning("Compile failed", exc_info=True)

        self.set_device(device_obj)

        # P2.7: cache the resolved device on a plain attribute so
        # ``forward`` can compare against it without walking
        # ``next(self.parameters()).device`` on every call (which is a
        # measurable per-batch overhead on small inputs).
        self._cached_device = device_obj

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

        # P2.7 + A3.3: ``self._cached_device`` is a plain attribute set
        # in ``__init__`` / ``set_device``; the ``device`` property's
        # fast path also returns ``self._device`` directly, but holding
        # a local reference avoids the property dispatch on the per-batch
        # hot path.
        device = getattr(self, "_cached_device", None)
        if device is None:
            device = self.device
            self._cached_device = device

        if input_ids.device != device:
            input_ids = input_ids.to(device)

        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        autocast_dtype = (
            torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        )

        with torch.autocast(
            device_type=device.type,
            enabled=self.use_amp and device.type == "cuda",
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
    # DEVICE  (P2.7: keep cached device in sync)
    # =====================================================

    def set_device(self, device):
        super().set_device(device)
        # ``BaseModel.set_device`` normalises strings into ``torch.device``
        # via ``self._device`` — surface that resolved value into the
        # ``_cached_device`` slot so ``forward`` keeps using the fast path.
        self._cached_device = self._device

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


__all__ = ["TransformerEncoder", "EncoderOutput"]
