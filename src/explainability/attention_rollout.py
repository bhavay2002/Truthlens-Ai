from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AttentionRollout:
    def __init__(self) -> None:
        logger.info("AttentionRollout initialized")

    @staticmethod
    def _validate_inputs(
        attentions: List[torch.Tensor],
        tokens: List[str],
        sample_index: int,
    ) -> None:
        if not attentions:
            raise ValueError("attentions list cannot be empty")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("tokens must be a non-empty list")
        if sample_index < 0:
            raise ValueError("sample_index must be >= 0")

        seq_len = None
        batch_size = None
        for tensor in attentions:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("attentions must contain torch.Tensor objects")
            if tensor.ndim != 4:
                raise ValueError("Each attention tensor must be (batch, heads, seq, seq)")
            b, _, s1, s2 = tensor.shape
            if s1 != s2:
                raise ValueError("attention matrices must be square")
            if batch_size is None:
                batch_size = b
            elif b != batch_size:
                raise ValueError("all attention tensors must have same batch size")
            if seq_len is None:
                seq_len = s1
            elif s1 != seq_len:
                raise ValueError("all attention tensors must have same seq_len")

        if batch_size is None or sample_index >= batch_size:
            raise ValueError("sample_index out of range")
        if len(tokens) > int(seq_len):
            raise ValueError("tokens length exceeds seq_len")

    @staticmethod
    def _aggregate_heads(attention: torch.Tensor, sample_index: int) -> torch.Tensor:
        return attention.mean(dim=1)[sample_index]

    @staticmethod
    def _add_residual_connection(attention: torch.Tensor) -> torch.Tensor:
        seq_len = attention.shape[0]
        identity = torch.eye(seq_len, device=attention.device, dtype=attention.dtype)
        attention = attention + identity
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return attention

    def compute_rollout(
        self,
        attentions: List[torch.Tensor],
        tokens: List[str],
        sample_index: int = 0,
    ) -> Dict[str, List[float]]:
        self._validate_inputs(attentions, tokens, sample_index)

        try:
            processed: List[torch.Tensor] = []
            for layer_attention in attentions:
                attn = self._aggregate_heads(layer_attention, sample_index)
                attn = self._add_residual_connection(attn)
                processed.append(attn)

            rollout = processed[0]
            for layer in processed[1:]:
                rollout = layer @ rollout

            rollout_scores = rollout[0].detach().cpu().numpy()
            rollout_scores = np.maximum(rollout_scores, 0.0)
            total = float(np.sum(rollout_scores))
            if total > 0:
                rollout_scores = rollout_scores / total

            return {"tokens": tokens, "rollout_scores": rollout_scores.tolist()[: len(tokens)]}

        except Exception as exc:
            logger.exception("Attention rollout computation failed")
            raise RuntimeError("Attention rollout failed") from exc
