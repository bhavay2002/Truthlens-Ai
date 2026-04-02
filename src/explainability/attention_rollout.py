"""
File Name: attention_rollout.py
Module: Explainability - Attention Rollout
Description:
    Implements Attention Rollout for transformer models. Attention rollout
    computes cumulative attention flow across transformer layers to produce
    more reliable token importance scores than raw attention weights.

    The algorithm aggregates attention matrices across heads and layers,
    propagating attention influence through the network to estimate how
    much each input token contributes to the final representation.

    This method is widely used in transformer interpretability research.

Dependencies:
    logging
    typing
    numpy
    torch

Inputs:
    attentions : List[torch.Tensor] with shape
        (batch, heads, seq_len, seq_len)
    tokens : List[str]

Outputs:
    Dictionary containing tokens and rollout importance scores
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AttentionRollout:
    """
    Compute attention rollout scores for transformer models.
    """

    def __init__(self) -> None:
        logger.info("AttentionRollout initialized")

    @staticmethod
    def _validate_inputs(
        attentions: List[torch.Tensor],
        tokens: List[str],
    ) -> None:
        if not attentions:
            raise ValueError("attentions list cannot be empty")

        if not isinstance(tokens, list) or not tokens:
            raise ValueError("tokens must be a non-empty list")

        for tensor in attentions:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("attentions must contain torch.Tensor objects")

            if tensor.ndim != 4:
                raise ValueError(
                    "Each attention tensor must have shape "
                    "(batch, heads, seq_len, seq_len)"
                )

    @staticmethod
    def _aggregate_heads(attention: torch.Tensor) -> torch.Tensor:
        """
        Average attention across heads.

        Shape:
            (batch, heads, seq, seq) → (seq, seq)
        """
        attention = attention.mean(dim=1)
        return attention[0]

    @staticmethod
    def _add_residual_connection(attention: torch.Tensor) -> torch.Tensor:
        """
        Add residual identity connection and normalize.
        """
        seq_len = attention.shape[0]

        identity = torch.eye(seq_len, device=attention.device)

        attention = attention + identity

        attention = attention / attention.sum(dim=-1, keepdim=True)

        return attention

    def compute_rollout(
        self,
        attentions: List[torch.Tensor],
        tokens: List[str],
    ) -> Dict[str, List[float]]:
        """
        Compute attention rollout importance scores.

        Parameters
        ----------
        attentions : list of attention tensors
        tokens : list of tokens corresponding to sequence

        Returns
        -------
        Dict containing tokens and rollout scores.
        """

        self._validate_inputs(attentions, tokens)

        try:
            processed: List[torch.Tensor] = []

            for layer_attention in attentions:
                attn = self._aggregate_heads(layer_attention)

                attn = self._add_residual_connection(attn)

                processed.append(attn)

            rollout = processed[0]

            for layer in processed[1:]:
                rollout = layer @ rollout

            rollout_scores = rollout[0].detach().cpu().numpy()

            rollout_scores = np.maximum(rollout_scores, 0)

            total = float(np.sum(rollout_scores))

            if total > 0:
                rollout_scores = rollout_scores / total

            scores = rollout_scores.tolist()

            return {
                "tokens": tokens,
                "rollout_scores": scores[: len(tokens)],
            }

        except Exception as exc:
            logger.exception("Attention rollout computation failed")
            raise RuntimeError("Attention rollout failed") from exc