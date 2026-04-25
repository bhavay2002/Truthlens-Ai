from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from src.explainability.explanation_calibrator import calibrate_explanation
from src.explainability.common_schema import ExplanationOutput, TokenImportance

logger = logging.getLogger(__name__)

EPS = 1e-12


class AttentionRollout:

    def __init__(self) -> None:
        logger.info("AttentionRollout initialized")

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def _validate_inputs(
        attentions: List[torch.Tensor],
        tokens: List[str],
        sample_index: int,
        source_token_index: int,
    ) -> int:

        if not attentions:
            raise ValueError("attentions list cannot be empty")

        if not isinstance(tokens, list) or not tokens:
            raise ValueError("tokens must be non-empty list")

        if sample_index < 0:
            raise ValueError("sample_index must be >= 0")

        seq_len = None
        batch_size = None

        for tensor in attentions:

            if tensor.ndim != 4:
                raise ValueError("attention must be (batch, heads, seq, seq)")

            b, _, s1, s2 = tensor.shape

            if s1 != s2:
                raise ValueError("attention matrices must be square")

            if batch_size is None:
                batch_size = b
            elif b != batch_size:
                raise ValueError("inconsistent batch size")

            if seq_len is None:
                seq_len = s1
            elif s1 != seq_len:
                raise ValueError("inconsistent seq_len")

        if sample_index >= batch_size:
            raise ValueError("sample_index out of range")

        if len(tokens) > seq_len:
            raise ValueError("tokens exceed seq_len")

        if not (0 <= source_token_index < seq_len):
            raise ValueError("invalid source_token_index")

        return int(seq_len)

    # =====================================================
    # CORE OPS
    # =====================================================

    @staticmethod
    def _aggregate_heads(attention: torch.Tensor, sample_index: int) -> torch.Tensor:
        return attention.mean(dim=1)[sample_index]

    @staticmethod
    def _add_residual(attention: torch.Tensor) -> torch.Tensor:
        seq_len = attention.shape[0]
        identity = torch.eye(seq_len, device=attention.device, dtype=attention.dtype)
        attention = attention + identity
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return attention

    # =====================================================
    # MAIN (FINAL)
    # =====================================================

    def compute_rollout(
        self,
        attentions: List[torch.Tensor],
        tokens: List[str],
        *,
        sample_index: int = 0,
        source_token_index: int = 0,
        mask_tokens: Optional[List[str]] = None,
        layer_weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
    ) -> ExplanationOutput:

        self._validate_inputs(attentions, tokens, sample_index, source_token_index)

        try:
            with torch.no_grad():

                processed: List[torch.Tensor] = []

                for i, layer_attention in enumerate(attentions):

                    attn = self._aggregate_heads(layer_attention, sample_index)

                    if attn.dtype in (torch.float16, torch.bfloat16):
                        attn = attn.to(torch.float32)

                    attn = self._add_residual(attn)

                    if layer_weights and i < len(layer_weights):
                        attn = attn * float(layer_weights[i])

                    processed.append(attn)

                rollout = torch.linalg.multi_dot(processed[::-1])

                scores = rollout[source_token_index]
                scores = scores.detach().cpu().numpy().astype(np.float32)

                scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                scores = np.maximum(scores, 0.0)

                # mask tokens
                if mask_tokens:
                    for i, t in enumerate(tokens):
                        if t in mask_tokens:
                            scores[i] = 0.0

                # =====================================================
                # 🔥 CALIBRATION
                # =====================================================
                cal = calibrate_explanation(scores.tolist(), method="attention")

                scores = cal["scores"]
                confidence = cal["confidence"]
                entropy = cal["entropy"]

                # =====================================================
                # TOP-K
                # =====================================================
                if top_k is not None and top_k > 0:
                    idx = np.argsort(scores)[-top_k:][::-1]
                    tokens = [tokens[i] for i in idx]
                    scores = scores[idx]

                structured = [
                    TokenImportance(token=t, importance=float(s))
                    for t, s in zip(tokens, scores)
                ]

                return ExplanationOutput(
                    method="attention",
                    tokens=tokens,
                    importance=scores.tolist(),
                    structured=structured,
                    confidence=confidence,
                    entropy=entropy,
                )

        except Exception as exc:
            logger.exception("Attention rollout computation failed")
            raise RuntimeError("Attention rollout failed") from exc