from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from src.explainability.token_alignment import align_tokens
from src.explainability.utils_validation import validate_tokens_scores

logger = logging.getLogger(__name__)


class PropagandaExplainer:
    def __init__(self, model: torch.nn.Module) -> None:
        if model is None:
            raise ValueError("model cannot be None")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be torch.nn.Module")
        self.model = model
        logger.info("PropagandaExplainer initialized")

    def _resolve_device(self) -> Optional[torch.device]:
        try:
            return next(self.model.parameters()).device
        except Exception:
            return None

    def explain(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, tokens: List[str]) -> Dict[str, float]:
        if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
            raise TypeError("input_ids and attention_mask must be torch.Tensor")
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be 2D")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have same shape")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError("tokens must be a non-empty list")
        if len(tokens) > input_ids.shape[1]:
            raise ValueError("tokens length exceeds sequence length")

        gradients = self._gradient_importance(input_ids, attention_mask)
        normalized = self._normalize_scores(gradients, tokens)
        validate_tokens_scores(tokens, normalized)
        merged_tokens, merged_scores = align_tokens(tokens, normalized)

        explanation: Dict[str, float] = {}
        token_count: Dict[str, int] = {}
        for token, score in zip(merged_tokens, merged_scores):
            idx = token_count.get(token, 0)
            key = token if idx == 0 else f"{token}_{idx}"
            explanation[key] = float(score)
            token_count[token] = idx + 1
        return explanation

    def _gradient_importance(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        device = self._resolve_device()
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        self.model.zero_grad(set_to_none=True)
        emb = self.model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        outputs = self.model(inputs_embeds=emb, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        logits.max().backward()
        grads = emb.grad
        return torch.abs(grads).sum(dim=-1).detach().cpu().numpy()[0]

    def _normalize_scores(self, scores: np.ndarray, tokens: List[str]) -> np.ndarray:
        vals = np.asarray(scores[: len(tokens)], dtype=float)
        vals = np.abs(vals)
        total = float(np.sum(vals))
        if total <= 0:
            return np.zeros(len(tokens), dtype=float)
        return vals / total

    def propaganda_intensity(self, explanation: Dict[str, float]) -> float:
        if not explanation:
            return 0.0
        return float(np.mean(list(explanation.values())))
