from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer for sequence representations.

    Computes a weighted sum of token embeddings where attention weights
    are learned dynamically.

    Inputs:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Optional Tensor of shape (batch_size, seq_len)

    Outputs:
        pooled: Tensor of shape (batch_size, hidden_dim)
        weights: Tensor of shape (batch_size, seq_len)
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        attention_dim = attention_dim or hidden_dim

        self.proj = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # (B, T, H) → (B, T, A)
        x = torch.tanh(self.proj(hidden_states))
        x = self.dropout(x)

        # (B, T, 1) → (B, T)
        scores = self.score(x).squeeze(-1)

        if attention_mask is not None:
            # mask padding tokens
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # (B, T)
        weights = F.softmax(scores, dim=-1)

        # (B, T, H)
        weights_expanded = weights.unsqueeze(-1)

        # weighted sum → (B, H)
        pooled = torch.sum(hidden_states * weights_expanded, dim=1)

        return pooled, weights