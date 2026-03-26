"""
File Name: attention_visualizer.py
Module: Model Analysis - Attention Visualization
Description:
    Provides utilities for extracting and visualizing transformer attention
    weights used in the TruthLens AI system. The module supports attention
    extraction from HuggingFace transformer models and generates interpretable
    visualizations for analysis, debugging, and research purposes.

Dependencies:
    logging
    typing
    torch
    numpy
    matplotlib
    transformers

Inputs:
    Tokenized model inputs and trained transformer model

Outputs:
    Attention matrices and attention visualization plots
"""

import logging
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Extracts and visualizes attention maps from transformer models.
    """

    def __init__(self, model) -> None:
        """Initialize visualizer with transformer model."""

        if model is None:
            raise ValueError("model cannot be None")

        self.model = model

        logger.info("AttentionVisualizer initialized")

    def _resolve_model_device(self) -> torch.device | None:
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return None

    def extract_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights from the transformer model."""

        if input_ids is None or attention_mask is None:
            raise ValueError("input tensors cannot be None")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            raise TypeError("input_ids and attention_mask must be torch tensors")
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be 2D tensors")
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                "input_ids and attention_mask must have the same shape"
            )

        model_device = self._resolve_model_device()
        if model_device is not None:
            input_ids = input_ids.to(model_device)
            attention_mask = attention_mask.to(model_device)

        try:
            with torch.no_grad():
                try:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True,
                        return_dict=True,
                    )
                except TypeError:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True,
                    )
        except Exception as exc:
            logger.exception("Failed to extract attention weights")
            raise RuntimeError("Attention extraction failed") from exc

        attentions = getattr(outputs, "attentions", None)
        if attentions is None and isinstance(outputs, (tuple, list)) and outputs:
            attentions = outputs[-1]
        if attentions is None:
            raise RuntimeError(
                "Model output does not include attentions. "
                "Ensure the model supports output_attentions=True."
            )

        return {"attentions": attentions}

    def aggregate_attention(
        self,
        attentions: List[torch.Tensor],
    ) -> np.ndarray:
        """Aggregate attention across heads and layers."""

        if not attentions:
            raise ValueError("attentions list cannot be empty")

        try:
            tensor_attentions: list[torch.Tensor] = []
            for tensor in attentions:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("attentions must contain torch.Tensor values")
                if tensor.ndim != 4:
                    raise ValueError(
                        "Each attention tensor must have shape "
                        "(batch, heads, seq_len, seq_len)."
                    )
                tensor_attentions.append(tensor.detach())

            stacked = torch.stack(tensor_attentions, dim=0)

            # Average across layers and heads, then collapse batch.
            avg_attention = stacked.mean(dim=0).mean(dim=1)
            if avg_attention.shape[0] == 1:
                avg_attention = avg_attention[0]
            else:
                avg_attention = avg_attention.mean(dim=0)

            return avg_attention.cpu().numpy()
        except Exception as exc:
            logger.exception("Attention aggregation failed")
            raise RuntimeError("Failed to aggregate attention") from exc

    def plot_attention(
        self,
        attention_matrix: np.ndarray,
        tokens: List[str],
        title: str = "Attention Map",
    ) -> None:
        """Visualize attention matrix using heatmap."""

        if attention_matrix is None or tokens is None:
            raise ValueError("attention_matrix and tokens must be provided")
        if not isinstance(attention_matrix, np.ndarray):
            attention_matrix = np.asarray(attention_matrix)
        if attention_matrix.ndim != 2:
            raise ValueError("attention_matrix must be 2D for plotting")
        if not tokens:
            raise ValueError("tokens cannot be empty")

        try:
            plot_size = min(
                attention_matrix.shape[0],
                attention_matrix.shape[1],
                len(tokens),
            )
            if plot_size == 0:
                raise ValueError("attention_matrix and tokens must be non-empty")

            matrix = attention_matrix[:plot_size, :plot_size]
            token_labels = tokens[:plot_size]

            fig, ax = plt.subplots(figsize=(10, 8))

            image = ax.imshow(matrix, cmap="viridis")

            fig.colorbar(image, ax=ax)

            ax.set_xticks(range(plot_size))
            ax.set_xticklabels(token_labels, rotation=90)

            ax.set_yticks(range(plot_size))
            ax.set_yticklabels(token_labels)

            ax.set_title(title)

            fig.tight_layout()

            plt.show()
            plt.close(fig)

        except Exception as exc:
            logger.exception("Attention visualization failed")
            raise RuntimeError("Failed to visualize attention") from exc
