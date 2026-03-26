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

    def extract_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights from the transformer model."""

        if input_ids is None or attention_mask is None:
            raise ValueError("input tensors cannot be None")

        try:
            outputs = self.model.encoder.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        except Exception as exc:
            logger.exception("Failed to extract attention weights")
            raise RuntimeError("Attention extraction failed") from exc

        attentions = outputs.attentions

        return {
            "attentions": attentions
        }

    def aggregate_attention(
        self,
        attentions: List[torch.Tensor],
    ) -> np.ndarray:
        """Aggregate attention across heads and layers."""

        if not attentions:
            raise ValueError("attentions list cannot be empty")

        try:
            stacked = torch.stack(attentions)

            avg_attention = stacked.mean(dim=0).mean(dim=1)

            return avg_attention.detach().cpu().numpy()
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

        try:
            plt.figure(figsize=(10, 8))

            plt.imshow(attention_matrix, cmap="viridis")

            plt.colorbar()

            plt.xticks(range(len(tokens)), tokens, rotation=90)

            plt.yticks(range(len(tokens)), tokens)

            plt.title(title)

            plt.tight_layout()

            plt.show()

        except Exception as exc:
            logger.exception("Attention visualization failed")
            raise RuntimeError("Failed to visualize attention") from exc