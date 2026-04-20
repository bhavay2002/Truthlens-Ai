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
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Extracts and visualizes attention maps from transformer models.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """
        Initialize visualizer with transformer model.

        Parameters
        ----------
        model : torch.nn.Module
            HuggingFace-compatible transformer model.
        """

        if model is None:
            raise ValueError("model cannot be None")

        self.model: torch.nn.Module = model

        logger.info("AttentionVisualizer initialized")

    def _resolve_model_device(self) -> Optional[torch.device]:
        """
        Resolve the device where the model parameters reside.

        Returns
        -------
        Optional[torch.device]
            Device of the model parameters or None if not available.
        """

        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return None

    def extract_attention(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Extract attention weights from the transformer model.

        Parameters
        ----------
        input_ids : torch.Tensor
            Tokenized input ids.
        attention_mask : torch.Tensor
            Attention mask tensor.

        Returns
        -------
        Dict[str, List[torch.Tensor]]
            Dictionary containing attention tensors.
        """

        if input_ids is None or attention_mask is None:
            raise ValueError("input tensors cannot be None")

        if not isinstance(input_ids, torch.Tensor) or not isinstance(
            attention_mask, torch.Tensor
        ):
            raise TypeError("input_ids and attention_mask must be torch tensors")

        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("input_ids and attention_mask must be 2D tensors")

        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask must have the same shape")

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

        if not isinstance(attentions, (list, tuple)):
            raise TypeError("Attentions returned by model must be a list or tuple")

        return {"attentions": list(attentions)}

    def aggregate_attention(
        self,
        attentions: List[torch.Tensor],
        sample_index: int = 0,
    ) -> np.ndarray:
        """
        Aggregate attention across layers and heads.

        Parameters
        ----------
        attentions : List[torch.Tensor]
            Attention tensors from transformer layers.

        Returns
        -------
        np.ndarray
            Aggregated attention matrix.
        """

        if not attentions:
            raise ValueError("attentions list cannot be empty")

        try:
            validated_tensors: List[torch.Tensor] = []

            for tensor in attentions:
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError("attentions must contain torch.Tensor values")

                if tensor.ndim != 4:
                    raise ValueError(
                        "Each attention tensor must have shape "
                        "(batch, heads, seq_len, seq_len)"
                    )
                b, _, s1, s2 = tensor.shape
                if s1 != s2:
                    raise ValueError("attention matrices must be square")
                if "batch_size" not in locals():
                    batch_size = b
                elif b != batch_size:
                    raise ValueError("all attention tensors must have same batch size")

                validated_tensors.append(tensor.detach())

            if not (0 <= sample_index < batch_size):
                raise ValueError("sample_index out of range")

            stacked = torch.stack(validated_tensors, dim=0)  # (layers,batch,heads,seq,seq)
            avg_attention = stacked[:, sample_index].mean(dim=0).mean(dim=0)  # (seq,seq)

            if avg_attention.ndim != 2:
                raise ValueError("aggregated attention must be 2D")
            if avg_attention.shape[0] != avg_attention.shape[1]:
                raise ValueError("aggregated attention must be square")

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
        """
        Visualize attention matrix using heatmap.

        Parameters
        ----------
        attention_matrix : np.ndarray
            Aggregated attention matrix.
        tokens : List[str]
            Tokens corresponding to the attention matrix.
        title : str
            Plot title.
        """

        if attention_matrix is None:
            raise ValueError("attention_matrix must be provided")

        if tokens is None:
            raise ValueError("tokens must be provided")

        if not isinstance(attention_matrix, np.ndarray):
            attention_matrix = np.asarray(attention_matrix)

        if attention_matrix.ndim != 2:
            raise ValueError("attention_matrix must be a 2D matrix")

        if attention_matrix.shape[0] != attention_matrix.shape[1]:
            raise ValueError("attention_matrix must be square")

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