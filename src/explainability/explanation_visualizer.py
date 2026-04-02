"""
File Name: explanation_visualizer.py
Module: Explainability - Visualization
Description:
    Centralized visualization utilities for explainability outputs in the
    TruthLens AI system. This module consolidates plotting functions used
    across explanation methods and provides standardized visualizations for:

        • Token importance heatmaps
        • Token importance bar charts
        • Attention maps
        • Explanation comparison plots

    The module is designed to support research analysis, debugging,
    dashboards, and reporting pipelines.

Author: TruthLens Engineering Team
Date: 2026-04-02

Dependencies:
    logging
    typing
    numpy
    matplotlib

Inputs:
    tokens
    explanation scores

Outputs:
    Matplotlib visualizations
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ExplanationVisualizer:
    """
    Visualization utilities for explanation outputs.
    """

    def __init__(self) -> None:
        logger.info("ExplanationVisualizer initialized")

    @staticmethod
    def _validate_tokens_scores(tokens: List[str], scores: List[float]) -> None:
        if not tokens or not scores:
            raise ValueError("tokens and scores must not be empty")

        if len(tokens) != len(scores):
            raise ValueError("tokens and scores must have the same length")

    def plot_token_heatmap(
        self,
        tokens: List[str],
        scores: List[float],
        title: str = "Token Importance Heatmap",
    ) -> None:
        """
        Plot token importance heatmap.
        """

        self._validate_tokens_scores(tokens, scores)

        matrix = np.array(scores).reshape(1, -1)

        fig, ax = plt.subplots(figsize=(max(len(tokens) * 0.5, 8), 2))

        heatmap = ax.imshow(matrix, cmap="viridis", aspect="auto")

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)

        ax.set_yticks([])

        ax.set_title(title)

        fig.colorbar(heatmap, ax=ax)

        plt.tight_layout()
        plt.show()

    def plot_importance_bar(
        self,
        tokens: List[str],
        scores: List[float],
        top_k: Optional[int] = 20,
        title: str = "Token Importance",
    ) -> None:
        """
        Plot token importance as a bar chart.
        """

        self._validate_tokens_scores(tokens, scores)

        tokens_arr = np.array(tokens)
        scores_arr = np.array(scores)

        order = np.argsort(np.abs(scores_arr))[::-1]

        if top_k:
            order = order[:top_k]

        tokens_sorted = tokens_arr[order]
        scores_sorted = scores_arr[order]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(tokens_sorted[::-1], scores_sorted[::-1])

        ax.set_xlabel("Importance Score")
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    def plot_attention_map(
        self,
        attention_matrix: np.ndarray,
        tokens: List[str],
        title: str = "Attention Map",
    ) -> None:
        """
        Plot transformer attention matrix.
        """

        if attention_matrix.ndim != 2:
            raise ValueError("attention_matrix must be 2D")

        if len(tokens) != attention_matrix.shape[0]:
            raise ValueError("tokens length must match attention matrix size")

        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(attention_matrix, cmap="viridis")

        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)

        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)

        ax.set_title(title)

        fig.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

    def plot_explanation_comparison(
        self,
        tokens: List[str],
        explanations: Dict[str, List[float]],
        top_k: Optional[int] = 15,
        title: str = "Explanation Comparison",
    ) -> None:
        """
        Compare explanation methods side-by-side.

        explanations format example:
        {
            "shap": [...],
            "integrated_gradients": [...],
            "attention": [...]
        }
        """

        if not explanations:
            raise ValueError("explanations dictionary cannot be empty")

        for name, scores in explanations.items():
            if len(scores) != len(tokens):
                raise ValueError(
                    f"Explanation '{name}' scores must match tokens length"
                )

        scores_matrix = np.vstack(list(explanations.values()))

        avg_scores = np.mean(np.abs(scores_matrix), axis=0)

        order = np.argsort(avg_scores)[::-1]

        if top_k:
            order = order[:top_k]

        tokens_top = np.array(tokens)[order]

        fig, ax = plt.subplots(figsize=(12, 6))

        for name, scores in explanations.items():
            scores_arr = np.array(scores)[order]
            ax.plot(tokens_top, scores_arr, marker="o", label=name)

        ax.set_title(title)
        ax.set_ylabel("Importance Score")

        ax.legend()

        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()