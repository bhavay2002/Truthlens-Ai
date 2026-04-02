"""
File Name: advanced_analysis.py
Module: TruthLens AI - Advanced Evaluation
Description:
    Research-grade evaluation diagnostics used for deeper model analysis
    in the TruthLens AI system. Provides graph-based narrative metrics,
    frame prediction coherence, and cross-task attention correlation
    analysis for multi-task transformer models.
Dependencies:
    logging
    typing
    numpy
    networkx
    pandas
Inputs:
    df: DataFrame containing narrative entity columns
    pred_frames: predicted frame labels
    true_frames: ground truth frame labels
    attention_maps: dictionary of task attention tensors
Outputs:
    Dictionary containing diagnostic evaluation metrics
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def _safe_array(x: Any) -> np.ndarray:
    """Safely convert input to numpy array."""
    try:
        return np.asarray(x)
    except Exception as exc:
        raise ValueError("Failed to convert input to numpy array") from exc


def actor_graph_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Build an actor interaction graph from hero and villain entities
    and compute structural network statistics.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    required_cols = {"hero_entities", "villain_entities"}

    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"DataFrame must contain columns: {required_cols}"
        )

    logger.info("Constructing actor interaction graph")

    graph = nx.DiGraph()

    for _, row in df.iterrows():

        hero = row.get("hero_entities")
        villain = row.get("villain_entities")

        if hero and villain:
            graph.add_edge(str(hero), str(villain))

    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()

    density = nx.density(graph) if nodes > 1 else 0.0

    try:
        avg_degree = float(np.mean([d for _, d in graph.degree()])) if nodes > 0 else 0.0
    except Exception:
        avg_degree = 0.0

    results = {
        "nodes": float(nodes),
        "edges": float(edges),
        "density": float(density),
        "avg_degree": float(avg_degree),
    }

    logger.info("Actor graph metrics computed")

    return results


def frame_coherence(
    pred_frames,
    true_frames
) -> float:
    """
    Measure agreement between predicted and true frame labels.
    """

    pred = _safe_array(pred_frames)
    true = _safe_array(true_frames)

    if pred.shape != true.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs true {true.shape}"
        )

    score = float(np.mean(pred == true))

    logger.info("Frame coherence score: %.4f", score)

    return score


def cross_task_attention(
    attention_maps: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute correlations between attention maps of different tasks
    to analyze shared attention patterns in multitask models.
    """

    if not isinstance(attention_maps, dict):
        raise TypeError("attention_maps must be a dictionary")

    tasks = list(attention_maps.keys())

    correlations: Dict[str, float] = {}

    logger.info("Computing cross-task attention correlations")

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):

            task_a = tasks[i]
            task_b = tasks[j]

            a = _safe_array(attention_maps[task_a]).flatten()
            b = _safe_array(attention_maps[task_b]).flatten()

            if a.size != b.size:
                logger.warning(
                    "Attention map size mismatch: %s vs %s",
                    task_a,
                    task_b,
                )
                continue

            try:
                corr = float(np.corrcoef(a, b)[0, 1])
            except Exception:
                corr = 0.0

            correlations[f"{task_a}_{task_b}"] = corr

    logger.info("Cross-task attention analysis complete")

    return correlations