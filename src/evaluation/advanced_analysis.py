"""
File Name: advanced_analysis.py
Module: TruthLens AI - Advanced Evaluation
Description:
    Research-grade evaluation diagnostics used for deeper model analysis
    in the TruthLens AI system. Provides graph-based narrative metrics,
    frame prediction coherence, and cross-task attention correlation
    analysis for multi-task transformer models.

    Integrates AttentionRollout for richer cross-task attention analysis:
    when raw per-layer attention tensors are provided, rollout scores are
    computed and used in place of (or alongside) flattened attention maps,
    yielding more reliable cumulative attention-flow correlations.

Dependencies:
    logging
    typing
    numpy
    networkx
    pandas
    src.explainability.attention_rollout

Inputs:
    df: DataFrame containing narrative entity columns
    pred_frames: predicted frame labels
    true_frames: ground truth frame labels
    attention_maps: dictionary of task attention tensors or rollout inputs
Outputs:
    Dictionary containing diagnostic evaluation metrics
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import networkx as nx
import pandas as pd
import torch

from src.explainability.attention_rollout import AttentionRollout

logger = logging.getLogger(__name__)

_rollout = AttentionRollout()


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
    attention_maps: Dict[str, Any],
    tokens: Optional[Dict[str, List[str]]] = None,
    use_rollout: bool = True,
) -> Dict[str, float]:
    """
    Compute correlations between attention maps of different tasks to
    analyse shared attention patterns in multitask transformer models.

    When ``use_rollout`` is True and a task's value is a list of
    per-layer attention tensors (shape batch × heads × seq × seq),
    AttentionRollout is applied first to obtain cumulative token
    importance scores. This gives more reliable cross-task correlations
    than raw attention weights because rollout propagates attention
    influence across all transformer layers.

    When a task value is already a flat numpy array (or cannot be
    processed by rollout), the function falls back to the original
    flatten-and-correlate approach.

    Parameters
    ----------
    attention_maps : dict
        Keys are task names. Values are either:
        - list[torch.Tensor] -- per-layer attention tensors suitable for
          AttentionRollout (use_rollout=True)
        - np.ndarray / array-like -- pre-flattened attention vectors

    tokens : dict, optional
        Mapping from task name to token list. Required for rollout path.
        If absent, dummy token labels are generated from sequence length.

    use_rollout : bool
        If True (default), attempt attention rollout for list inputs.

    Returns
    -------
    Dict[str, float] with pairwise task correlation keys, e.g.:
        bias_propaganda, bias_emotion, propaganda_emotion, ...
    """

    if not isinstance(attention_maps, dict):
        raise TypeError("attention_maps must be a dictionary")

    tasks = list(attention_maps.keys())
    task_vectors: Dict[str, np.ndarray] = {}

    for task, value in attention_maps.items():
        if (
            use_rollout
            and isinstance(value, list)
            and value
            and isinstance(value[0], torch.Tensor)
        ):
            try:
                task_tokens = (tokens or {}).get(task)
                if task_tokens is None:
                    seq_len = value[0].shape[-1]
                    task_tokens = [str(i) for i in range(seq_len)]

                rollout_result = _rollout.compute_rollout(
                    attentions=value,
                    tokens=task_tokens,
                )
                task_vectors[task] = np.asarray(
                    rollout_result["rollout_scores"], dtype=float
                )
                logger.debug(
                    "Rollout applied for task '%s' (%d tokens)",
                    task,
                    len(task_tokens),
                )
            except Exception as exc:
                logger.warning(
                    "AttentionRollout failed for task '%s', falling back: %s",
                    task,
                    exc,
                )
                task_vectors[task] = _safe_array(value).flatten()
        else:
            task_vectors[task] = _safe_array(value).flatten()

    correlations: Dict[str, float] = {}

    logger.info("Computing cross-task attention correlations")

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):

            task_a = tasks[i]
            task_b = tasks[j]

            a = task_vectors[task_a]
            b = task_vectors[task_b]

            if a.size != b.size:
                min_len = min(a.size, b.size)
                a = a[:min_len]
                b = b[:min_len]
                logger.debug(
                    "Truncated vectors for %s vs %s to length %d",
                    task_a,
                    task_b,
                    min_len,
                )

            try:
                corr = float(np.corrcoef(a, b)[0, 1])
            except Exception:
                corr = 0.0

            if np.isnan(corr):
                corr = 0.0

            correlations[f"{task_a}_{task_b}"] = corr

    logger.info("Cross-task attention analysis complete")

    return correlations