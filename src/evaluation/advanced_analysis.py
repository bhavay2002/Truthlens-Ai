"""
File: advanced_analysis.py (REFRACTORED - MULTI-TASK READY)

Key Upgrades:
- Task-aware prediction (task routing)
- Batched inference (no OOM)
- Proper logits → predictions handling
- GPU-safe execution
- Attention alignment strategies
- Scalable SHAP + importance methods
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Any, List, Optional, Literal

import numpy as np
import pandas as pd
import torch
import networkx as nx

from src.explainability.attention_rollout import AttentionRollout
from src.features.importance.feature_ablation import FeatureAblation
from src.features.importance.permutation_importance import PermutationImportance
from src.features.importance.shap_importance import ShapImportance

logger = logging.getLogger(__name__)
_rollout = AttentionRollout()


# =========================================================
# DEVICE UTIL
# =========================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# BATCHED PREDICTION (CRITICAL)
# =========================================================
def batched_predict(
    model,
    X: np.ndarray,
    task: str,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
):
    device = device or get_device()
    model.to(device)
    model.eval()

    outputs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]

            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)

            result = model.predict(batch_tensor, task=task)
            logits = result["logits"].detach().cpu().numpy()

            outputs.append(logits)

    return np.vstack(outputs)


# =========================================================
# PREDICTION POST-PROCESSING
# =========================================================
def postprocess_predictions(logits: np.ndarray, task_type: str):
    if task_type == "multiclass":
        probs = softmax(logits)
        preds = np.argmax(probs, axis=1)

    elif task_type == "multilabel":
        probs = sigmoid(logits)
        preds = (probs > 0.5).astype(int)

    elif task_type == "binary":
        probs = sigmoid(logits)
        preds = (probs > 0.5).astype(int)

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    return preds, probs


def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =========================================================
# ACTOR GRAPH (UPGRADED)
# =========================================================
def actor_graph_metrics(df: pd.DataFrame) -> Dict[str, float]:
    graph = nx.DiGraph()

    for h, v in zip(df["hero_entities"], df["villain_entities"]):
        if pd.notna(h) and pd.notna(v):
            graph.add_edge(str(h), str(v))

    if graph.number_of_nodes() == 0:
        return {}

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_degree": float(np.mean([d for _, d in graph.degree()])),
        "pagerank_mean": float(np.mean(list(nx.pagerank(graph).values()))),
        "components": nx.number_weakly_connected_components(graph),
    }


# =========================================================
# FRAME COHERENCE
# =========================================================
def frame_coherence(pred, true) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)

    if pred.shape != true.shape:
        raise ValueError("Shape mismatch")

    return float(np.mean(pred == true))


# =========================================================
# ATTENTION ALIGNMENT
# =========================================================
def align_attention(a, b, strategy="truncate"):
    if a.size == b.size:
        return a, b

    if strategy == "truncate":
        min_len = min(len(a), len(b))
        return a[:min_len], b[:min_len]

    elif strategy == "pad":
        max_len = max(len(a), len(b))
        a = np.pad(a, (0, max_len - len(a)))
        b = np.pad(b, (0, max_len - len(b)))
        return a, b

    else:
        raise ValueError("Invalid alignment strategy")


# =========================================================
# CROSS TASK ATTENTION (FIXED)
# =========================================================
def cross_task_attention(
    attention_maps: Dict[str, Any],
    use_rollout: bool = True,
    align_strategy: str = "truncate",
):
    task_vectors = {}

    for task, value in attention_maps.items():
        if use_rollout and isinstance(value, list):
            rollout = _rollout.compute_rollout(attentions=value)
            vec = np.asarray(rollout["rollout_scores"])
        else:
            vec = np.asarray(value).ravel()

        task_vectors[task] = vec

    correlations = {}

    tasks = list(task_vectors.keys())

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            a, b = align_attention(
                task_vectors[tasks[i]],
                task_vectors[tasks[j]],
                align_strategy,
            )

            if len(a) < 2:
                corr = 0.0
            else:
                corr = np.corrcoef(a, b)[0, 1]

            correlations[f"{tasks[i]}_{tasks[j]}"] = float(
                0.0 if np.isnan(corr) else corr
            )

    return correlations


# =========================================================
# ABLATION IMPORTANCE (MULTI-TASK SAFE)
# =========================================================
def ablation_importance(
    model,
    X,
    y,
    feature_names,
    task: str,
    task_type: str,
    metric: Callable,
    batch_size: int = 32,
):
    def predict_fn(X_batch):
        logits = batched_predict(model, X_batch, task, batch_size)
        preds, _ = postprocess_predictions(logits, task_type)
        return preds

    ablator = FeatureAblation(model=None, metric=metric)
    return ablator.single_feature_ablation(
        X=X,
        y=y,
        feature_names=feature_names,
        predict_fn=predict_fn,
    )


# =========================================================
# PERMUTATION IMPORTANCE
# =========================================================
def permutation_importance(
    model,
    X,
    y,
    feature_names,
    task,
    task_type,
    metric,
    n_repeats=5,
    batch_size=32,
):
    def predict_fn(X_batch):
        logits = batched_predict(model, X_batch, task, batch_size)
        preds, _ = postprocess_predictions(logits, task_type)
        return preds

    perm = PermutationImportance(metric=metric)

    return perm.compute(
        X=X,
        y=y,
        feature_names=feature_names,
        predict_fn=predict_fn,
        n_repeats=n_repeats,
    )


# =========================================================
# SHAP IMPORTANCE (OPTIMIZED)
# =========================================================
def shap_importance(
    model,
    X,
    feature_names,
    task,
    task_type,
    max_samples=500,
    batch_size=32,
):
    X_small = X[:max_samples]

    def predict_fn(X_batch):
        logits = batched_predict(model, X_batch, task, batch_size)
        _, probs = postprocess_predictions(logits, task_type)
        return probs

    shap_calc = ShapImportance(model=None)
    scores = shap_calc.compute_with_function(
        predict_fn=predict_fn,
        X=X_small,
        feature_names=feature_names,
    )

    return scores


# =========================================================
# FEATURE DIAGNOSTICS (UNCHANGED CORE, SAFE)
# =========================================================
def feature_diagnostics(texts: List[str]) -> Dict[str, Any]:
    from src.features.dataset_feature_generator import DatasetFeatureGenerator
    from src.features.feature_statistics import FeatureStatistics

    generator = DatasetFeatureGenerator()
    X, names = generator.generate(texts)

    stats = FeatureStatistics()

    return {
        "summary": stats.dataset_summary(X),
        "variance": stats.compute_variance(X),
    }