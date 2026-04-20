"""
File Name: advanced_analysis.py
Module: TruthLens AI - Advanced Evaluation
Description:
    Research-grade evaluation diagnostics used for deeper model analysis
    in the TruthLens AI system. Provides graph-based narrative metrics,
    frame prediction coherence, cross-task attention correlation analysis
    for multi-task transformer models, and feature importance diagnostics
    via ablation, permutation, and SHAP-based methods.

    Integrates AttentionRollout for richer cross-task attention analysis:
    when raw per-layer attention tensors are provided, rollout scores are
    computed and used in place of (or alongside) flattened attention maps,
    yielding more reliable cumulative attention-flow correlations.

    Feature importance integration:
        FeatureAblation       — contribution via systematic feature removal
        PermutationImportance — contribution via random feature shuffling
        ShapImportance        — contribution via SHAP Shapley values

Dependencies:
    logging
    typing
    numpy
    networkx
    pandas
    src.explainability.attention_rollout
    src.features.importance.feature_ablation
    src.features.importance.permutation_importance
    src.features.importance.shap_importance

Inputs:
    df: DataFrame containing narrative entity columns
    pred_frames: predicted frame labels
    true_frames: ground truth frame labels
    attention_maps: dictionary of task attention tensors or rollout inputs
    model: any object exposing a predict() interface
    X: numpy feature matrix
    y: label array
    feature_names: list of feature name strings
Outputs:
    Dictionary containing diagnostic evaluation metrics
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Any, List, Optional, Literal

import numpy as np
import networkx as nx
import pandas as pd
import torch

from src.explainability.attention_rollout import AttentionRollout
from src.features.importance.feature_ablation import FeatureAblation
from src.features.importance.permutation_importance import PermutationImportance
from src.features.importance.shap_importance import ShapImportance

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

    for hero, villain in zip(df["hero_entities"], df["villain_entities"]):

        if (
            pd.notna(hero)
            and pd.notna(villain)
            and str(hero).strip()
            and str(villain).strip()
        ):
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
    align: Literal["strict", "truncate"] = "strict",
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
                try:
                    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                        task_vectors[task] = np.concatenate(
                            [v.detach().cpu().numpy().ravel() for v in value]
                        ).astype(float, copy=False)
                    else:
                        task_vectors[task] = _safe_array(value).astype(float, copy=False).ravel()
                except Exception as inner_exc:
                    raise ValueError(
                        f"Failed to build fallback attention vector for task '{task}'"
                    ) from inner_exc
        else:
            task_vectors[task] = _safe_array(value).astype(float, copy=False).ravel()

    correlations: Dict[str, float] = {}

    logger.info("Computing cross-task attention correlations")

    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):

            task_a = tasks[i]
            task_b = tasks[j]

            a = task_vectors[task_a]
            b = task_vectors[task_b]

            if a.size != b.size:
                if align == "strict":
                    raise ValueError(
                        f"Mismatched attention lengths for {task_a} vs {task_b}: {a.size} != {b.size}"
                    )
                min_len = min(a.size, b.size)
                a = a[:min_len]
                b = b[:min_len]
                logger.debug(
                    "Truncated vectors for %s vs %s to length %d",
                    task_a,
                    task_b,
                    min_len,
                )

            if a.size < 2 or b.size < 2:
                correlations[f"{task_a}_{task_b}"] = 0.0
                continue

            try:
                corr = float(np.corrcoef(a, b)[0, 1])
            except Exception:
                corr = 0.0

            if np.isnan(corr):
                corr = 0.0

            correlations[f"{task_a}_{task_b}"] = corr

    logger.info("Cross-task attention analysis complete")

    return correlations


# ---------------------------------------------------------------------------
# Feature Importance Diagnostics
# ---------------------------------------------------------------------------

def ablation_importance(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Measure feature importance via systematic ablation.

    Each feature is zeroed out in turn and the resulting drop in model
    performance is used as its importance score.

    Parameters
    ----------
    model : object
        Any model that exposes a ``predict(X)`` method.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Ground-truth label array of shape (n_samples,).
    feature_names : List[str]
        Names aligned with the columns of X.
    metric : callable, optional
        Evaluation function ``metric(y_true, y_pred) -> float``.
        Defaults to accuracy if not provided.
    top_k : int, optional
        If given, return only the top-k most important features.

    Returns
    -------
    Dict[str, Any] with keys:
        "scores"   — dict mapping feature name to importance score
        "ranked"   — list of (feature_name, score) sorted descending
        "top_k"    — list of (feature_name, score) for the top-k features
                     (only present when top_k is specified)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"X column count ({X.shape[1]})"
        )
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    logger.info(
        "Running ablation importance | features=%d samples=%d",
        len(feature_names),
        X.shape[0],
    )

    kwargs: Dict[str, Any] = {}
    if metric is not None:
        kwargs["metric"] = metric

    ablator = FeatureAblation(model=model, **kwargs)
    scores = ablator.single_feature_ablation(X=X, y=y, feature_names=feature_names)
    ranked = ablator.rank_features(scores)

    result: Dict[str, Any] = {
        "scores": scores,
        "ranked": ranked,
    }

    if top_k is not None:
        result["top_k"] = ablator.top_k(scores, k=top_k)

    logger.info("Ablation importance complete | features_scored=%d", len(scores))
    return result


def permutation_importance(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    metric: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_repeats: int = 5,
    random_seed: int = 42,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Measure feature importance via random permutation.

    Each feature's values are shuffled ``n_repeats`` times and the mean
    performance drop across repeats is used as the importance score.

    Parameters
    ----------
    model : object
        Any model that exposes a ``predict(X)`` method.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Ground-truth label array of shape (n_samples,).
    feature_names : List[str]
        Names aligned with the columns of X.
    metric : callable, optional
        Evaluation function ``metric(y_true, y_pred) -> float``.
        Defaults to accuracy if not provided.
    n_repeats : int
        Number of shuffle repeats per feature. Default is 5.
    random_seed : int
        Random seed for reproducibility. Default is 42.
    top_k : int, optional
        If given, return only the top-k most important features.

    Returns
    -------
    Dict[str, Any] with keys:
        "scores"   — dict mapping feature name to importance score
        "ranked"   — list of (feature_name, score) sorted descending
        "top_k"    — list of (feature_name, score) for the top-k features
                     (only present when top_k is specified)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of samples")
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"X column count ({X.shape[1]})"
        )
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    logger.info(
        "Running permutation importance | features=%d samples=%d repeats=%d",
        len(feature_names),
        X.shape[0],
        n_repeats,
    )

    kwargs: Dict[str, Any] = {"model": model}
    if metric is not None:
        kwargs["metric"] = metric

    perm = PermutationImportance(**kwargs)
    scores = perm.compute(
        X=X,
        y=y,
        feature_names=feature_names,
        n_repeats=n_repeats,
        random_seed=random_seed,
    )
    ranked = perm.rank_features(scores)

    result: Dict[str, Any] = {
        "scores": scores,
        "ranked": ranked,
    }

    if top_k is not None:
        result["top_k"] = perm.top_k(scores, k=top_k)

    logger.info("Permutation importance complete | features_scored=%d", len(scores))
    return result


def shap_importance(
    model: object,
    X: np.ndarray,
    feature_names: List[str],
    max_samples: Optional[int] = 1000,
    random_seed: int = 42,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Measure feature importance via SHAP (Shapley Additive Explanations).

    SHAP values are computed using the model's prediction interface.
    The explainer is selected automatically; KernelExplainer is used
    as a fallback when tree or linear explainers are not applicable.

    Parameters
    ----------
    model : object
        Any model that exposes a ``predict(X)`` method.
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    feature_names : List[str]
        Names aligned with the columns of X.
    max_samples : int, optional
        Maximum number of background samples for the SHAP explainer.
        Default is 1000.
    random_seed : int
        Random seed for reproducibility. Default is 42.
    top_k : int, optional
        If given, return only the top-k most important features.

    Returns
    -------
    Dict[str, Any] with keys:
        "scores"   — dict mapping feature name to mean |SHAP| value
        "ranked"   — list of (feature_name, score) sorted descending
        "top_k"    — list of (feature_name, score) for the top-k features
                     (only present when top_k is specified)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy ndarray")
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length ({len(feature_names)}) must match "
            f"X column count ({X.shape[1]})"
        )
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")

    logger.info(
        "Running SHAP importance | features=%d samples=%d",
        len(feature_names),
        X.shape[0],
    )

    shap_calc = ShapImportance(
        model=model,
        max_samples=max_samples,
        random_seed=random_seed,
    )
    scores = shap_calc.compute(X=X, feature_names=feature_names)
    ranked = shap_calc.rank_features(scores)

    result: Dict[str, Any] = {
        "scores": scores,
        "ranked": ranked,
    }

    if top_k is not None:
        result["top_k"] = shap_calc.top_k(scores, k=top_k)

    logger.info("SHAP importance complete | features_scored=%d", len(scores))
    return result


def feature_diagnostics(
    texts: List[str],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract TruthLens pipeline features from raw texts and return dataset-level
    diagnostics.

    Uses ``DatasetFeatureGenerator`` to run the full feature pipeline on each
    text, ``FeatureStatistics`` to compute aggregate statistics, and
    ``FeatureSchemaValidator`` to verify that the extracted feature set
    matches the expected schema.

    Parameters
    ----------
    texts:
        Raw article texts to process.
    feature_names:
        Optional list of expected feature names for schema validation.
        When ``None`` the schema is inferred from the extracted feature names.

    Returns
    -------
    Dict with keys:

    * ``summary``           — dataset-level stats (num_samples, num_features,
                              mean / std / min / max variance).
    * ``constant_features`` — names of zero-variance features.
    * ``basic_statistics``  — per-feature mean, std, min, max.
    * ``variance``          — per-feature variance values.
    * ``schema_validated``  — ``True`` when ``FeatureSchemaValidator`` passed.
    """
    if not texts:
        raise ValueError("texts must not be empty")
    if any((t is None or not str(t).strip()) for t in texts):
        raise ValueError("texts must contain non-empty strings")

    from src.features.dataset_feature_generator import DatasetFeatureGenerator
    from src.features.feature_schema_validator import FeatureSchemaValidator
    from src.features.feature_statistics import FeatureStatistics
    from src.features.pipelines.batch_feature_pipeline import BatchFeaturePipeline
    from src.features.pipelines.feature_pipeline import FeaturePipeline

    logger.info(
        "Running feature_diagnostics | samples=%d",
        len(texts),
    )

    fp = FeaturePipeline()
    fp.initialize()
    batch_pipeline = BatchFeaturePipeline(pipeline=fp)
    generator = DatasetFeatureGenerator(pipeline=batch_pipeline)

    feature_matrix, inferred_names = generator.generate(texts)

    feature_dicts: List[Dict[str, float]] = [
        dict(zip(inferred_names, row)) for row in feature_matrix.tolist()
    ]

    schema: List[str] = feature_names if feature_names is not None else inferred_names

    stats = FeatureStatistics()
    summary = stats.dataset_summary(feature_dicts)
    basic = stats.compute_basic_statistics(feature_dicts)
    variance = stats.compute_variance(feature_dicts)
    constant = stats.detect_constant_features(feature_dicts)

    validator = FeatureSchemaValidator(
        expected_features=schema,
        strict=False,
        allow_missing=True,
        allow_extra=True,
    )
    validator.validate_batch(feature_dicts)

    logger.info(
        "feature_diagnostics complete | features=%d samples=%d constant=%d",
        int(summary["num_features"]),
        int(summary["num_samples"]),
        len(constant),
    )

    return {
        "summary": summary,
        "constant_features": constant,
        "basic_statistics": basic,
        "variance": variance,
        "schema_validated": True,
    }
