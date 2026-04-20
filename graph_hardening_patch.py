"""
graph_hardening_patch.py
Utility functions for the TruthLens graph subsystem.
Provides helper implementations required by entity_graph, graph_analysis,
graph_config, graph_embeddings, graph_features, graph_pipeline, and
graph_visualization modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# entity_graph helpers
# ---------------------------------------------------------------------------

def normalize_graph_adjacency(
    graph: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Normalise an adjacency-list graph so that node names are lower-cased and
    stripped of surrounding whitespace.  Duplicate neighbour entries are
    removed and each neighbour list is sorted for determinism.
    """
    normalised: Dict[str, List[str]] = {}
    for node, neighbours in graph.items():
        node_key = node.strip().lower()
        seen: Set[str] = set()
        clean_neighbours: List[str] = []
        for n in neighbours:
            if isinstance(n, str):
                nk = n.strip().lower()
                if nk and nk != node_key and nk not in seen:
                    seen.add(nk)
                    clean_neighbours.append(nk)
        normalised[node_key] = sorted(clean_neighbours)
    return normalised


def to_undirected(
    graph: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Convert a (possibly directed) adjacency-list graph to an undirected one by
    ensuring every edge (u, v) also has a reverse edge (v, u).
    """
    adj: Dict[str, Set[str]] = {node: set(neighbours) for node, neighbours in graph.items()}

    for node, neighbours in graph.items():
        for neighbour in neighbours:
            if neighbour not in adj:
                adj[neighbour] = set()
            adj[neighbour].add(node)

    return {node: sorted(neighbours) for node, neighbours in adj.items()}


def unique_undirected_edges(
    graph: Dict[str, List[str]]
) -> List[Tuple[str, str]]:
    """
    Return the set of unique undirected edges from an adjacency-list graph as
    sorted (u, v) pairs where u < v.
    """
    edges: Set[Tuple[str, str]] = set()
    for node, neighbours in graph.items():
        for neighbour in neighbours:
            edge = (min(node, neighbour), max(node, neighbour))
            edges.add(edge)
    return sorted(edges)


_ENTITY_GRAPH_KEYS: List[str] = [
    "entity_graph_nodes",
    "entity_graph_edges",
    "entity_graph_avg_degree",
    "entity_graph_density",
    "entity_graph_dominant_degree",
    "entity_graph_degree_variance",
]


def ordered_entity_graph_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Return a fixed-order numpy vector of entity-graph features.
    Missing keys default to 0.0.
    """
    return np.array(
        [float(features.get(k, 0.0)) for k in _ENTITY_GRAPH_KEYS],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# graph_analysis helpers
# ---------------------------------------------------------------------------

_GRAPH_METRICS_KEYS: List[str] = [
    "graph_nodes",
    "graph_edges",
    "graph_avg_degree",
    "graph_max_degree",
    "graph_min_degree",
    "graph_degree_variance",
    "graph_density",
    "graph_centralization",
    "graph_clustering_estimate",
]


def compute_undirected_basic_metrics(
    graph: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Compute basic structural metrics for an undirected adjacency-list graph.
    """
    undirected = to_undirected(normalize_graph_adjacency(graph))
    nodes = list(undirected.keys())
    node_count = len(nodes)
    edges = unique_undirected_edges(undirected)
    edge_count = len(edges)

    degrees = [len(undirected[n]) for n in nodes]

    avg_degree = float(np.mean(degrees)) if degrees else 0.0
    max_degree = float(max(degrees, default=0))
    min_degree = float(min(degrees, default=0))
    degree_variance = float(np.var(degrees)) if degrees else 0.0

    density = (
        float(2 * edge_count / (node_count * (node_count - 1)))
        if node_count > 1
        else 0.0
    )

    centralization = (
        float((max_degree - avg_degree) / (node_count - 1))
        if node_count > 1
        else 0.0
    )

    # Clustering estimate: fraction of triangles possible
    triangle_count = 0
    adj_sets = {n: set(undirected[n]) for n in nodes}
    for node in nodes:
        nbrs = list(adj_sets[node])
        for i, u in enumerate(nbrs):
            for v in nbrs[i + 1:]:
                if v in adj_sets.get(u, set()):
                    triangle_count += 1

    max_triangles = sum(d * (d - 1) // 2 for d in degrees)
    clustering_estimate = (
        float(triangle_count / max_triangles) if max_triangles > 0 else 0.0
    )

    return {
        "graph_nodes": float(node_count),
        "graph_edges": float(edge_count),
        "graph_avg_degree": avg_degree,
        "graph_max_degree": max_degree,
        "graph_min_degree": min_degree,
        "graph_degree_variance": degree_variance,
        "graph_density": density,
        "graph_centralization": centralization,
        "graph_clustering_estimate": clustering_estimate,
    }


def ordered_graph_metrics_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Return a fixed-order numpy vector of graph metric features.
    Missing keys default to 0.0.
    """
    return np.array(
        [float(features.get(k, 0.0)) for k in _GRAPH_METRICS_KEYS],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# graph_config helpers
# ---------------------------------------------------------------------------

def load_yaml_as_dict(path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at {path}, got {type(data)}")
    return data


def parse_graph_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate the graph subsection from a configuration dictionary.
    Falls back to sensible defaults for missing keys.
    """
    graph_section = config_data.get("graph", config_data)

    parsed: Dict[str, Any] = {
        "enable_entity_graph": bool(graph_section.get("enable_entity_graph", True)),
        "enable_narrative_graph": bool(graph_section.get("enable_narrative_graph", True)),
        "min_keyword_length": int(graph_section.get("min_keyword_length", 4)),
        "max_keywords_per_sentence": int(graph_section.get("max_keywords_per_sentence", 4)),
    }
    return parsed


# ---------------------------------------------------------------------------
# graph_embeddings helpers
# ---------------------------------------------------------------------------

def spectral_eigen_embedding(
    adjacency_matrix: np.ndarray,
    dim: int = 8,
) -> np.ndarray:
    """
    Compute a spectral embedding from the top-`dim` eigenvalues of the
    adjacency matrix.
    """
    if adjacency_matrix.size == 0:
        return np.zeros(dim, dtype=np.float32)

    try:
        eigenvalues = np.linalg.eigvalsh(adjacency_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
    except np.linalg.LinAlgError:
        logger.warning("Eigenvalue decomposition failed; returning zeros")
        return np.zeros(dim, dtype=np.float32)

    if len(eigenvalues) >= dim:
        result = eigenvalues[:dim].astype(np.float32)
    else:
        result = np.pad(
            eigenvalues.astype(np.float32),
            (0, dim - len(eigenvalues)),
            mode="constant",
        )
    return result


# ---------------------------------------------------------------------------
# graph_features helpers
# ---------------------------------------------------------------------------

_NARRATIVE_GRAPH_KEYS: List[str] = [
    "narrative_graph_nodes",
    "narrative_graph_edges",
    "narrative_graph_avg_degree",
    "narrative_graph_density",
    "narrative_graph_isolated_nodes",
    "narrative_graph_components",
]


def ordered_narrative_graph_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Return a fixed-order numpy vector of narrative-graph features.
    Missing keys default to 0.0.
    """
    return np.array(
        [float(features.get(k, 0.0)) for k in _NARRATIVE_GRAPH_KEYS],
        dtype=np.float32,
    )


def merge_feature_blocks_strict(*blocks: Dict[str, float]) -> Dict[str, float]:
    """
    Merge multiple feature dictionaries into one.  Raises a ValueError if the
    same key appears in more than one block (strict mode).
    """
    merged: Dict[str, float] = {}
    for block in blocks:
        for key, value in block.items():
            if key in merged:
                raise ValueError(
                    f"Duplicate feature key '{key}' found during merge"
                )
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# graph_pipeline helpers
# ---------------------------------------------------------------------------

@dataclass
class _PipelineExtractorConfig:
    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True


def build_pipeline_feature_extractor_config(
    enable_entity_graph: bool = True,
    enable_narrative_graph: bool = True,
) -> _PipelineExtractorConfig:
    """
    Build a feature-extractor configuration object for the graph pipeline.
    """
    return _PipelineExtractorConfig(
        enable_entity_graph=enable_entity_graph,
        enable_narrative_graph=enable_narrative_graph,
    )


# ---------------------------------------------------------------------------
# graph_visualization helpers
# ---------------------------------------------------------------------------

def ensure_headless_matplotlib_backend() -> None:
    """
    Switch matplotlib to the 'Agg' (non-interactive) backend so that
    visualisation code can run in headless server environments without a
    display.  Safe to call multiple times.
    """
    import matplotlib
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
