"""
File Name: graph_analysis.py
Module: Graph Analysis - Network Metrics
Description:
    Provides utilities for computing structural metrics from graphs used in the
    TruthLens AI system. The module analyzes graphs such as entity graphs and
    narrative graphs to compute network statistics including degree metrics,
    density, centralization, connectivity, and clustering signals.

Dependencies:
    logging
    typing
    dataclasses
    itertools
    numpy

Inputs:
    Graph represented as adjacency dictionary

Outputs:
    Graph metric dictionary and numerical feature vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_graph_adjacency(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for node, neighbors in graph.items():
        node_key = node.strip().lower()
        seen: Set[str] = set()
        clean_neighbors: List[str] = []
        for neighbor in neighbors:
            if isinstance(neighbor, str):
                nk = neighbor.strip().lower()
                if nk and nk != node_key and nk not in seen:
                    seen.add(nk)
                    clean_neighbors.append(nk)
        normalized[node_key] = sorted(clean_neighbors)
    return normalized


def _to_undirected(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    adj: Dict[str, Set[str]] = {node: set(neighbors) for node, neighbors in graph.items()}

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in adj:
                adj[neighbor] = set()
            adj[neighbor].add(node)

    return {node: sorted(neighbors) for node, neighbors in adj.items()}


def _unique_undirected_edges(graph: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edge = (min(node, neighbor), max(node, neighbor))
            edges.add(edge)
    return sorted(edges)


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
    undirected = _to_undirected(_normalize_graph_adjacency(graph))
    nodes = list(undirected.keys())
    node_count = len(nodes)
    edges = _unique_undirected_edges(undirected)
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

    triangle_count = 0
    adj_sets = {n: set(undirected[n]) for n in nodes}
    for node in nodes:
        nbrs = list(adj_sets[node])
        for i, u in enumerate(nbrs):
            for v in nbrs[i + 1 :]:
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
    Return fixed-order numpy vector of graph metric features.
    """
    return np.array(
        [float(features.get(k, 0.0)) for k in _GRAPH_METRICS_KEYS],
        dtype=np.float32,
    )


@dataclass(slots=True)
class GraphMetrics:
    """
    Dataclass container for graph metrics.
    """

    graph_nodes: float
    graph_edges: float
    graph_avg_degree: float
    graph_max_degree: float
    graph_min_degree: float
    graph_degree_variance: float
    graph_density: float
    graph_centralization: float
    graph_clustering_estimate: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics dataclass to dictionary."""
        return {
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "graph_avg_degree": self.graph_avg_degree,
            "graph_max_degree": self.graph_max_degree,
            "graph_min_degree": self.graph_min_degree,
            "graph_degree_variance": self.graph_degree_variance,
            "graph_density": self.graph_density,
            "graph_centralization": self.graph_centralization,
            "graph_clustering_estimate": self.graph_clustering_estimate,
        }


class GraphAnalyzer:
    """
    Computes network-level metrics from adjacency-list graphs.
    """

    def __init__(self) -> None:
        """Initialize graph analyzer."""
        logger.info("GraphAnalyzer initialized")

    def _validate_graph(self, graph: Dict[str, List[str]]) -> None:
        """Validate graph structure."""
        if not isinstance(graph, dict):
            raise TypeError("graph must be a dictionary")

        for node, neighbors in graph.items():
            if not isinstance(node, str):
                raise ValueError("graph keys must be strings")
            if not isinstance(neighbors, list):
                raise ValueError("graph values must be lists of neighbors")
            if not all(isinstance(n, str) for n in neighbors):
                raise ValueError("all neighbors must be strings")

    def analyze(self, graph: Dict[str, List[str]]) -> GraphMetrics:
        """
        Compute structural statistics from a graph.

        Parameters
        ----------
        graph : Dict[str, List[str]]

        Returns
        -------
        GraphMetrics
        """

        self._validate_graph(graph)

        metrics_dict = compute_undirected_basic_metrics(graph)

        metrics = GraphMetrics(
            graph_nodes=metrics_dict["graph_nodes"],
            graph_edges=metrics_dict["graph_edges"],
            graph_avg_degree=metrics_dict["graph_avg_degree"],
            graph_max_degree=metrics_dict["graph_max_degree"],
            graph_min_degree=metrics_dict["graph_min_degree"],
            graph_degree_variance=metrics_dict["graph_degree_variance"],
            graph_density=metrics_dict["graph_density"],
            graph_centralization=metrics_dict["graph_centralization"],
            graph_clustering_estimate=metrics_dict["graph_clustering_estimate"],
        )

        logger.debug("Graph metrics computed: %s", metrics)

        return metrics


def graph_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert graph metric dictionary into numerical vector.
    """

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = ordered_graph_metrics_vector(features)
        return vector
    except Exception as exc:  # pragma: no cover
        logger.exception("Graph feature vector conversion failed")
        raise RuntimeError("Failed to convert graph metrics") from exc