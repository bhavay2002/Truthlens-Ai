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
from typing import Dict, List

import numpy as np  # retained for compatibility if helpers are reintroduced

from graph_hardening_patch import compute_undirected_basic_metrics, ordered_graph_metrics_vector

logger = logging.getLogger(__name__)


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