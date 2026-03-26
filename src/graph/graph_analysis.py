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
    collections
    numpy

Inputs:
    Graph represented as adjacency dictionary

Outputs:
    Graph metric dictionary and numerical feature vector
"""

import logging
from itertools import combinations
from typing import Dict, List

import numpy as np


logger = logging.getLogger(__name__)


class GraphAnalyzer:
    """
    Computes network-level metrics from adjacency-list graphs.
    """

    def __init__(self) -> None:
        """Initialize graph analyzer."""
        logger.info("GraphAnalyzer initialized")

    def analyze(self, graph: Dict[str, List[str]]) -> Dict[str, float]:
        """Compute structural statistics from a graph."""

        if not isinstance(graph, dict):
            raise ValueError("graph must be a dictionary")

        adjacency: dict[str, set[str]] = {}
        all_nodes: set[str] = set()

        for node, neighbors in graph.items():
            if not isinstance(node, str):
                raise ValueError("graph keys must be strings")
            if not isinstance(neighbors, list):
                raise ValueError("graph values must be lists")

            node_key = node.strip().lower()
            neighbor_set = {
                str(neighbor).strip().lower()
                for neighbor in neighbors
                if isinstance(neighbor, str)
                and neighbor.strip()
                and str(neighbor).strip().lower() != node_key
            }
            adjacency[node_key] = neighbor_set
            all_nodes.add(node_key)
            all_nodes.update(neighbor_set)

        for node in all_nodes:
            adjacency.setdefault(node, set())

        node_count = len(all_nodes)
        edge_count = sum(len(neighbors) for neighbors in adjacency.values())
        degrees = [len(adjacency[node]) for node in sorted(all_nodes)]

        avg_degree = float(np.mean(degrees)) if degrees else 0.0
        max_degree = float(max(degrees)) if degrees else 0.0
        min_degree = float(min(degrees)) if degrees else 0.0

        degree_variance = float(np.var(degrees)) if degrees else 0.0

        density = self._compute_density(node_count, edge_count)

        centralization = self._compute_centralization(degrees)

        clustering = self._estimate_clustering(adjacency)

        features = {
            "graph_nodes": float(node_count),
            "graph_edges": float(edge_count),
            "graph_avg_degree": avg_degree,
            "graph_max_degree": max_degree,
            "graph_min_degree": min_degree,
            "graph_degree_variance": degree_variance,
            "graph_density": density,
            "graph_centralization": centralization,
            "graph_clustering_estimate": clustering,
        }

        return features

    def _compute_density(self, nodes: int, edges: int) -> float:
        """Compute graph density."""

        if nodes <= 1:
            return 0.0

        possible_edges = nodes * (nodes - 1)

        return float(edges / possible_edges)

    def _compute_centralization(self, degrees: List[int]) -> float:
        """Estimate network centralization."""

        if not degrees:
            return 0.0

        max_degree = max(degrees)

        diff_sum = sum(max_degree - d for d in degrees)

        normalization = len(degrees) * (len(degrees) - 1)

        if normalization == 0:
            return 0.0

        return float(diff_sum / normalization)

    def _estimate_clustering(self, graph: Dict[str, set[str]]) -> float:
        """Estimate clustering coefficient using neighbor overlap."""
        if not graph:
            return 0.0

        # Convert to undirected adjacency for clustering coefficient estimate.
        undirected = {node: set(neighbors) for node, neighbors in graph.items()}
        for node, neighbors in list(undirected.items()):
            for neighbor in neighbors:
                undirected.setdefault(neighbor, set()).add(node)

        local_coefficients: list[float] = []
        for neighbors in undirected.values():
            degree = len(neighbors)
            if degree < 2:
                local_coefficients.append(0.0)
                continue

            neighbor_list = sorted(neighbors)
            links_between_neighbors = 0
            for left, right in combinations(neighbor_list, 2):
                if right in undirected.get(left, set()):
                    links_between_neighbors += 1

            possible_links = degree * (degree - 1) / 2.0
            local_coefficients.append(links_between_neighbors / possible_links)

        return float(np.mean(local_coefficients)) if local_coefficients else 0.0


def graph_feature_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert graph metric dictionary into numerical vector."""

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = np.array(list(features.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Graph feature vector conversion failed")
        raise RuntimeError("Failed to convert graph metrics") from exc
