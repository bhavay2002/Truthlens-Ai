from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_graph(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}

    for node, neighbors in graph.items():
        nk = node.strip().lower()
        seen: Set[str] = set()

        clean = []
        for nbr in neighbors:
            if isinstance(nbr, str):
                n = nbr.strip().lower()
                if n and n != nk and n not in seen:
                    seen.add(n)
                    clean.append(n)

        normalized[nk] = sorted(clean)

    return normalized


def to_undirected(graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
    adj: Dict[str, Set[str]] = {n: set(v) for n, v in graph.items()}

    for node, neighbors in graph.items():
        for nbr in neighbors:
            adj.setdefault(nbr, set()).add(node)

    return {k: sorted(v) for k, v in adj.items()}


def unique_edges(graph: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    edges = set()

    for a, nbrs in graph.items():
        for b in nbrs:
            edges.add(tuple(sorted((a, b))))

    return list(edges)


# =========================================================
# CORE METRICS
# =========================================================

def compute_graph_metrics(graph: Dict[str, List[str]]) -> Dict[str, float]:

    graph = to_undirected(normalize_graph(graph))

    nodes = list(graph.keys())
    n = len(nodes)

    edges = unique_edges(graph)
    e = len(edges)

    degrees = np.array([len(graph[n]) for n in nodes], dtype=float)

    # -------------------------
    # BASIC
    # -------------------------
    avg_degree = float(np.mean(degrees)) if n > 0 else 0.0
    max_degree = float(np.max(degrees)) if n > 0 else 0.0
    min_degree = float(np.min(degrees)) if n > 0 else 0.0
    var_degree = float(np.var(degrees)) if n > 0 else 0.0

    density = float((2 * e) / (n * (n - 1) + EPS)) if n > 1 else 0.0

    centralization = (
        float((max_degree - avg_degree) / (n - 1 + EPS)) if n > 1 else 0.0
    )

    # -------------------------
    # CLUSTERING
    # -------------------------
    adj_sets = {k: set(v) for k, v in graph.items()}
    triangles = 0
    triplets = 0

    for node in nodes:
        nbrs = list(adj_sets[node])
        k = len(nbrs)

        if k < 2:
            continue

        triplets += k * (k - 1) / 2

        for i in range(k):
            for j in range(i + 1, k):
                if nbrs[j] in adj_sets[nbrs[i]]:
                    triangles += 1

    clustering = float(triangles / (triplets + EPS))

    # -------------------------
    # CENTRALITY (degree-based)
    # -------------------------
    centrality = degrees / (n - 1 + EPS) if n > 1 else degrees
    centrality_mean = float(np.mean(centrality)) if n > 0 else 0.0
    centrality_var = float(np.var(centrality)) if n > 0 else 0.0

    # -------------------------
    # ENTROPY (🔥 important)
    # -------------------------
    if np.sum(degrees) > 0:
        p = degrees / (np.sum(degrees) + EPS)
        entropy = float(-np.sum(p * np.log(p + EPS)))
    else:
        entropy = 0.0

    return {
        "graph_nodes": float(n),
        "graph_edges": float(e),
        "graph_avg_degree": avg_degree,
        "graph_max_degree": max_degree,
        "graph_min_degree": min_degree,
        "graph_degree_variance": var_degree,
        "graph_density": density,
        "graph_centralization": centralization,
        "graph_clustering": clustering,
        "graph_centrality_mean": centrality_mean,
        "graph_centrality_variance": centrality_var,
        "graph_entropy": entropy,
    }


# =========================================================
# DATACLASS
# =========================================================

@dataclass
class GraphMetrics:
    graph_nodes: float
    graph_edges: float
    graph_avg_degree: float
    graph_max_degree: float
    graph_min_degree: float
    graph_degree_variance: float
    graph_density: float
    graph_centralization: float
    graph_clustering: float
    graph_centrality_mean: float
    graph_centrality_variance: float
    graph_entropy: float

    def to_dict(self) -> Dict[str, float]:
        return self.__dict__


# =========================================================
# ANALYZER
# =========================================================

class GraphAnalyzer:

    def __init__(self):
        logger.info("GraphAnalyzer initialized")

    def analyze(self, graph: Dict[str, List[str]]) -> GraphMetrics:

        if not isinstance(graph, dict):
            raise TypeError("graph must be dictionary")

        metrics = compute_graph_metrics(graph)

        return GraphMetrics(**metrics)


# =========================================================
# VECTOR
# =========================================================

def graph_to_vector(features: Dict[str, float]) -> np.ndarray:

    keys = [
        "graph_nodes",
        "graph_edges",
        "graph_avg_degree",
        "graph_max_degree",
        "graph_min_degree",
        "graph_degree_variance",
        "graph_density",
        "graph_centralization",
        "graph_clustering",
        "graph_centrality_mean",
        "graph_centrality_variance",
        "graph_entropy",
    ]

    return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)