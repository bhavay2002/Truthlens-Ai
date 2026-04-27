from __future__ import annotations

import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Set, Tuple

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_graph(graph: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    normalized: Dict[str, Dict[str, float]] = {}

    for node, neighbors in graph.items():
        nk = node.strip().lower()
        normalized[nk] = {}

        for nbr, w in neighbors.items():
            nbrk = nbr.strip().lower()
            if nbrk and nbrk != nk:
                normalized[nk][nbrk] = float(w)

    return normalized


def to_undirected(graph: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    undirected: Dict[str, Dict[str, float]] = defaultdict(dict)

    for node, neighbors in graph.items():
        for nbr, w in neighbors.items():
            undirected[node][nbr] = undirected[node].get(nbr, 0.0) + w
            undirected[nbr][node] = undirected[nbr].get(node, 0.0) + w

    return dict(undirected)


def unique_edges(graph: Dict[str, Dict[str, float]]) -> List[Tuple[str, str]]:
    seen = set()
    edges = []

    for a, neighbors in graph.items():
        for b in neighbors:
            edge = tuple(sorted((a, b)))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

    return edges


# =========================================================
# FEATURES
# =========================================================

@dataclass
class EntityGraphFeatures:
    nodes: float
    edges: float
    avg_degree: float
    density: float
    dominant_degree: float
    degree_variance: float
    clustering_coeff: float
    centrality_mean: float

    def to_dict(self):
        return self.__dict__


# =========================================================
# BUILDER
# =========================================================

class EntityGraphBuilder:

    _NLP_CACHE: ClassVar[dict[str, Language]] = {}

    def __init__(self, model: str = "en_core_web_sm"):

        if model not in self._NLP_CACHE:
            try:
                self._NLP_CACHE[model] = spacy.load(model)
            except Exception:
                logger.warning("Fallback to blank spaCy model")
                self._NLP_CACHE[model] = spacy.blank("en")

        self.nlp = self._NLP_CACHE[model]

    # =====================================================
    # GRAPH BUILD (🔥 WEIGHTED)
    # =====================================================

    def build_graph(self, text: str) -> Dict[str, Dict[str, float]]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text")

        doc: Doc = self.nlp(text)

        graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for sent in doc.sents:

            ents = [
                ent.text.lower().strip()
                for ent in sent.ents
                if ent.text.strip()
            ]

            ents = list(dict.fromkeys(ents))  # unique

            for i, a in enumerate(ents):
                for b in ents[i + 1:]:

                    # 🔥 weighted co-occurrence
                    graph[a][b] += 1.0
                    graph[b][a] += 1.0

        graph = normalize_graph(graph)
        graph = to_undirected(graph)

        return graph

    # =====================================================
    # FEATURES
    # =====================================================

    def extract_features(self, graph: Dict[str, Dict[str, float]]) -> EntityGraphFeatures:

        nodes = list(graph.keys())
        n = len(nodes)

        edges = unique_edges(graph)
        e = len(edges)

        degrees = {node: len(neigh) for node, neigh in graph.items()}

        degree_vals = list(degrees.values())

        avg_degree = float(np.mean(degree_vals)) if degree_vals else 0.0
        dominant = max(degree_vals, default=0)

        density = (2 * e) / (n * (n - 1) + EPS) if n > 1 else 0.0
        variance = float(np.var(degree_vals)) if degree_vals else 0.0

        # =================================================
        # 🔥 CLUSTERING COEFFICIENT
        # =================================================
        clustering_vals = []

        for node in nodes:
            neighbors = list(graph[node].keys())
            k = len(neighbors)

            if k < 2:
                clustering_vals.append(0.0)
                continue

            links = 0
            for i in range(k):
                for j in range(i + 1, k):
                    if neighbors[j] in graph.get(neighbors[i], {}):
                        links += 1

            clustering_vals.append((2 * links) / (k * (k - 1) + EPS))

        clustering = float(np.mean(clustering_vals)) if clustering_vals else 0.0

        # =================================================
        # 🔥 CENTRALITY (degree proxy)
        # =================================================
        centrality = [deg / (n - 1 + EPS) for deg in degree_vals]
        centrality_mean = float(np.mean(centrality)) if centrality else 0.0

        return EntityGraphFeatures(
            nodes=float(n),
            edges=float(e),
            avg_degree=avg_degree,
            density=float(density),
            dominant_degree=float(dominant),
            degree_variance=variance,
            clustering_coeff=clustering,
            centrality_mean=centrality_mean,
        )


# =========================================================
# VECTOR
# =========================================================

def graph_to_vector(features: Dict[str, float]) -> np.ndarray:

    keys = [
        "nodes",
        "edges",
        "avg_degree",
        "density",
        "dominant_degree",
        "degree_variance",
        "clustering_coeff",
        "centrality_mean",
    ]

    return np.array([features.get(k, 0.0) for k in keys], dtype=np.float32)

# Alias maintained for backward compatibility with src.graph.graph_features.
ordered_entity_graph_vector = graph_to_vector

