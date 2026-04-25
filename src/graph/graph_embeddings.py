from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)
EPS = 1e-12

Graph = Dict[str, List[str]]


# =========================================================
# SPECTRAL
# =========================================================

def spectral_eigen_embedding(
    adjacency_matrix: np.ndarray,
    dim: int = 8,
) -> np.ndarray:

    if adjacency_matrix.size == 0:
        return np.zeros(dim, dtype=np.float32)

    try:
        eigenvalues = np.linalg.eigvalsh(adjacency_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
    except np.linalg.LinAlgError:
        logger.warning("Eigen decomposition failed")
        return np.zeros(dim, dtype=np.float32)

    if len(eigenvalues) < dim:
        eigenvalues = np.pad(eigenvalues, (0, dim - len(eigenvalues)))

    return eigenvalues[:dim].astype(np.float32)


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class GraphEmbeddingConfig:

    embedding_type: str = "hybrid"  # degree | centrality | spectral | hybrid | node2vec
    spectral_dim: int = 8
    normalize: bool = True

    # Node2Vec-lite
    walk_length: int = 10
    num_walks: int = 10
    embedding_dim: int = 16


# =========================================================
# CORE
# =========================================================

class GraphEmbeddingGenerator:

    def __init__(self, config: Optional[GraphEmbeddingConfig] = None):

        self.config = config or GraphEmbeddingConfig()

        if self.config.spectral_dim < 1:
            raise ValueError("spectral_dim must be >= 1")

        logger.info(
            "GraphEmbeddingGenerator initialized (%s)",
            self.config.embedding_type,
        )

    # =====================================================
    # UTILS
    # =====================================================

    def _validate(self, graph: Graph):
        if not isinstance(graph, dict):
            raise TypeError("graph must be dict")

    def _to_nx(self, graph: Graph) -> nx.Graph:
        G = nx.Graph()

        for node, neighbors in graph.items():
            n = node.strip().lower()
            G.add_node(n)

            for nbr in neighbors:
                if isinstance(nbr, str):
                    m = nbr.strip().lower()
                    if m and m != n:
                        G.add_edge(n, m)

        return G

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        if not self.config.normalize:
            return vec

        norm = np.linalg.norm(vec) + EPS
        return vec / norm

    # =====================================================
    # DEGREE
    # =====================================================

    def _degree(self, G: nx.Graph) -> np.ndarray:
        d = np.array([deg for _, deg in G.degree()], dtype=float)

        if d.size == 0:
            return np.zeros(4, dtype=np.float32)

        return np.array(
            [np.mean(d), np.max(d), np.min(d), np.var(d)],
            dtype=np.float32,
        )

    # =====================================================
    # CENTRALITY
    # =====================================================

    def _centrality(self, G: nx.Graph) -> np.ndarray:
        c = list(nx.degree_centrality(G).values())

        if not c:
            return np.zeros(4, dtype=np.float32)

        c = np.array(c, dtype=float)

        return np.array(
            [np.mean(c), np.max(c), np.min(c), np.var(c)],
            dtype=np.float32,
        )

    # =====================================================
    # NODE2VEC (LITE)
    # =====================================================

    def _node2vec(self, G: nx.Graph) -> np.ndarray:

        if G.number_of_nodes() == 0:
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        nodes = list(G.nodes())
        walks = []

        for _ in range(self.config.num_walks):
            for node in nodes:
                walk = [node]
                current = node

                for _ in range(self.config.walk_length - 1):
                    neighbors = list(G.neighbors(current))
                    if not neighbors:
                        break
                    current = np.random.choice(neighbors)
                    walk.append(current)

                walks.append(walk)

        vocab = {n: i for i, n in enumerate(nodes)}
        mat = np.zeros((len(nodes), len(nodes)))

        for walk in walks:
            for i in range(len(walk) - 1):
                a, b = walk[i], walk[i + 1]
                mat[vocab[a], vocab[b]] += 1

        vec = np.mean(mat, axis=0)

        if vec.size < self.config.embedding_dim:
            vec = np.pad(vec, (0, self.config.embedding_dim - vec.size))

        return vec[: self.config.embedding_dim].astype(np.float32)

    # =====================================================
    #  TEMPORAL SCALING
    # =====================================================

    def _apply_temporal_weight(
        self,
        vec: np.ndarray,
        temporal_features: Optional[Dict[str, float]],
    ) -> np.ndarray:

        if not temporal_features:
            return vec

        drift = float(temporal_features.get("narrative_drift", 0.0))

        # safety clip
        scale = 1.0 + np.clip(drift, 0.0, 1.0)

        return vec * scale

    # =====================================================
    # MAIN
    # =====================================================

    def generate_embedding(
        self,
        graph: Graph,
        *,
        temporal_features: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:

        self._validate(graph)

        G = self._to_nx(graph)

        if G.number_of_nodes() == 0:
            return np.zeros(1, dtype=np.float32)

        etype = self.config.embedding_type.lower()

        # -------------------------
        # BASE EMBEDDING
        # -------------------------
        if etype == "degree":
            vec = self._degree(G)

        elif etype == "centrality":
            vec = self._centrality(G)

        elif etype == "spectral":
            A = nx.to_numpy_array(G)
            vec = spectral_eigen_embedding(A, self.config.spectral_dim)

        elif etype == "node2vec":
            vec = self._node2vec(G)

        elif etype == "hybrid":
            deg = self._degree(G)
            cen = self._centrality(G)
            spec = spectral_eigen_embedding(
                nx.to_numpy_array(G),
                self.config.spectral_dim,
            )
            vec = np.concatenate([deg, cen, spec])

        else:
            raise ValueError(f"Unknown embedding type: {etype}")

        # -------------------------
        #  TEMPORAL ADAPTATION
        # -------------------------
        vec = self._apply_temporal_weight(vec, temporal_features)

        # -------------------------
        # NORMALIZE
        # -------------------------
        return self._normalize(vec)


# =========================================================
# API
# =========================================================

def graph_embedding_vector(
    graph: Graph,
    config: Optional[GraphEmbeddingConfig] = None,
    *,
    temporal_features: Optional[Dict[str, float]] = None,
) -> np.ndarray:

    generator = GraphEmbeddingGenerator(config)

    return generator.generate_embedding(
        graph,
        temporal_features=temporal_features,
    )