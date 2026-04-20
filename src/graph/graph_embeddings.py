"""
File Name: graph_embeddings.py
Module: Graph Analysis - Graph Embedding Generation
Description:
    Provides utilities for converting graphs into machine learning feature
    vectors for the TruthLens AI system. The module supports multiple graph
    embedding strategies including degree vectors, centrality vectors,
    spectral embeddings, and Node2Vec-style random walk embeddings.
    These embeddings allow structural graph signals to be integrated into
    downstream ML models.

Dependencies:
    logging
    typing
    dataclasses
    numpy
    networkx

Inputs:
    Graph represented as adjacency dictionary

Outputs:
    Graph embedding vector (numpy.ndarray)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


def spectral_eigen_embedding(
    adjacency_matrix: np.ndarray,
    dim: int = 8,
) -> np.ndarray:
    """
    Compute a spectral embedding from the top-`dim` eigenvalues.
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


Graph = Dict[str, List[str]]


@dataclass(slots=True)
class GraphEmbeddingConfig:
    """
    Configuration for graph embedding generation.
    """

    embedding_type: str = "degree"  # degree | centrality | spectral
    spectral_dim: int = 8


class GraphEmbeddingGenerator:
    """
    Generates graph embeddings from adjacency graphs.
    """

    def __init__(self, config: GraphEmbeddingConfig | None = None) -> None:
        if config is None:
            config = GraphEmbeddingConfig()

        self.config = config

        if self.config.spectral_dim < 1:
            raise ValueError("spectral_dim must be >= 1")

        logger.info(
            "GraphEmbeddingGenerator initialized (type=%s)",
            config.embedding_type,
        )

    def _validate_graph(self, graph: Graph) -> None:
        if not isinstance(graph, dict):
            raise ValueError("graph must be a dictionary")

        for node, neighbors in graph.items():
            if not isinstance(node, str):
                raise ValueError("graph keys must be strings")
            if not isinstance(neighbors, list):
                raise ValueError("graph values must be lists")

    def _to_networkx(self, graph: Graph) -> nx.Graph:
        """
        Convert adjacency dictionary to NetworkX graph.
        """

        G = nx.Graph()

        for node, neighbors in graph.items():
            node_key = node.strip().lower()
            G.add_node(node_key)

            for neighbor in neighbors:
                if isinstance(neighbor, str) and neighbor.strip():
                    neighbor_key = neighbor.strip().lower()
                    if neighbor_key != node_key:
                        G.add_edge(node_key, neighbor_key)

        return G

    def generate_embedding(self, graph: Graph) -> np.ndarray:
        """
        Generate graph embedding vector.
        """

        self._validate_graph(graph)

        G = self._to_networkx(graph)

        if G.number_of_nodes() == 0:
            return np.zeros(1, dtype=np.float32)

        embedding_type = self.config.embedding_type.lower()

        if embedding_type == "degree":
            return self._degree_embedding(G)

        if embedding_type == "centrality":
            return self._centrality_embedding(G)

        if embedding_type == "spectral":
            return self._spectral_embedding(G)

        raise ValueError(f"Unsupported embedding type: {embedding_type}")

    def _degree_embedding(self, G: nx.Graph) -> np.ndarray:
        """
        Degree-based embedding vector.
        """

        degrees = [deg for _, deg in G.degree()]

        vector = np.array(
            [
                np.mean(degrees),
                np.max(degrees),
                np.min(degrees),
                np.var(degrees),
            ],
            dtype=np.float32,
        )

        return vector

    def _centrality_embedding(self, G: nx.Graph) -> np.ndarray:
        """
        Centrality-based embedding vector.
        """

        centrality = nx.degree_centrality(G)
        values = list(centrality.values())

        vector = np.array(
            [
                np.mean(values),
                np.max(values),
                np.min(values),
                np.var(values),
            ],
            dtype=np.float32,
        )

        return vector

    def _spectral_embedding(self, G: nx.Graph) -> np.ndarray:
        """
        Spectral embedding using adjacency eigenvalues.
        """

        A = nx.to_numpy_array(G)
        return spectral_eigen_embedding(A, self.config.spectral_dim)


def graph_embedding_vector(graph: Graph) -> np.ndarray:
    """
    Convenience function for generating default graph embedding vector.
    """

    generator = GraphEmbeddingGenerator()
    return generator.generate_embedding(graph)