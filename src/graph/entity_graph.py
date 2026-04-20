"""
File Name: entity_graph.py
Module: Graph Analysis - Entity Graph Construction
Description:
    Builds entity interaction graphs from text for the TruthLens AI system.
    The module extracts named entities and constructs a co-occurrence graph
    representing relationships between them across sentences. It also derives
    structural graph features describing entity connectivity, dominance,
    and interaction density within the discourse.

Dependencies:
    logging
    typing
    dataclasses
    collections
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Entity interaction graph and graph feature dictionary
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Set, Tuple

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


def normalize_graph_adjacency(
    graph: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Normalize adjacency list nodes and neighbors.
    """
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


def to_undirected(
    graph: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """
    Convert adjacency list to undirected representation.
    """
    adj: Dict[str, Set[str]] = {node: set(neighbors) for node, neighbors in graph.items()}

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in adj:
                adj[neighbor] = set()
            adj[neighbor].add(node)

    return {node: sorted(neighbors) for node, neighbors in adj.items()}


def unique_undirected_edges(
    graph: Dict[str, List[str]]
) -> List[Tuple[str, str]]:
    """
    Return unique undirected edges from adjacency list.
    """
    edges: Set[Tuple[str, str]] = set()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            edge = (min(node, neighbor), max(node, neighbor))
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
    Return fixed-order numpy vector of entity-graph features.
    """
    return np.array(
        [float(features.get(k, 0.0)) for k in _ENTITY_GRAPH_KEYS],
        dtype=np.float32,
    )


@dataclass(slots=True)
class EntityGraphFeatures:
    """
    Structured container for entity graph features.
    """

    entity_graph_nodes: float
    entity_graph_edges: float
    entity_graph_avg_degree: float
    entity_graph_density: float
    entity_graph_dominant_degree: float
    entity_graph_degree_variance: float

    def to_dict(self) -> Dict[str, float]:
        """Convert dataclass to dictionary."""
        return {
            "entity_graph_nodes": self.entity_graph_nodes,
            "entity_graph_edges": self.entity_graph_edges,
            "entity_graph_avg_degree": self.entity_graph_avg_degree,
            "entity_graph_density": self.entity_graph_density,
            "entity_graph_dominant_degree": self.entity_graph_dominant_degree,
            "entity_graph_degree_variance": self.entity_graph_degree_variance,
        }


class EntityGraphBuilder:
    """
    Constructs and analyzes entity co-occurrence graphs from text.
    """
    _NLP_CACHE: ClassVar[dict[str, Language]] = {}

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """
        Initialize spaCy NLP pipeline for entity extraction.

        Parameters
        ----------
        spacy_model : str
            spaCy model name.
        """

        if not isinstance(spacy_model, str) or not spacy_model:
            raise ValueError("spacy_model must be a valid model name")

        try:
            if spacy_model not in self._NLP_CACHE:
                self._NLP_CACHE[spacy_model] = spacy.load(spacy_model)
            self.nlp = self._NLP_CACHE[spacy_model]
        except Exception:  # pragma: no cover
            logger.warning(
                "spaCy model not found (%s), falling back to blank English pipeline",
                spacy_model,
            )
            self.nlp = spacy.blank("en")

        logger.info("EntityGraphBuilder initialized with model: %s", spacy_model)

    def _validate_text(self, text: str) -> None:
        """Validate input text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must not be empty")

    def build_graph(self, text: str) -> Dict[str, List[str]]:
        """
        Construct an entity co-occurrence graph from text.

        Parameters
        ----------
        text : str
            Input document.

        Returns
        -------
        Dict[str, List[str]]
            Entity adjacency list graph.
        """

        self._validate_text(text)

        try:
            doc: Doc = self.nlp(text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        graph_sets: Dict[str, set[str]] = defaultdict(set)

        for sentence in doc.sents:
            entities = [
                ent.text.lower().strip()
                for ent in sentence.ents
                if ent.text and ent.text.strip()
            ]

            # remove duplicates while preserving order
            entities = list(dict.fromkeys(entities))

            if not entities:
                continue

            for entity in entities:
                graph_sets.setdefault(entity, set())

            if len(entities) < 2:
                continue

            for i, entity_a in enumerate(entities):
                for entity_b in entities[i + 1 :]:
                    graph_sets[entity_a].add(entity_b)
                    graph_sets[entity_b].add(entity_a)

        graph = {k: sorted(v) for k, v in graph_sets.items()}
        logger.debug("Entity graph built with %d nodes", len(graph))

        return graph

    def extract_graph_features(
        self, graph: Dict[str, List[str]]
    ) -> EntityGraphFeatures:
        """
        Compute structural graph metrics.

        Parameters
        ----------
        graph : Dict[str, List[str]]
            Entity adjacency graph.

        Returns
        -------
        EntityGraphFeatures
            Structured feature object.
        """

        if not isinstance(graph, dict):
            raise TypeError("graph must be a dictionary")

        adjacency = normalize_graph_adjacency(graph)
        undirected = to_undirected(adjacency)

        nodes = sorted(undirected.keys())
        node_count = len(nodes)
        edges = unique_undirected_edges(undirected)
        edge_count = len(edges)

        degrees = {node: len(undirected[node]) for node in nodes}

        dominant_degree = max(degrees.values(), default=0)

        avg_degree = float(np.mean(list(degrees.values()))) if degrees else 0.0

        density = float((2 * edge_count) / (node_count * (node_count - 1))) if node_count > 1 else 0.0

        connectivity_variance = (
            float(np.var(list(degrees.values()))) if degrees
            else 0.0
        )

        features = EntityGraphFeatures(
            entity_graph_nodes=float(node_count),
            entity_graph_edges=float(edge_count),
            entity_graph_avg_degree=float(avg_degree),
            entity_graph_density=float(density),
            entity_graph_dominant_degree=float(dominant_degree),
            entity_graph_degree_variance=float(connectivity_variance),
        )

        logger.debug("Graph features extracted: %s", features)

        return features


def entity_graph_vector(features: Dict[str, float]) -> np.ndarray:
    """
    Convert entity graph features into a numerical vector.

    Parameters
    ----------
    features : Dict[str, float]

    Returns
    -------
    np.ndarray
        Feature vector.
    """

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = ordered_entity_graph_vector(features)
        return vector
    except Exception as exc:  # pragma: no cover
        logger.exception("Entity graph vector conversion failed")
        raise RuntimeError(
            "Failed to convert entity graph features"
        ) from exc