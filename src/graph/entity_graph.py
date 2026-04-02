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
from typing import Dict, List, Set

import numpy as np
import spacy
from spacy.language import Language
from spacy.tokens import Doc


logger = logging.getLogger(__name__)


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
            self.nlp: Language = spacy.load(spacy_model)
        except Exception as exc:  # pragma: no cover
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

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

        graph: Dict[str, List[str]] = defaultdict(list)

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
                graph.setdefault(entity, [])

            if len(entities) < 2:
                continue

            for i, entity_a in enumerate(entities):
                for entity_b in entities[i + 1 :]:
                    graph[entity_a].append(entity_b)
                    graph[entity_b].append(entity_a)

        logger.debug("Entity graph built with %d nodes", len(graph))

        return dict(graph)

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

        adjacency: Dict[str, Set[str]] = {}
        all_nodes: Set[str] = set()

        for entity, neighbors in graph.items():
            if not isinstance(entity, str):
                raise ValueError("graph keys must be strings")
            if not isinstance(neighbors, list):
                raise ValueError("graph values must be lists")

            entity_key = entity.strip().lower()

            neighbor_set = {
                str(neighbor).strip().lower()
                for neighbor in neighbors
                if isinstance(neighbor, str)
                and neighbor.strip()
                and str(neighbor).strip().lower() != entity_key
            }

            adjacency[entity_key] = neighbor_set
            all_nodes.add(entity_key)
            all_nodes.update(neighbor_set)

        for node in all_nodes:
            adjacency.setdefault(node, set())

        edge_pairs = {
            (source, target)
            for source, neighbors in adjacency.items()
            for target in neighbors
            if source != target
        }

        node_count = len(all_nodes)
        edge_count = len(edge_pairs)

        degree_counts = Counter(
            {
                node: len(adjacency[node])
                for node in all_nodes
            }
        )

        dominant_degree = degree_counts.most_common(1)[0][1] if degree_counts else 0

        avg_degree = edge_count / max(node_count, 1)

        density = edge_count / max(node_count * (node_count - 1), 1)

        connectivity_variance = (
            float(np.var(list(degree_counts.values())))
            if degree_counts
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
        vector = np.array(list(features.values()), dtype=np.float32)
        return vector
    except Exception as exc:  # pragma: no cover
        logger.exception("Entity graph vector conversion failed")
        raise RuntimeError(
            "Failed to convert entity graph features"
        ) from exc