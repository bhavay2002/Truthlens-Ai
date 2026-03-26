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
    collections
    numpy
    spacy

Inputs:
    Raw text string

Outputs:
    Entity interaction graph and graph feature dictionary
"""

import logging
from collections import defaultdict, Counter
from typing import Dict, List

import numpy as np
import spacy


logger = logging.getLogger(__name__)


class EntityGraphBuilder:
    """
    Constructs and analyzes entity co-occurrence graphs.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        """Initialize NLP pipeline for entity extraction."""

        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        logger.info("EntityGraphBuilder initialized")

    def build_graph(self, text: str) -> Dict[str, List[str]]:
        """Construct an entity co-occurrence graph from text."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        try:
            doc = self.nlp(text)
        except Exception as exc:
            logger.exception("spaCy processing failed")
            raise RuntimeError("Text processing failed") from exc

        graph: Dict[str, List[str]] = defaultdict(list)

        for sentence in doc.sents:

            entities = [ent.text.lower() for ent in sentence.ents]

            if len(entities) < 2:
                continue

            for i, entity_a in enumerate(entities):
                for j, entity_b in enumerate(entities):
                    if i == j:
                        continue

                    graph[entity_a].append(entity_b)

        return dict(graph)

    def extract_graph_features(self, graph: Dict[str, List[str]]) -> Dict[str, float]:
        """Compute structural metrics from the entity graph."""

        if not isinstance(graph, dict):
            raise ValueError("graph must be a dictionary")

        node_count = len(graph)

        edge_count = sum(len(neighbors) for neighbors in graph.values())

        degree_counts = Counter()

        for entity, neighbors in graph.items():
            degree_counts[entity] = len(neighbors)

        dominant_degree = degree_counts.most_common(1)[0][1] if degree_counts else 0

        avg_degree = edge_count / max(node_count, 1)

        density = edge_count / max(node_count * (node_count - 1), 1)

        connectivity_variance = float(np.var(list(degree_counts.values()))) if degree_counts else 0.0

        features = {
            "entity_graph_nodes": float(node_count),
            "entity_graph_edges": float(edge_count),
            "entity_graph_avg_degree": float(avg_degree),
            "entity_graph_density": float(density),
            "entity_graph_dominant_degree": float(dominant_degree),
            "entity_graph_degree_variance": float(connectivity_variance),
        }

        return features


def entity_graph_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert entity graph features into a numerical vector."""

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary")

    try:
        vector = np.array(list(features.values()), dtype=np.float32)
        return vector
    except Exception as exc:
        logger.exception("Entity graph vector conversion failed")
        raise RuntimeError("Failed to convert entity graph features") from exc