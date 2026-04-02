"""
File Name: interaction_graph_features.py
Module: Feature Engineering - Graph Features
Description:
    Builds an interaction graph capturing relationships between entities,
    actors, and referenced subjects appearing within the same contextual
    windows (sentences or paragraphs). The graph models interaction
    structures in narrative discourse and extracts structural metrics
    describing connectivity, clustering, and interaction complexity.

    These features are useful for analyzing narrative propagation,
    actor interaction dynamics, and discourse structure.

Dependencies:
    dataclasses
    typing
    logging
    re
    itertools
    networkx (optional)
    spacy (optional)

Inputs:
    FeatureContext containing input text

Outputs:
    Dict[str, float] containing interaction graph metrics
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from typing import Dict, List

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except Exception:  # noqa: BLE001
    NETWORKX_AVAILABLE = False
    logger.warning("networkx not available. Interaction graph metrics limited.")

try:
    import spacy

    _NLP = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:  # noqa: BLE001
    _NLP = None
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Using heuristic entity detection.")


def _split_sentences(text: str) -> List[str]:
    """Basic sentence segmentation."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _heuristic_entities(sentence: str) -> List[str]:
    """Fallback entity detection using capitalized tokens."""
    tokens = re.findall(r"\b[A-Z][a-zA-Z]+\b", sentence)
    return list(set(tokens))


def _extract_entities(sentence: str) -> List[str]:
    """Extract named entities using spaCy if available."""
    if SPACY_AVAILABLE:
        doc = _NLP(sentence)
        return list({ent.text for ent in doc.ents})
    return _heuristic_entities(sentence)


@dataclass
@register_feature
class InteractionGraphFeatures(BaseFeature):
    """
    Extracts structural features from entity interaction graphs.

    Output Features
    ---------------
    interaction_node_count
    interaction_edge_count
    interaction_avg_degree
    interaction_density
    interaction_clustering
    interaction_component_count
    """

    name: str = "interaction_graph_features"
    description: str = "Graph-based interaction structure indicators"

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        sentences = _split_sentences(context.text)

        nodes = set()
        edges = []

        for sentence in sentences:
            entities = _extract_entities(sentence)

            nodes.update(entities)

            for pair in itertools.combinations(entities, 2):
                edges.append(pair)

        node_count = len(nodes)
        edge_count = len(edges)

        if not NETWORKX_AVAILABLE or node_count == 0:
            return {
                "interaction_node_count": float(node_count),
                "interaction_edge_count": float(edge_count),
                "interaction_avg_degree": 0.0,
                "interaction_density": 0.0,
                "interaction_clustering": 0.0,
                "interaction_component_count": 0.0,
            }

        G = nx.Graph()

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        degrees = [deg for _, deg in G.degree()]

        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0

        density = nx.density(G)

        clustering = nx.average_clustering(G) if node_count > 1 else 0.0

        component_count = nx.number_connected_components(G)

        features: Dict[str, float] = {
            "interaction_node_count": float(node_count),
            "interaction_edge_count": float(edge_count),
            "interaction_avg_degree": float(avg_degree),
            "interaction_density": float(density),
            "interaction_clustering": float(clustering),
            "interaction_component_count": float(component_count),
        }

        logger.debug(
            "Interaction graph features extracted | nodes=%d edges=%d",
            node_count,
            edge_count,
        )

        return features