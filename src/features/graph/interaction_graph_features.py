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
from dataclasses import dataclass, field
from typing import Dict, List

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.graph.graph_analysis import GraphAnalyzer
from src.graph.narrative_graph_builder import NarrativeGraphBuilder

logger = logging.getLogger(__name__)

def _split_sentences(text: str) -> List[str]:
    """Basic sentence segmentation."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _heuristic_entities(sentence: str) -> List[str]:
    """Fallback entity detection using capitalized tokens."""
    tokens = re.findall(r"\b[A-Z][a-zA-Z]+\b", sentence)
    return list(set(tokens))


def _extract_entities(sentence: str) -> List[str]:
    """Fallback entity extraction."""
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
    _builder: NarrativeGraphBuilder | None = field(default=None, init=False, repr=False)
    _analyzer: GraphAnalyzer | None = field(default=None, init=False, repr=False)

    def initialize(self) -> None:
        if self._builder is not None and self._analyzer is not None:
            return
        try:
            self._builder = NarrativeGraphBuilder()
            self._analyzer = GraphAnalyzer()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "InteractionGraphFeatures using fallback due to graph init failure: %s",
                exc,
            )
            self._builder = None
            self._analyzer = None

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")
        if not context.text.strip():
            return {}

        if self._builder is not None and self._analyzer is not None:
            graph = self._builder.build_graph(context.text)
            narrative_metrics = self._builder.extract_graph_features(graph).to_dict()
            graph_metrics = self._analyzer.analyze(graph).to_dict()

            features: Dict[str, float] = {
                "interaction_node_count": float(
                    narrative_metrics.get("narrative_graph_nodes", 0.0)
                ),
                "interaction_edge_count": float(
                    narrative_metrics.get("narrative_graph_edges", 0.0)
                ),
                "interaction_avg_degree": float(
                    narrative_metrics.get("narrative_graph_avg_degree", 0.0)
                ),
                "interaction_density": float(
                    narrative_metrics.get("narrative_graph_density", 0.0)
                ),
                "interaction_clustering": float(
                    graph_metrics.get("graph_clustering_estimate", 0.0)
                ),
                "interaction_component_count": float(
                    narrative_metrics.get("narrative_graph_components", 0.0)
                ),
            }

            for key, value in narrative_metrics.items():
                features[f"interaction_native_{key}"] = float(value)

            for key, value in graph_metrics.items():
                features[f"interaction_native_{key}"] = float(value)

            return features

        # Fallback when graph subsystem is unavailable.
        sentences = _split_sentences(context.text)
        nodes = set()
        edges = set()

        for sentence in sentences:
            entities = _extract_entities(sentence)
            nodes.update(entities)
            for pair in itertools.combinations(sorted(set(entities)), 2):
                edges.add(pair)

        node_count = len(nodes)
        edge_count = len(edges)
        density = 0.0
        if node_count > 1:
            max_edges = node_count * (node_count - 1) / 2.0
            density = edge_count / max_edges

        features = {
            "interaction_node_count": float(node_count),
            "interaction_edge_count": float(edge_count),
            "interaction_avg_degree": float((2.0 * edge_count) / max(node_count, 1)),
            "interaction_density": float(density),
            "interaction_clustering": 0.0,
            "interaction_component_count": 0.0,
        }

        logger.debug(
            "Interaction graph features extracted | nodes=%d edges=%d",
            node_count,
            edge_count,
        )

        return features
