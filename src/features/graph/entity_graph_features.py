"""
File Name: entity_graph_features.py
Module: Feature Engineering - Graph Features
Description:
    Builds an entity co-occurrence graph from text and extracts structural
    graph statistics describing relationships between entities mentioned in
    the document. The graph is constructed by connecting entities that appear
    within the same sentence or context window.

    The extracted features characterize entity interaction complexity,
    narrative centralization, and connectivity patterns which are useful
    for narrative analysis, bias detection, and discourse structure modeling.

    The module optionally uses spaCy for entity recognition. If spaCy is not
    available, a heuristic fallback based on capitalized tokens is used.

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
    Dict[str, float] representing entity graph statistics
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.graph.entity_graph import EntityGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer

logger = logging.getLogger(__name__)

def _sentence_split(text: str) -> List[str]:
    """Basic sentence splitter."""
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _heuristic_entities(sentence: str) -> List[str]:
    """
    Simple fallback entity detection using capitalized tokens.
    """
    tokens = re.findall(r"\b[A-Z][a-zA-Z]+\b", sentence)
    return list(set(tokens))


def _extract_entities(sentence: str) -> List[str]:
    """Fallback entity extraction."""
    return _heuristic_entities(sentence)


@dataclass
@register_feature
class EntityGraphFeatures(BaseFeature):
    """
    Extracts entity graph statistics.

    Output Features
    ---------------
    entity_count
    entity_edge_count
    entity_avg_degree
    entity_density
    entity_centralization
    """

    name: str = "entity_graph_features"
    description: str = "Entity interaction graph statistics"
    _builder: EntityGraphBuilder | None = field(default=None, init=False, repr=False)
    _analyzer: GraphAnalyzer | None = field(default=None, init=False, repr=False)

    def initialize(self) -> None:
        if self._builder is not None and self._analyzer is not None:
            return
        try:
            self._builder = EntityGraphBuilder()
            self._analyzer = GraphAnalyzer()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EntityGraphFeatures using heuristic fallback due to graph init failure: %s",
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
            entity_metrics = self._builder.extract_graph_features(graph).to_dict()
            graph_metrics = self._analyzer.analyze(graph).to_dict()

            features: Dict[str, float] = {
                "entity_count": float(entity_metrics.get("entity_graph_nodes", 0.0)),
                "entity_edge_count": float(entity_metrics.get("entity_graph_edges", 0.0)),
                "entity_avg_degree": float(entity_metrics.get("entity_graph_avg_degree", 0.0)),
                "entity_density": float(entity_metrics.get("entity_graph_density", 0.0)),
                "entity_centralization": float(
                    graph_metrics.get("graph_centralization", 0.0)
                ),
            }

            for key, value in entity_metrics.items():
                features[f"entity_native_{key}"] = float(value)

            for key, value in graph_metrics.items():
                features[f"entity_native_{key}"] = float(value)

            return features

        # Fallback when graph subsystem is unavailable.
        sentences = _sentence_split(context.text)
        entities = set()
        edge_count = 0

        for sent in sentences:
            sent_entities = _extract_entities(sent)
            entities.update(sent_entities)
            n = len(sent_entities)
            if n > 1:
                edge_count += (n * (n - 1)) // 2

        entity_count = len(entities)
        density = 0.0
        if entity_count > 1:
            max_edges = entity_count * (entity_count - 1) / 2.0
            density = edge_count / max_edges

        features = {
            "entity_count": float(entity_count),
            "entity_edge_count": float(edge_count),
            "entity_avg_degree": float((2.0 * edge_count) / max(entity_count, 1)),
            "entity_density": float(density),
            "entity_centralization": 0.0,
        }

        logger.debug(
            "Entity graph features extracted | nodes=%d edges=%d",
            entity_count,
            edge_count,
        )

        return features
