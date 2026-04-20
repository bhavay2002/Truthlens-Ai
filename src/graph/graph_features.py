"""
File Name: graph_features.py
Module: Graph Analysis - Unified Graph Feature Extraction
Description:
    Provides a unified feature extraction layer for graph-based discourse
    analysis in the TruthLens AI system. This module orchestrates multiple
    graph subsystems including entity interaction graphs, narrative transition
    graphs, and network structural metrics. It centralizes graph feature
    generation to avoid duplication across pipelines.

Dependencies:
    logging
    typing
    dataclasses
    numpy
    src.graph.entity_graph
    src.graph.narrative_graph_builder
    src.graph.graph_analysis

Inputs:
    Raw article text

Outputs:
    Unified graph feature dictionary and numerical vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.graph.entity_graph import EntityGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer
from src.graph.narrative_graph_builder import NarrativeGraphBuilder
from graph_hardening_patch import (
    merge_feature_blocks_strict,
    ordered_entity_graph_vector,
    ordered_graph_metrics_vector,
    ordered_narrative_graph_vector,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphFeatureExtractorConfig:
    """
    Configuration for graph feature extraction.
    """

    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True


class GraphFeatureExtractor:
    """
    Unified graph feature extraction pipeline.

    Responsibilities
    ----------------
    - Build entity interaction graph
    - Build narrative transition graph
    - Compute graph structural metrics
    - Merge all graph features into one dictionary
    """

    def __init__(
        self,
        config: GraphFeatureExtractorConfig | None = None,
    ) -> None:
        if config is None:
            config = GraphFeatureExtractorConfig()

        self.config = config

        self.entity_graph_builder = (
            EntityGraphBuilder() if config.enable_entity_graph else None
        )

        self.narrative_graph_builder = (
            NarrativeGraphBuilder() if config.enable_narrative_graph else None
        )

        self.graph_analyzer = GraphAnalyzer()

        logger.info("GraphFeatureExtractor initialized")

    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract unified graph features from raw text.

        Parameters
        ----------
        text : str
            Input article text.

        Returns
        -------
        Dict[str, float]
            Combined graph feature dictionary.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        entity_graph = None
        narrative_graph = None

        # -------------------------------------------------
        # Entity Graph
        # -------------------------------------------------
        if self.entity_graph_builder:
            entity_graph = self.entity_graph_builder.build_graph(text)

        # -------------------------------------------------
        # Narrative Graph
        # -------------------------------------------------
        if self.narrative_graph_builder:
            narrative_graph = self.narrative_graph_builder.build_graph(text)

        features = self.extract_from_graphs(
            entity_graph=entity_graph,
            narrative_graph=narrative_graph,
        )

        logger.debug("Graph features extracted: %d features", len(features))

        return features

    def extract_feature_vector(self, text: str) -> np.ndarray:
        """
        Extract graph features and convert to vector.

        Parameters
        ----------
        text : str

        Returns
        -------
        np.ndarray
        """

        features = self.extract_features(text)

        return self.extract_feature_vector_from_features(features)

    def extract_from_graphs(
        self,
        entity_graph: Dict[str, List[str]] | None = None,
        narrative_graph: Dict[str, List[str]] | None = None,
    ) -> Dict[str, float]:
        feature_blocks: List[Dict[str, float]] = []

        if entity_graph is not None:
            if self.entity_graph_builder is None:
                raise RuntimeError("Entity graph builder is not available")
            entity_features_obj = self.entity_graph_builder.extract_graph_features(entity_graph)
            entity_features = entity_features_obj.to_dict()
            graph_metrics_obj = self.graph_analyzer.analyze(entity_graph)
            graph_metrics = graph_metrics_obj.to_dict()
            feature_blocks.append(entity_features)
            feature_blocks.append(graph_metrics)

        if narrative_graph is not None:
            if self.narrative_graph_builder is None:
                raise RuntimeError("Narrative graph builder is not available")
            narrative_features_obj = self.narrative_graph_builder.extract_graph_features(narrative_graph)
            narrative_features = narrative_features_obj.to_dict()
            feature_blocks.append(narrative_features)

        if feature_blocks:
            return merge_feature_blocks_strict(*feature_blocks)

        return {}

    def extract_feature_vector_from_features(
        self,
        features: Dict[str, float],
    ) -> np.ndarray:
        if not isinstance(features, dict):
            raise ValueError("features must be a dictionary")

        vectors: List[np.ndarray] = []

        if self.config.enable_entity_graph:
            required_entity = {
                "entity_graph_nodes", "entity_graph_edges", "entity_graph_avg_degree",
                "entity_graph_density", "entity_graph_dominant_degree", "entity_graph_degree_variance",
                "graph_nodes", "graph_edges", "graph_avg_degree", "graph_max_degree",
                "graph_min_degree", "graph_degree_variance", "graph_density",
                "graph_centralization", "graph_clustering_estimate",
            }
            if required_entity.issubset(features.keys()):
                vectors.append(ordered_entity_graph_vector(features))
                vectors.append(ordered_graph_metrics_vector(features))
            else:
                logger.warning("Missing required entity graph feature keys; skipping entity vectors")

        if self.config.enable_narrative_graph:
            required_narrative = {
                "narrative_graph_nodes", "narrative_graph_edges", "narrative_graph_avg_degree",
                "narrative_graph_density", "narrative_graph_isolated_nodes", "narrative_graph_components",
            }
            if required_narrative.issubset(features.keys()):
                vectors.append(ordered_narrative_graph_vector(features))
            else:
                logger.warning("Missing required narrative graph feature keys; skipping narrative vector")

        if not vectors:
            return np.zeros(0, dtype=np.float32)

        try:
            vector = np.concatenate(vectors).astype(np.float32)
            return vector
        except Exception as exc:
            logger.exception("Graph feature vector conversion failed")
            raise RuntimeError("Failed to convert graph features") from exc