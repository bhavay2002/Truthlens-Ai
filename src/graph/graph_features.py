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
from typing import Dict

import numpy as np

from src.graph.entity_graph import EntityGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer
from src.graph.narrative_graph_builder import NarrativeGraphBuilder


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

        features: Dict[str, float] = {}

        # -------------------------------------------------
        # Entity Graph
        # -------------------------------------------------
        if self.entity_graph_builder:

            entity_graph = self.entity_graph_builder.build_graph(text)

            entity_features_obj = (
                self.entity_graph_builder.extract_graph_features(entity_graph)
            )

            entity_features = entity_features_obj.to_dict()

            graph_metrics_obj = self.graph_analyzer.analyze(entity_graph)

            graph_metrics = graph_metrics_obj.to_dict()

            features.update(entity_features)
            features.update(graph_metrics)

        # -------------------------------------------------
        # Narrative Graph
        # -------------------------------------------------
        if self.narrative_graph_builder:

            narrative_graph = self.narrative_graph_builder.build_graph(text)

            narrative_features_obj = (
                self.narrative_graph_builder.extract_graph_features(narrative_graph)
            )

            narrative_features = narrative_features_obj.to_dict()

            features.update(narrative_features)

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

        try:
            vector = np.array(list(features.values()), dtype=np.float32)
            return vector
        except Exception as exc:
            logger.exception("Graph feature vector conversion failed")
            raise RuntimeError("Failed to convert graph features") from exc