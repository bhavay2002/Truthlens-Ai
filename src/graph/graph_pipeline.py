"""
File Name: graph_pipeline.py
Module: Graph Analysis - End-to-End Graph Pipeline
Description:
    Implements the end-to-end graph processing pipeline for the TruthLens AI
    system. The pipeline orchestrates entity graph construction, narrative
    graph construction, graph metric analysis, and unified graph feature
    extraction. It serves as the primary entry point for graph-based feature
    generation used by higher-level pipelines such as feature extraction and
    prediction pipelines.

Dependencies:
    logging
    typing
    dataclasses
    numpy
    src.graph.entity_graph
    src.graph.narrative_graph_builder
    src.graph.graph_analysis
    src.graph.graph_features
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from src.graph.entity_graph import EntityGraphBuilder
from src.graph.narrative_graph_builder import NarrativeGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer
from src.graph.graph_features import GraphFeatureExtractor, GraphFeatureExtractorConfig
from src.analysis.integration_runner import AnalysisIntegrationRunner
from graph_hardening_patch import build_pipeline_feature_extractor_config


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphPipelineConfig:
    """
    Configuration for the graph processing pipeline.
    """

    enable_entity_graph: bool = True
    enable_narrative_graph: bool = True
    return_vector: bool = True
    run_analysis_modules: bool = True


class GraphPipeline:
    """
    End-to-end graph feature pipeline.

    Processing Flow
    ---------------
    text
        ↓
    entity_graph_builder
        ↓
    narrative_graph_builder
        ↓
    graph_analyzer
        ↓
    graph_features
        ↓
    feature_vector
    """

    def __init__(self, config: GraphPipelineConfig | None = None) -> None:
        if config is None:
            config = GraphPipelineConfig()

        self.config = config

        self.entity_graph_builder = (
            EntityGraphBuilder() if config.enable_entity_graph else None
        )

        self.narrative_graph_builder = (
            NarrativeGraphBuilder() if config.enable_narrative_graph else None
        )

        self.graph_analyzer = GraphAnalyzer()
        extractor_cfg = build_pipeline_feature_extractor_config(
            enable_entity_graph=config.enable_entity_graph,
            enable_narrative_graph=config.enable_narrative_graph,
        )
        self.graph_feature_extractor = GraphFeatureExtractor(
            GraphFeatureExtractorConfig(
                enable_entity_graph=extractor_cfg.enable_entity_graph,
                enable_narrative_graph=extractor_cfg.enable_narrative_graph,
            )
        )
        self.analysis_runner = AnalysisIntegrationRunner() if config.run_analysis_modules else None

        logger.info("GraphPipeline initialized")

    def _validate_text(self, text: str) -> None:
        """Validate input text."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must be non-empty")

    def run(self, text: str) -> Dict[str, Any]:
        """
        Execute the graph processing pipeline.

        Parameters
        ----------
        text : str

        Returns
        -------
        Dict[str, Any]
            Contains graphs, features, and optionally feature vector.
        """

        self._validate_text(text)

        results: Dict[str, Any] = {}
        entity_graph = None
        narrative_graph = None

        # -------------------------------------------
        # Entity Graph
        # -------------------------------------------
        if self.entity_graph_builder:
            entity_graph = self.entity_graph_builder.build_graph(text)
            results["entity_graph"] = entity_graph

        # -------------------------------------------
        # Narrative Graph
        # -------------------------------------------
        if self.narrative_graph_builder:
            narrative_graph = self.narrative_graph_builder.build_graph(text)
            results["narrative_graph"] = narrative_graph

        if entity_graph is not None:
            results["entity_graph_metrics"] = self.graph_analyzer.analyze(entity_graph).to_dict()
        if narrative_graph is not None:
            results["narrative_graph_metrics"] = self.graph_analyzer.analyze(narrative_graph).to_dict()

        # -------------------------------------------
        # Unified Graph Features
        # -------------------------------------------
        features = self.graph_feature_extractor.extract_from_graphs(
            entity_graph=entity_graph,
            narrative_graph=narrative_graph,
        )

        results["graph_features"] = features
        if self.analysis_runner is not None:
            results["analysis_modules"] = self.analysis_runner.analyze_text(text)

        # -------------------------------------------
        # Feature Vector
        # -------------------------------------------
        if self.config.return_vector:
            try:
                vector = self.graph_feature_extractor.extract_feature_vector_from_features(features)
                results["graph_feature_vector"] = vector
            except Exception as exc:
                logger.exception("Failed to build graph feature vector")
                raise RuntimeError("Graph feature vector creation failed") from exc

        logger.debug(
            "GraphPipeline completed: %d features extracted",
            len(features),
        )

        return results
