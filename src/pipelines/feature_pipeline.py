"""
File Name: feature_pipeline.py
Module: TruthLens Pipeline - Feature Aggregation
Description:
    Runs bias, narrative, discourse, and graph feature extraction for a text
    sample and returns a unified structured feature bundle.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from src.features.bias.bias_features import BiasFeatureExtractor
from src.features.discourse.discourse_features import DiscourseFeatureExtractor
from src.features.narrative.narrative_features import NarrativeFeatureExtractor
from src.graph.entity_graph import EntityGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer


logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Extract and aggregate non-emotion analytical feature groups."""

    def __init__(self) -> None:
        self.bias_extractor = BiasFeatureExtractor()
        self.narrative_extractor = NarrativeFeatureExtractor()
        self.discourse_extractor = DiscourseFeatureExtractor()
        self.entity_graph_builder = EntityGraphBuilder()
        self.graph_analyzer = GraphAnalyzer()

        logger.info("FeaturePipeline initialized")

    def extract_features(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Extract all supported feature groups for one text sample."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            bias_features = self.bias_extractor.extract_features(text)
            narrative_features = self.narrative_extractor.extract(text)
            discourse_features = self.discourse_extractor.extract(text)

            entity_graph = self.entity_graph_builder.build_graph(text)
            graph_features = self.entity_graph_builder.extract_graph_features(
                entity_graph
            )
            graph_metrics = self.graph_analyzer.analyze(entity_graph)
        except Exception as exc:
            logger.exception("Feature extraction failed")
            raise RuntimeError("Feature pipeline execution failed") from exc

        return {
            "bias": bias_features,
            "narrative": narrative_features,
            "discourse": discourse_features,
            "graph": {**graph_features, **graph_metrics},
        }

    def extract_vector(self, text: str) -> np.ndarray:
        """Flatten numeric features across groups into a 1D vector."""
        grouped = self.extract_features(text)

        values: list[float] = []
        for group_name in ("bias", "narrative", "discourse", "graph"):
            section = grouped.get(group_name, {})
            if not isinstance(section, dict):
                continue

            for key in sorted(section.keys()):
                value = section[key]
                if isinstance(value, (int, float)):
                    values.append(float(value))

        if not values:
            raise ValueError("No numeric features were extracted")

        return np.asarray(values, dtype=np.float32)
