"""
File Name: feature_pipeline.py
Module: TruthLens Pipeline - Feature Aggregation
Description:
    Pipeline wrapper that builds structured feature bundles from the
    registered TruthLens feature extraction system.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import FeaturePipeline as CoreFeaturePipeline
from src.graph.entity_graph import EntityGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class FeatureBundle:
    """
    Structured feature bundle returned by the pipeline.
    """

    tokens: List[str]
    embedding: np.ndarray
    bias: Dict[str, Any]
    narrative: Dict[str, Any]
    discourse: Dict[str, Any]
    linguistic: Dict[str, Any]
    graph: Dict[str, Any]


class FeaturePipeline:
    """
    Unified feature extraction pipeline for TruthLens.
    """

    def __init__(self) -> None:
        self._core = CoreFeaturePipeline()
        self._core.initialize()

        self.entity_graph_builder = EntityGraphBuilder()
        self.graph_analyzer = GraphAnalyzer()

        logger.info("FeaturePipeline initialized successfully")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _split_feature_groups(flat_features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        groups: Dict[str, Dict[str, float]] = {
            "bias": {},
            "narrative": {},
            "discourse": {},
            "linguistic": {},
        }

        for key, value in flat_features.items():
            if key.startswith("bias_"):
                groups["bias"][key] = value
            elif key.startswith("narrative_"):
                groups["narrative"][key] = value
            elif key.startswith("discourse_"):
                groups["discourse"][key] = value
            else:
                groups["linguistic"][key] = value

        return groups

    def extract_features(self, text: str) -> FeatureBundle:
        """
        Extract all supported feature groups for a single text input.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        context = FeatureContext(text=text)

        try:
            flat_features = self._core.extract(context)
            groups = self._split_feature_groups(flat_features)

            embedding = None
            if isinstance(flat_features, dict):
                embedding = flat_features.get("embedding")
            if embedding is None:
                embedding = np.zeros(1, dtype=np.float32)
            else:
                embedding = np.asarray(embedding, dtype=np.float32)

            entity_graph = self.entity_graph_builder.build_graph(text)
            graph_features = self.entity_graph_builder.extract_graph_features(entity_graph)
            graph_metrics = self.graph_analyzer.analyze(entity_graph)
            graph_bundle = {**graph_features.to_dict(), **graph_metrics.to_dict()}
        except Exception as exc:
            logger.exception("Feature extraction failed")
            raise RuntimeError("Feature pipeline execution failed") from exc

        return FeatureBundle(
            tokens=context.tokens or self._tokenize(text),
            embedding=embedding,
            bias=groups["bias"],
            narrative=groups["narrative"],
            discourse=groups["discourse"],
            linguistic=groups["linguistic"],
            graph=graph_bundle,
        )

    def extract_feature_dict(self, text: str) -> Dict[str, Any]:
        bundle = self.extract_features(text)

        return {
            "tokens": bundle.tokens,
            "embedding": bundle.embedding,
            "bias": bundle.bias,
            "narrative": bundle.narrative,
            "discourse": bundle.discourse,
            "linguistic": bundle.linguistic,
            "graph": bundle.graph,
        }

    def extract_vector(self, text: str) -> np.ndarray:
        bundle = self.extract_features(text)

        values: List[float] = []

        for group in (
            bundle.bias,
            bundle.narrative,
            bundle.discourse,
            bundle.linguistic,
            bundle.graph,
        ):
            if not isinstance(group, dict):
                continue

            for key in sorted(group.keys()):
                value = group[key]
                if isinstance(value, (int, float)):
                    values.append(float(value))

        if not values:
            raise ValueError("No numeric features were extracted")

        return np.asarray(values, dtype=np.float32)
