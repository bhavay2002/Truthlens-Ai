# src/features/interaction_graph_features.py

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# ---------------------------------------------------------
# Fallback utilities
# ---------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _heuristic_entities(sentence: str) -> List[str]:
    return list(set(re.findall(r"\b[A-Z][a-zA-Z]+\b", sentence)))


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class InteractionGraphFeatures(BaseFeature):

    name: str = "interaction_graph_features"
    group: str = "graph"
    description: str = "Normalized interaction graph features"

    _builder: object | None = field(default=None, init=False)
    _analyzer: object | None = field(default=None, init=False)

    # -----------------------------------------------------

    def initialize(self) -> None:
        if self._builder is not None:
            return
        try:
            from src.graph.narrative_graph_builder import NarrativeGraphBuilder
            from src.graph.graph_analysis import GraphAnalyzer

            self._builder = NarrativeGraphBuilder()
            self._analyzer = GraphAnalyzer()

        except Exception as e:
            logger.warning("Graph fallback mode: %s", e)
            self._builder = None
            self._analyzer = None

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        self.initialize()

        # =====================================================
        # GRAPH BUILD
        # =====================================================

        if self._builder and self._analyzer:

            graph = self._builder.build_graph(text)

            metrics = self._builder.extract_graph_features(graph).to_dict()
            gmetrics = self._analyzer.analyze(graph).to_dict()

            nodes = float(metrics.get("narrative_graph_nodes", 0.0))
            edges = float(metrics.get("narrative_graph_edges", 0.0))
            components = float(metrics.get("narrative_graph_components", 1.0))
            clustering = float(gmetrics.get("graph_clustering_estimate", 0.0))

        else:
            # fallback
            sentences = _split_sentences(text)

            nodes_set = set()
            edges_set = set()

            for s in sentences:
                ents = _heuristic_entities(s)
                nodes_set.update(ents)

                for pair in itertools.combinations(sorted(set(ents)), 2):
                    edges_set.add(pair)

            nodes = float(len(nodes_set))
            edges = float(len(edges_set))
            components = 1.0
            clustering = 0.0

        # =====================================================
        # NORMALIZATION
        # =====================================================

        max_edges = nodes * (nodes - 1) / 2.0 if nodes > 1 else 1.0

        density = edges / (max_edges + EPS)
        sparsity = 1.0 - density

        # normalized degree
        avg_degree = (2.0 * edges) / (nodes + EPS)
        degree_norm = avg_degree / (nodes + EPS)

        # component ratio
        component_ratio = components / (nodes + EPS)

        # =====================================================
        # ENTROPY (CRITICAL)
        # =====================================================

        probs = np.array([density, sparsity, clustering], dtype=np.float32)

        if probs.sum() > 0:
            probs = probs / (probs.sum() + EPS)
            entropy = -np.sum(probs * np.log(probs + EPS))
        else:
            entropy = 0.0

        # =====================================================
        # INTENSITY
        # =====================================================

        intensity = float(np.linalg.norm([density, degree_norm, clustering]))

        # =====================================================
        # OUTPUT
        # =====================================================

        return {
            "interaction_nodes_norm": self._safe(np.log1p(nodes) / 10.0),
            "interaction_edges_norm": self._safe(np.log1p(edges) / 10.0),

            "interaction_density": self._safe(density),
            "interaction_sparsity": self._safe(sparsity),

            "interaction_degree_norm": self._safe(degree_norm),
            "interaction_clustering": self._safe(clustering),

            "interaction_component_ratio": self._safe(component_ratio),

            "interaction_entropy": self._safe(entropy),
            "interaction_intensity": self._safe(intensity),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))