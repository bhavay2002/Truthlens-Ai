# src/features/entity_graph_features.py

from __future__ import annotations

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

def _sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


def _heuristic_entities(sentence: str) -> List[str]:
    return list(set(re.findall(r"\b[A-Z][a-zA-Z]+\b", sentence)))


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class EntityGraphFeatures(BaseFeature):

    name: str = "entity_graph_features"
    group: str = "graph"
    description: str = "Normalized entity graph structural features"

    _builder: object | None = field(default=None, init=False)
    _analyzer: object | None = field(default=None, init=False)

    # -----------------------------------------------------

    def initialize(self) -> None:
        if self._builder is not None:
            return
        try:
            from src.graph.entity_graph import EntityGraphBuilder
            from src.graph.graph_analysis import GraphAnalyzer

            self._builder = EntityGraphBuilder()
            self._analyzer = GraphAnalyzer()

        except Exception as e:
            logger.warning("Graph system unavailable → fallback: %s", e)
            self._builder = None
            self._analyzer = None

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        self.initialize()

        # =====================================================
        # GRAPH PIPELINE
        # =====================================================

        if self._builder and self._analyzer:

            graph = self._builder.build_graph(text)

            metrics = self._builder.extract_graph_features(graph).to_dict()
            gmetrics = self._analyzer.analyze(graph).to_dict()

            nodes = float(metrics.get("entity_graph_nodes", 0.0))
            edges = float(metrics.get("entity_graph_edges", 0.0))

        else:
            # fallback
            sentences = _sentence_split(text)

            entities = set()
            edges = 0

            for s in sentences:
                ents = _heuristic_entities(s)
                entities.update(ents)

                n = len(ents)
                if n > 1:
                    edges += (n * (n - 1)) // 2

            nodes = float(len(entities))
            gmetrics = {}

        # =====================================================
        # NORMALIZATION
        # =====================================================

        max_edges = nodes * (nodes - 1) / 2.0 if nodes > 1 else 1.0

        density = edges / (max_edges + EPS)

        # normalized degree
        avg_degree = (2.0 * edges) / (nodes + EPS)
        degree_norm = avg_degree / (nodes + EPS)

        # sparsity
        sparsity = 1.0 - density

        # =====================================================
        # ENTROPY (CRITICAL)
        # =====================================================

        probs = np.array([density, sparsity], dtype=np.float32)

        if probs.sum() > 0:
            probs = probs / (probs.sum() + EPS)
            entropy = -np.sum(probs * np.log(probs + EPS))
        else:
            entropy = 0.0

        # =====================================================
        # INTENSITY
        # =====================================================

        intensity = float(np.linalg.norm([density, degree_norm]))

        # =====================================================
        # OUTPUT
        # =====================================================

        # Audit fix §1.1 — emit raw log-magnitudes and let
        # FeatureScalingPipeline learn the normalisation. The previous
        # ``/ 10.0`` divisor implicitly assumed a corpus where
        # ``log1p(nodes) ~ 10`` (≈ 22k nodes), saturating short docs at
        # near-zero and clipping long docs at 1.0.

        return {
            "graph_nodes_log": self._safe_unbounded(float(np.log1p(nodes))),
            "graph_edges_log": self._safe_unbounded(float(np.log1p(edges))),

            "graph_density": self._safe(density),
            "graph_sparsity": self._safe(sparsity),

            "graph_degree_norm": self._safe(degree_norm),

            "graph_entropy": self._safe(entropy),
            "graph_intensity": self._safe(intensity),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    def _safe_unbounded(self, v: float) -> float:
        """Drop NaN / negative values without applying an upper clip.

        Audit fix §1.1 — raw log-magnitudes flow through to the
        :class:`FeatureScalingPipeline` for corpus-aware normalisation.
        """
        if not np.isfinite(v) or v < 0:
            return 0.0
        return float(v)