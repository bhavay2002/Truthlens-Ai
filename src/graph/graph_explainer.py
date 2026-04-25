from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.graph.graph_analysis import compute_graph_metrics
from src.graph.temporal_graph import TemporalGraphAnalyzer

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# OUTPUT STRUCTURE
# =========================================================

@dataclass
class GraphExplanation:

    node_importance: Dict[str, float]
    edge_importance: Dict[str, float]
    temporal_importance: Dict[str, float]

    overall_score: float

    def to_dict(self) -> Dict:
        return {
            "node_importance": self.node_importance,
            "edge_importance": self.edge_importance,
            "temporal_importance": self.temporal_importance,
            "overall_score": self.overall_score,
        }


# =========================================================
# CORE EXPLAINER
# =========================================================

class GraphExplainer:

    def __init__(self):
        self.temporal = TemporalGraphAnalyzer()
        logger.info("GraphExplainer initialized")

    # =====================================================
    # NODE IMPORTANCE
    # =====================================================

    def _node_importance(
        self,
        graph: Dict[str, List[str]],
    ) -> Dict[str, float]:

        importance = {}

        if not graph:
            return importance

        total_nodes = len(graph)

        for node, neighbors in graph.items():

            degree = len(neighbors)

            # normalized degree centrality
            score = degree / (total_nodes + EPS)

            importance[node] = float(score)

        # normalize
        total = sum(importance.values()) + EPS

        return {k: v / total for k, v in importance.items()}

    # =====================================================
    # EDGE IMPORTANCE
    # =====================================================

    def _edge_importance(
        self,
        graph: Dict[str, List[str]],
    ) -> Dict[str, float]:

        edge_scores: Dict[Tuple[str, str], float] = {}

        for src, nbrs in graph.items():
            for tgt in nbrs:

                if src == tgt:
                    continue

                edge = tuple(sorted((src, tgt)))

                edge_scores[edge] = edge_scores.get(edge, 0.0) + 1.0

        if not edge_scores:
            return {}

        total = sum(edge_scores.values()) + EPS

        return {
            f"{a}->{b}": float(v / total)
            for (a, b), v in edge_scores.items()
        }

    # =====================================================
    # TEMPORAL IMPORTANCE
    # =====================================================

    def _temporal_importance(
        self,
        text: Optional[str],
    ) -> Dict[str, float]:

        if not text:
            return {}

        features = self.temporal.analyze(text).to_dict()

        # normalize temporal features
        vals = np.array(list(features.values()), dtype=float)

        if np.sum(vals) == 0:
            return features

        vals = vals / (np.sum(vals) + EPS)

        return dict(zip(features.keys(), vals.tolist()))

    # =====================================================
    # OVERALL SCORE
    # =====================================================

    def _overall_score(
        self,
        node_imp: Dict[str, float],
        edge_imp: Dict[str, float],
        temporal_imp: Dict[str, float],
    ) -> float:

        node_score = np.mean(list(node_imp.values())) if node_imp else 0.0
        edge_score = np.mean(list(edge_imp.values())) if edge_imp else 0.0
        temp_score = np.mean(list(temporal_imp.values())) if temporal_imp else 0.0

        return float(
            0.4 * node_score +
            0.3 * edge_score +
            0.3 * temp_score
        )

    # =====================================================
    # PUBLIC API
    # =====================================================

    def explain(
        self,
        *,
        entity_graph: Optional[Dict[str, List[str]]] = None,
        narrative_graph: Optional[Dict[str, List[str]]] = None,
        text: Optional[str] = None,
    ) -> GraphExplanation:

        graph = entity_graph or narrative_graph or {}

        node_imp = self._node_importance(graph)
        edge_imp = self._edge_importance(graph)
        temporal_imp = self._temporal_importance(text)

        score = self._overall_score(node_imp, edge_imp, temporal_imp)

        return GraphExplanation(
            node_importance=node_imp,
            edge_importance=edge_imp,
            temporal_importance=temporal_imp,
            overall_score=score,
        )

    # =====================================================
    # TEXT-ONLY SHORTCUT
    # =====================================================

    def explain_from_text(self, text: str) -> Dict:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Invalid text")

        temporal_imp = self._temporal_importance(text)

        return {
            "temporal_importance": temporal_imp,
            "overall_score": float(np.mean(list(temporal_imp.values())) if temporal_imp else 0.0),
        }