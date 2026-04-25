from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.explainability.explanation_consistency import ExplanationConsistency
from src.explainability.common_schema import AggregatedExplanation, TokenImportance

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# WEIGHTS
# =========================================================

@dataclass
class AggregationWeights:
    shap: float = 0.4
    integrated_gradients: float = 0.3
    attention: float = 0.2
    lime: float = 0.1


# =========================================================
# CORE
# =========================================================

class ExplanationAggregator:

    def __init__(
        self,
        weights: Optional[AggregationWeights] = None,
    ) -> None:

        w = weights or AggregationWeights()

        total = w.shap + w.integrated_gradients + w.attention + w.lime

        self.weights = {
            "shap": w.shap / total,
            "ig": w.integrated_gradients / total,
            "attn": w.attention / total,
            "lime": w.lime / total,
        }

        self._consistency = ExplanationConsistency()

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize(self, v):
        v = np.abs(np.asarray(v, dtype=float))
        return v / (np.sum(v) + EPS)

    # =====================================================
    # MAIN
    # =====================================================

    def aggregate(
        self,
        shap: Optional[Dict] = None,
        integrated_gradients: Optional[Dict] = None,
        attention: Optional[Dict] = None,
        lime: Optional[Dict] = None,
        graph_explanation: Optional[Dict] = None,  # 🔥 NEW
    ) -> AggregatedExplanation:

        sources = {}
        confidences = {}

        # -------------------------------------------------
        # EXTRACT
        # -------------------------------------------------
        if shap:
            sources["shap"] = dict(zip(shap.tokens, shap.importance))
            confidences["shap"] = shap.confidence or 0.5

        if integrated_gradients:
            sources["ig"] = dict(zip(integrated_gradients.tokens, integrated_gradients.importance))
            confidences["ig"] = integrated_gradients.confidence or 0.5

        if attention:
            sources["attn"] = dict(zip(attention.tokens, attention.importance))
            confidences["attn"] = attention.confidence or 0.5

        if lime:
            sources["lime"] = dict(zip(lime.tokens, lime.importance))
            confidences["lime"] = lime.confidence or 0.5

        # -------------------------------------------------
        # 🔥 GRAPH EXPLANATION EXTRACTION
        # -------------------------------------------------
        graph_node_importance = {}
        graph_confidence = 0.0

        if graph_explanation:
            graph_node_importance = graph_explanation.get("node_importance", {})
            graph_confidence = float(graph_explanation.get("overall_score", 0.5))

        if not sources and not graph_node_importance:
            raise ValueError("No sources provided")

        tokens = sorted(set().union(*[set(s.keys()) for s in sources.values()]))

        # include graph-only tokens if needed
        tokens = sorted(set(tokens) | set(graph_node_importance.keys()))

        # -------------------------------------------------
        # AGREEMENT SCORE
        # -------------------------------------------------
        agreement_score = 0.0
        try:
            res = self._consistency.compute(
                shap_importance=shap.structured if shap else None,
                integrated_gradients=integrated_gradients.structured if integrated_gradients else None,
                attention_scores=attention.structured if attention else None,
                lime_importance=[(e.token, e.importance) for e in lime.structured] if lime else None,
            )
            if res:
                agreement_score = float(np.mean(list(res.values())))
        except Exception:
            pass

        # -------------------------------------------------
        # 🔥 FUSION (WITH GRAPH)
        # -------------------------------------------------
        final_scores = []
        token_confidence = []

        for t in tokens:

            weighted_vals = []
            vals = []

            # ---------- standard explainers ----------
            for name, src in sources.items():
                if t in src:
                    val = src[t]
                    w = self.weights[name]
                    c = confidences[name]

                    weighted_vals.append(val * w * c)
                    vals.append(val)

            # ---------- 🔥 graph contribution ----------
            if t in graph_node_importance:
                graph_score = float(graph_node_importance[t])

                weighted_vals.append(graph_score * graph_confidence)
                vals.append(graph_score)

            if not weighted_vals:
                final_scores.append(0.0)
                token_confidence.append(0.0)
                continue

            score = float(np.mean(weighted_vals))

            # confidence = agreement
            conf = float(1.0 - np.std(vals)) if len(vals) > 1 else 1.0

            final_scores.append(score)
            token_confidence.append(np.clip(conf, 0.0, 1.0))

        final_scores = self._normalize(final_scores)

        # -------------------------------------------------
        # OVERALL CONFIDENCE
        # -------------------------------------------------
        overall_confidence = float(np.mean(token_confidence)) if token_confidence else 0.0

        # -------------------------------------------------
        # STRUCTURED OUTPUT
        # -------------------------------------------------
        structured = [
            TokenImportance(token=t, importance=float(s))
            for t, s in zip(tokens, final_scores)
        ]

        return AggregatedExplanation(
            tokens=tokens,
            final_token_importance=final_scores.tolist(),
            structured=structured,
            method_weights=self.weights,
            confidence_score=overall_confidence,
            agreement_score=agreement_score,
        )