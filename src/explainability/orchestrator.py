from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from src.explainability.attention_rollout import AttentionRollout
from src.explainability.bias_explainer import explain_bias
from src.explainability.emotion_explainer import explain_emotion
from src.explainability.explanation_aggregator import (
    AggregationWeights,
    ExplanationAggregator,
)
from src.explainability.explanation_cache import ExplanationCache
from src.explainability.explanation_consistency import ExplanationConsistency
from src.explainability.explanation_metrics import ExplanationMetrics
from src.explainability.explanation_monitor import ExplanationMonitor

from src.explainability.lime_explainer import explain_prediction as _lime_explain
from src.explainability.propaganda_explainer import explain_propaganda as _propaganda_explain
from src.explainability.shap_explainer import explain_text as _shap_explain

from src.graph.graph_explainer import GraphExplainer


# =========================================================
# HELPERS
# =========================================================

def _make_batch_predict_fn(predict_fn: Callable) -> Callable:
    """Wrap a single-text predict_fn into the batched List[str] → List[Dict] signature
    expected by ExplanationMetrics."""
    def _batch(texts: List[str]) -> List[Dict]:
        return [predict_fn(t) for t in texts]
    return _batch

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class ExplainabilityConfig:
    enabled: bool = True
    use_lime: bool = True
    use_shap: bool = False
    use_attention_rollout: bool = True
    use_bias_emotion: bool = True
    use_propaganda_explainer: bool = False
    use_aggregation: bool = True
    use_consistency: bool = True
    use_explanation_metrics: bool = True

    cache_enabled: bool = True
    cache_max_size: int = 128
    cache_dir: Optional[str] = None

    aggregation_weights: AggregationWeights = field(
        default_factory=AggregationWeights
    )


# =========================================================
# ORCHESTRATOR
# =========================================================

class ExplainabilityOrchestrator:

    def __init__(self, config: ExplainabilityConfig):
        self.config = config

        self.cache = (
            ExplanationCache(
                max_size=config.cache_max_size,
                cache_dir=config.cache_dir,
            )
            if config.cache_enabled
            else None
        )

        self.rollout = AttentionRollout()

        self.aggregator = (
            ExplanationAggregator(config.aggregation_weights)
            if config.use_aggregation
            else None
        )

        self.consistency = (
            ExplanationConsistency()
            if config.use_consistency
            else None
        )

        self.metrics = (
            ExplanationMetrics()
            if config.use_explanation_metrics
            else None
        )

        #  NEW
        self.monitor = ExplanationMonitor()
        self.graph_explainer = GraphExplainer()

        logger.info("ExplainabilityOrchestrator initialized")

    # =====================================================
    # SAFE EXECUTION
    # =====================================================

    def _run(self, name: str, fn: Callable):
        start = time.time()
        try:
            result = fn()
            latency = (time.time() - start) * 1000
            return result, latency, True
        except Exception as e:
            logger.warning("%s failed: %s", name, e)
            latency = (time.time() - start) * 1000
            return None, latency, False

    # =====================================================
    # MAIN
    # =====================================================

    def explain(
        self,
        text: str,
        predict_fn: Callable[[str], Dict[str, Any]],
        *,
        tokens: Optional[List[str]] = None,
        attentions: Optional[List[torch.Tensor]] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:

        if not self.config.enabled:
            return {}

        if self.cache:
            cached = self.cache.get(text)
            if cached:
                return cached

        metadata = {
            "pipeline_version": "v4",
            "latency_ms": {},
            "modules": {},
        }

        explanation: Dict[str, Any] = {}

        # =================================================
        # SHAP
        # =================================================
        shap_out = None
        if self.config.use_shap:
            shap_out, t, ok = self._run("shap", lambda: _shap_explain(predict_fn, text))
            metadata["latency_ms"]["shap"] = t
            metadata["modules"]["shap"] = ok
            explanation["shap_explanation"] = shap_out

        # =================================================
        # LIME
        # =================================================
        lime_out = None
        if self.config.use_lime:
            lime_out, t, ok = self._run("lime", lambda: _lime_explain(predict_fn, text))
            metadata["latency_ms"]["lime"] = t
            metadata["modules"]["lime"] = ok
            explanation["lime_explanation"] = lime_out

        # =================================================
        # PROPAGANDA
        # =================================================
        propaganda_out = None
        if self.config.use_propaganda_explainer:
            propaganda_out, t, ok = self._run("propaganda", lambda: _propaganda_explain(text))
            metadata["latency_ms"]["propaganda"] = t
            metadata["modules"]["propaganda"] = ok
            explanation["propaganda_explanation"] = propaganda_out

        # =================================================
        # BIAS + EMOTION
        # =================================================
        if self.config.use_bias_emotion and model and tokenizer:
            bias, t1, ok1 = self._run("bias", lambda: explain_bias(model, tokenizer, text))
            emo, t2, ok2 = self._run("emotion", lambda: explain_emotion(text, model, tokenizer))

            metadata["latency_ms"]["bias"] = t1
            metadata["latency_ms"]["emotion"] = t2
            metadata["modules"]["bias"] = ok1
            metadata["modules"]["emotion"] = ok2

            explanation["bias_explanation"] = bias
            explanation["emotion_explanation"] = emo

        # =================================================
        # ATTENTION
        # =================================================
        attention_out = None
        if self.config.use_attention_rollout and tokens and attentions:
            attention_out, t, ok = self._run(
                "attention",
                lambda: self.rollout.compute_rollout(attentions, tokens),
            )
            metadata["latency_ms"]["attention"] = t
            metadata["modules"]["attention"] = ok
            explanation["attention_explanation"] = attention_out

        # =================================================
        #  GRAPH EXPLANATION
        # =================================================
        graph_expl = None
        graph_expl, t, ok = self._run(
            "graph_explainer",
            lambda: self.graph_explainer.explain_from_text(text),
        )
        metadata["latency_ms"]["graph_explainer"] = t
        metadata["modules"]["graph_explainer"] = ok
        explanation["graph_explanation"] = graph_expl

        # =================================================
        # AGGREGATION
        # =================================================
        if self.aggregator:
            agg, t, ok = self._run(
                "aggregation",
                lambda: self.aggregator.aggregate(
                    shap=shap_out,
                    integrated_gradients=None,
                    attention=attention_out,
                    lime=lime_out,
                    graph_explanation=graph_expl,  # 🔥 integrated
                ),
            )

            metadata["latency_ms"]["aggregation"] = t
            metadata["modules"]["aggregation"] = ok

            explanation["aggregated_explanation"] = agg

            #  MONITORING
            if agg:
                scores = agg.final_token_importance
                self.monitor.update(scores)
                explanation["monitoring"] = self.monitor.summary()

        # =================================================
        # CONSISTENCY
        # =================================================
        if self.consistency:
            def _to_dict_list(structured):
                if not structured:
                    return None
                return [{"token": e.token, "importance": e.importance} for e in structured]

            cons, t, ok = self._run(
                "consistency",
                lambda: self.consistency.compute(
                    shap_importance=_to_dict_list(shap_out.structured) if shap_out else None,
                    integrated_gradients=None,
                    attention_scores=_to_dict_list(attention_out.structured) if attention_out else None,
                    lime_importance=[(e.token, e.importance) for e in lime_out.structured] if lime_out else None,
                ),
            )
            metadata["latency_ms"]["consistency"] = t
            metadata["modules"]["consistency"] = ok
            explanation["consistency_metrics"] = cons

        # =================================================
        # METRICS
        # =================================================
        if self.metrics and "aggregated_explanation" in explanation:
            try:
                agg = explanation["aggregated_explanation"]

                batch_predict_fn = _make_batch_predict_fn(predict_fn)

                metrics = self.metrics.evaluate(
                    agg.tokens,
                    agg.final_token_importance,
                    batch_predict_fn,
                )

                explanation["explanation_metrics"] = metrics
                explanation["explanation_quality_score"] = metrics.get("overall_score")

            except Exception as e:
                logger.warning("metrics failed: %s", e)

        # =================================================
        # METADATA
        # =================================================
        explanation["metadata"] = metadata

        if self.cache:
            self.cache.set(text, explanation)

        return explanation

    # =====================================================
    # FAST MODE
    # =====================================================

    def explain_fast(self, text: str, predict_fn):

        lime, t, ok = self._run("lime", lambda: _lime_explain(predict_fn, text))

        return {
            "prediction": predict_fn(text),
            "lime_explanation": lime,
            "metadata": {
                "mode": "fast",
                "latency_ms": t,
                "lime_success": ok,
            },
        }