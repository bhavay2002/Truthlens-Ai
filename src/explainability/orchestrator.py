"""
File Name: orchestrator.py
Module: Explainability — ExplainabilityOrchestrator
Description:
    Single owner of the full explainability lifecycle for TruthLens AI.

    This module consolidates logic that was previously split across three layers:
        - ExplainabilityLayer  (prediction_pipeline.py) — cache, rollout, propaganda,
          aggregation, consistency
        - explain_prediction_full / explain_fast  (model_explainer.py) — SHAP, LIME,
          bias, emotion
        - ExplanationAggregator  (explanation_aggregator.py) — token-level merging

    All explanation work now flows through one public entry-point::

        orchestrator.explain(text, predict_fn, ...)

    Backward-compatibility shims are preserved in both ``model_explainer.py`` and
    ``prediction_pipeline.py`` so existing call-sites continue to work without changes.

Public API
----------
ExplainabilityConfig
    Dataclass that controls which explainability components are enabled.
ExplainabilityOrchestrator
    Stateful class that holds sub-component instances and runs explanations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from src.explainability.attention_rollout import AttentionRollout
from src.explainability.attention_visualizer import AttentionVisualizer
from src.explainability.bias_explainer import explain_bias
from src.explainability.emotion_explainer import explain_emotion
from src.explainability.explanation_aggregator import (
    AggregationWeights,
    ExplanationAggregator,
)
from src.explainability.explanation_cache import ExplanationCache
from src.explainability.explanation_consistency import ExplanationConsistency
from src.explainability.explanation_metrics import ExplanationMetrics
from src.explainability.explanation_visualizer import ExplanationVisualizer
from src.explainability.lime_explainer import explain_prediction as _lime_explain
from src.explainability.propaganda_explainer import PropagandaExplainer
from src.explainability.shap_explainer import explain_text as _shap_explain
from src.explainability.token_alignment import align_tokens
from src.explainability.utils_validation import validate_tokens_scores

logger = logging.getLogger(__name__)

_SPECIAL_TOKENS = {"[CLS]", "[SEP]", "<s>", "</s>", "<pad>", "[PAD]"}


def _filter_special(items: List[Dict]) -> List[Dict]:
    return [it for it in items if str(it.get("token")) not in _SPECIAL_TOKENS]


def _run(name: str, fn: Callable) -> Any:
    """Execute an explainability component, logging failures without raising."""
    try:
        return fn()
    except Exception as exc:
        logger.warning("%s explanation failed: %s", name, exc)
        return None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ExplainabilityConfig:
    """
    Controls which sub-systems the orchestrator activates.

    Attributes
    ----------
    enabled : bool
        Master switch.  When False ``explain()`` returns an empty dict immediately.
    use_lime : bool
        Run LIME token attribution.
    use_shap : bool
        Run SHAP token attribution (expensive; disabled by default).
    use_attention_rollout : bool
        Run attention rollout when ``attentions`` tensors are provided.
    use_propaganda_explainer : bool
        Run gradient-based propaganda attribution when a propaganda model is
        available and ``input_ids`` / ``attention_mask`` are provided.
    use_bias_emotion : bool
        Run bias / emotion explanation when ``model`` and ``tokenizer`` are provided.
    use_aggregation : bool
        Merge per-method token scores into a single ranked list.
    use_consistency : bool
        Compute pairwise correlation between explanation methods.
    use_explanation_metrics : bool
        Compute faithfulness / comprehensiveness metrics (expensive).
    cache_enabled : bool
        LRU-cache explanation results keyed on article text.
    cache_max_size : int
        Maximum number of cached explanations.
    cache_dir : str or None
        Optional on-disk cache directory.
    aggregation_weights : AggregationWeights
        Per-method weights used when combining token attributions.
    """

    enabled: bool = True
    use_lime: bool = True
    use_shap: bool = False
    use_attention_rollout: bool = True
    use_propaganda_explainer: bool = True
    use_bias_emotion: bool = True
    use_aggregation: bool = True
    use_consistency: bool = True
    use_explanation_metrics: bool = False
    cache_enabled: bool = True
    cache_max_size: int = 128
    cache_dir: Optional[str] = None
    aggregation_weights: AggregationWeights = field(
        default_factory=AggregationWeights
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class ExplainabilityOrchestrator:
    """
    Single owner of the complete explainability pipeline.

    Replaces the two-layer delegation that previously existed between
    ``ExplainabilityLayer`` (in ``prediction_pipeline.py``) and
    ``explain_prediction_full`` (in ``model_explainer.py``).  All explanation
    work is now expressed as a single linear sequence within ``explain()``.

    Parameters
    ----------
    config : ExplainabilityConfig
        Feature flags and tuning parameters.
    propaganda_model : torch.nn.Module, optional
        If provided, gradient-based propaganda attribution is enabled.
    attention_model : torch.nn.Module, optional
        If provided, attention heat-map visualizations are enabled.
    """

    def __init__(
        self,
        config: ExplainabilityConfig,
        propaganda_model: Optional[torch.nn.Module] = None,
        attention_model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.config = config

        self.cache: Optional[ExplanationCache] = (
            ExplanationCache(
                max_size=config.cache_max_size,
                cache_dir=config.cache_dir,
            )
            if config.cache_enabled
            else None
        )

        self.attention_rollout = AttentionRollout()

        self.aggregator: Optional[ExplanationAggregator] = (
            ExplanationAggregator(weights=config.aggregation_weights)
            if config.use_aggregation
            else None
        )

        self.consistency: Optional[ExplanationConsistency] = (
            ExplanationConsistency() if config.use_consistency else None
        )

        self.metrics: Optional[ExplanationMetrics] = (
            ExplanationMetrics() if config.use_explanation_metrics else None
        )

        self.visualizer = ExplanationVisualizer()

        self.propaganda_explainer: Optional[PropagandaExplainer] = None
        if config.use_propaganda_explainer and propaganda_model is not None:
            self.propaganda_explainer = PropagandaExplainer(propaganda_model)

        self.attention_visualizer: Optional[AttentionVisualizer] = None
        if attention_model is not None:
            self.attention_visualizer = AttentionVisualizer(attention_model)

        logger.info("ExplainabilityOrchestrator initialized")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain(
        self,
        text: str,
        predict_fn: Callable[[str], Dict[str, Any]],
        *,
        tokens: Optional[List[str]] = None,
        attentions: Optional[List[torch.Tensor]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete explanation package for a single prediction.

        Parameters
        ----------
        text : str
            Raw article text to explain.
        predict_fn : Callable
            ``text → prediction-dict`` function (used by LIME and SHAP).
        tokens : list of str, optional
            Pre-tokenised token list aligned with the model input sequence.
        attentions : list of torch.Tensor, optional
            Per-layer attention tensors for attention rollout computation.
        input_ids : torch.Tensor, optional
            Token-id tensor forwarded to the propaganda explainer.
        attention_mask : torch.Tensor, optional
            Attention mask forwarded to the propaganda explainer.
        model : optional
            Transformer model for bias and emotion explanation.
        tokenizer : optional
            Tokenizer paired with *model*.

        Returns
        -------
        dict
            Keys: ``shap_explanation``, ``lime_explanation``,
            ``bias_explanation``, ``emotion_explanation``,
            ``attention_rollout``, ``propaganda_token_scores``,
            ``propaganda_intensity``, ``aggregated_explanation``,
            ``consistency_metrics``.  Keys are absent when the
            corresponding component was disabled or failed.
        """
        if not self.config.enabled:
            return {}

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")
        if not callable(predict_fn):
            raise TypeError("predict_fn must be callable.")

        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None:
                logger.debug("Explanation cache hit")
                return cached

        explanation: Dict[str, Any] = {}

        # ── 1. SHAP attribution ──────────────────────────────────────────
        shap_items: Optional[List[Dict]] = None
        if self.config.use_shap:
            raw = _run("SHAP", lambda: _shap_explain(predict_fn, text))
            if isinstance(raw, dict):
                items = raw.get("token_importance")
                if isinstance(items, list) and items:
                    toks, scores = [], []
                    for it in items:
                        if isinstance(it, dict):
                            t = it.get("token")
                            s = it.get("importance")
                            if isinstance(t, str) and isinstance(s, (int, float)):
                                toks.append(t)
                                scores.append(s)
                    if toks and len(toks) == len(scores):
                        validate_tokens_scores(toks, scores)
                        toks, scores = align_tokens(toks, scores)
                        shap_items = [
                            {"token": t, "importance": float(s)}
                            for t, s in zip(toks, scores)
                        ]
                        raw["token_importance"] = shap_items
            explanation["shap_explanation"] = raw

        # ── 2. LIME attribution ──────────────────────────────────────────
        lime_items: Optional[List] = None
        if self.config.use_lime:
            raw = _run("LIME", lambda: _lime_explain(predict_fn, text))
            if isinstance(raw, dict):
                items = raw.get("important_features")
                if isinstance(items, list) and items:
                    toks, scores = [], []
                    for it in items:
                        if isinstance(it, (list, tuple)) and len(it) >= 2:
                            t, s = it[0], it[1]
                            if isinstance(t, str) and isinstance(s, (int, float)):
                                toks.append(t)
                                scores.append(s)
                    if toks and len(toks) == len(scores):
                        validate_tokens_scores(toks, scores)
                        toks, scores = align_tokens(toks, scores)
                        lime_items = list(zip(toks, scores))
                        raw["important_features"] = lime_items
            explanation["lime_explanation"] = raw

        # ── 3. Bias and emotion explanation ─────────────────────────────
        if self.config.use_bias_emotion and model is not None and tokenizer is not None:
            explanation["bias_explanation"] = _run(
                "Bias", lambda: explain_bias(model, tokenizer, text)
            )
            explanation["emotion_explanation"] = _run(
                "Emotion", lambda: explain_emotion(text, model, tokenizer)
            )
        else:
            if model is None or tokenizer is None:
                logger.debug(
                    "Skipping bias/emotion explanations: model/tokenizer not provided"
                )

        # ── 4. Attention rollout ─────────────────────────────────────────
        rollout_result: Optional[Dict[str, Any]] = None
        if self.config.use_attention_rollout and attentions and tokens:
            try:
                rollout_result = self.attention_rollout.compute_rollout(
                    attentions=attentions,
                    tokens=tokens,
                )
                aligned_toks, aligned_scores = align_tokens(
                    rollout_result["tokens"],
                    rollout_result["rollout_scores"],
                )
                rollout_result["aligned_tokens"] = aligned_toks
                rollout_result["aligned_scores"] = aligned_scores
                explanation["attention_rollout"] = rollout_result
            except Exception as exc:
                logger.warning("Attention rollout failed: %s", exc)

        # ── 5. Propaganda explainer ──────────────────────────────────────
        propaganda_token_scores: Optional[Dict[str, float]] = None
        if (
            self.propaganda_explainer is not None
            and input_ids is not None
            and attention_mask is not None
            and tokens is not None
        ):
            try:
                propaganda_token_scores = self.propaganda_explainer.explain(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokens=tokens,
                )
                explanation["propaganda_token_scores"] = propaganda_token_scores
                explanation["propaganda_intensity"] = (
                    self.propaganda_explainer.propaganda_intensity(propaganda_token_scores)
                )
            except Exception as exc:
                logger.warning("Propaganda explainer failed: %s", exc)

        # ── 6. Aggregation ───────────────────────────────────────────────
        if self.aggregator is not None:
            try:
                shap_agg = shap_items if isinstance(shap_items, list) else None
                lime_agg = lime_items if isinstance(lime_items, list) else None
                attention_agg: Optional[List[Dict]] = None
                if rollout_result and "aligned_tokens" in rollout_result:
                    attention_agg = _filter_special([
                        {"token": t, "attention": s}
                        for t, s in zip(
                            rollout_result["aligned_tokens"],
                            rollout_result["aligned_scores"],
                        )
                    ])
                aggregated = self.aggregator.aggregate(
                    shap_importance=shap_agg,
                    attention_scores=attention_agg,
                    lime_importance=lime_agg,
                )
                explanation["aggregated_explanation"] = aggregated
            except Exception as exc:
                logger.warning("Explanation aggregation failed: %s", exc)

        # ── 7. Consistency metrics ───────────────────────────────────────
        if self.consistency is not None:
            try:
                shap_c = shap_items if isinstance(shap_items, list) else None
                lime_c = lime_items if isinstance(lime_items, list) else None
                attention_c: Optional[List[Dict]] = None
                if rollout_result and "aligned_tokens" in rollout_result:
                    attention_c = _filter_special([
                        {"token": t, "attention": s}
                        for t, s in zip(
                            rollout_result["aligned_tokens"],
                            rollout_result["aligned_scores"],
                        )
                    ])
                consistency_scores = self.consistency.compute(
                    shap_importance=shap_c,
                    attention_scores=attention_c,
                    lime_importance=lime_c,
                )
                explanation["consistency_metrics"] = consistency_scores
            except Exception as exc:
                logger.warning("Explanation consistency failed: %s", exc)

        logger.info("ExplainabilityOrchestrator.explain completed")

        if self.cache is not None:
            self.cache.set(text, explanation)

        return explanation

    def explain_fast(
        self,
        text: str,
        predict_fn: Callable[[str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Low-latency explanation: LIME only, no caching, no aggregation.

        Intended for production API endpoints where full explanation latency
        is unacceptable.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string.")
        if not callable(predict_fn):
            raise TypeError("predict_fn must be callable.")

        logger.info("ExplainabilityOrchestrator.explain_fast running")

        lime_explanation = _run(
            "LIME", lambda: _lime_explain(predict_fn=predict_fn, text=text)
        )

        return {
            "prediction": predict_fn(text),
            "lime_explanation": lime_explanation,
        }

    def clear_cache(self) -> None:
        """Evict all in-memory and on-disk cached explanations."""
        if self.cache is not None:
            self.cache.clear_memory()
            self.cache.clear_disk()
