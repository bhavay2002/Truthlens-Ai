"""
old File Name: model_explainer.py
src\explainability\explainability_pipeline.py
Module: Explainability - Unified Pipeline (FINAL)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict

from src.explainability.orchestrator import (
    ExplainabilityConfig,
    ExplainabilityOrchestrator,
)

logger = logging.getLogger(__name__)

PredictionFn = Callable[[str], Dict[str, Any]]


# =========================================================
# 🔥 FINAL RESULT WRAPPER
# =========================================================

class ExplainabilityResult(BaseModel):
    """
    Final unified explainability output.
    """

    model_config = ConfigDict(extra="forbid")

    prediction: Dict[str, Any]

    shap_explanation: Optional[Any] = None
    lime_explanation: Optional[Any] = None
    attention_explanation: Optional[Any] = None

    bias_explanation: Optional[Any] = None
    emotion_explanation: Optional[Any] = None

    aggregated_explanation: Optional[Any] = None

    consistency_metrics: Optional[Dict[str, float]] = None
    explanation_metrics: Optional[Dict[str, float]] = None

    monitoring: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# =========================================================
# CORE PIPELINE
# =========================================================

def run_explainability_pipeline(
    text: str,
    predict_fn: PredictionFn,
    *,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    tokens: Optional[list[str]] = None,
    attentions: Optional[Any] = None,
    config: Optional[ExplainabilityConfig] = None,
) -> ExplainabilityResult:

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be non-empty")

    config = config or ExplainabilityConfig()

    orchestrator = ExplainabilityOrchestrator(config=config)

    logger.info("Running explainability pipeline")

    prediction = predict_fn(text)

    explanation = orchestrator.explain(
        text=text,
        predict_fn=predict_fn,
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        attentions=attentions,
    )

    # =====================================================
    # 🔥 FINAL WRAP
    # =====================================================

    result = ExplainabilityResult(
        prediction=prediction,

        shap_explanation=explanation.get("shap_explanation"),
        lime_explanation=explanation.get("lime_explanation"),
        attention_explanation=explanation.get("attention_explanation"),

        bias_explanation=explanation.get("bias_explanation"),
        emotion_explanation=explanation.get("emotion_explanation"),

        aggregated_explanation=explanation.get("aggregated_explanation"),

        consistency_metrics=explanation.get("consistency_metrics"),
        explanation_metrics=explanation.get("explanation_metrics"),

        monitoring=explanation.get("monitoring"),
        metadata=explanation.get("metadata"),
    )

    return result


# =========================================================
# BACKWARD COMPAT WRAPPERS
# =========================================================

def explain_prediction_full(
    text: str,
    predict_fn: PredictionFn,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    use_lime: bool = True,
    use_shap: bool = True,
) -> ExplainabilityResult:
    """
    Backward compatible full explanation.
    """

    config = ExplainabilityConfig(
        enabled=True,
        use_lime=use_lime,
        use_shap=use_shap,
        use_bias_emotion=True,
        use_attention_rollout=False,
        use_aggregation=True,
        use_consistency=True,
        use_explanation_metrics=True,
        cache_enabled=False,
    )

    return run_explainability_pipeline(
        text=text,
        predict_fn=predict_fn,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )


def explain_fast(
    text: str,
    predict_fn: PredictionFn,
) -> ExplainabilityResult:
    """
    Fast explainability (low latency).
    """

    config = ExplainabilityConfig(
        enabled=True,
        use_lime=True,
        use_shap=False,
        use_bias_emotion=False,
        use_attention_rollout=False,
        use_aggregation=False,
        use_consistency=False,
        use_explanation_metrics=False,
        cache_enabled=False,
    )

    return run_explainability_pipeline(
        text=text,
        predict_fn=predict_fn,
        config=config,
    )