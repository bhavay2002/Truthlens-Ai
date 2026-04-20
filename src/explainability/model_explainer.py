"""
File Name: model_explainer.py
Module: Explainability - Unified Explanation Engine (backward-compatibility shim)
Description:
    Thin wrappers kept for backward compatibility.

    The canonical implementation has moved to
    :class:`~src.explainability.orchestrator.ExplainabilityOrchestrator`.
    These module-level functions construct a minimal, stateless orchestrator
    and delegate to it so that any existing call-sites continue to work without
    modification.

    New code should use ``ExplainabilityOrchestrator`` directly.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from src.explainability.orchestrator import ExplainabilityConfig, ExplainabilityOrchestrator

logger = logging.getLogger(__name__)

PredictionFn = Callable[[str], Dict[str, Any]]


def explain_prediction_full(
    text: str,
    predict_fn: PredictionFn,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    use_lime: bool = True,
    use_shap: bool = True,
) -> Dict[str, Any]:
    """
    Generate a full unified explanation package for a model prediction.

    .. deprecated::
        Delegate to :meth:`ExplainabilityOrchestrator.explain` instead.

    This wrapper constructs a stateless orchestrator on every call (no
    cache, no attention rollout, no propaganda explainer) and returns a
    dict with the same keys as the old implementation.
    """
    logger.debug(
        "explain_prediction_full called — delegating to ExplainabilityOrchestrator"
    )

    config = ExplainabilityConfig(
        enabled=True,
        use_lime=use_lime,
        use_shap=use_shap,
        use_bias_emotion=True,
        use_attention_rollout=False,
        use_propaganda_explainer=False,
        use_aggregation=False,
        use_consistency=False,
        cache_enabled=False,
    )
    orchestrator = ExplainabilityOrchestrator(config=config)

    result = orchestrator.explain(
        text=text,
        predict_fn=predict_fn,
        model=model,
        tokenizer=tokenizer,
    )

    prediction = predict_fn(text)

    return {
        "prediction": prediction,
        "bias_explanation": result.get("bias_explanation"),
        "emotion_explanation": result.get("emotion_explanation"),
        "shap_explanation": result.get("shap_explanation"),
        "lime_explanation": result.get("lime_explanation"),
    }


def explain_fast(
    text: str,
    predict_fn: PredictionFn,
) -> Dict[str, Any]:
    """
    Fast explanation pipeline intended for low-latency API endpoints.

    .. deprecated::
        Use :meth:`ExplainabilityOrchestrator.explain_fast` instead.
    """
    logger.debug(
        "explain_fast called — delegating to ExplainabilityOrchestrator"
    )

    config = ExplainabilityConfig(
        enabled=True,
        use_lime=True,
        use_shap=False,
        use_bias_emotion=False,
        use_attention_rollout=False,
        use_propaganda_explainer=False,
        use_aggregation=False,
        use_consistency=False,
        cache_enabled=False,
    )
    orchestrator = ExplainabilityOrchestrator(config=config)
    return orchestrator.explain_fast(text=text, predict_fn=predict_fn)
