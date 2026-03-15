"""
File: src/explainability/model_explainer.py

Purpose
-------
Unified explanation engine for TruthLens AI.

Combines:
- Model predictions
- Bias explanations
- Emotion explanations
- SHAP explanations
- LIME explanations

Provides a single interpretable output
for fake news predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from src.explainability.bias_explainer import explain_bias
from src.explainability.emotion_explainer import explain_emotion
from src.explainability.lime_explainer import explain_prediction
from src.explainability.shap_explainer import explain_text

logger = logging.getLogger(__name__)


PredictionFn = Callable[[str], Dict[str, Any]]


def _run_component(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive safety net
        logger.warning("%s explanation failed: %s", name, exc)
        return None


def explain_prediction_full(
    text: str,
    predict_fn: PredictionFn,
    model=None,
    tokenizer=None,
    use_lime: bool = True,
    use_shap: bool = True,
) -> Dict[str, Any]:
    """Generate a complete explanation package for one prediction."""
    logger.info("Running unified model explanation")

    prediction = predict_fn(text)

    bias_explanation = _run_component(
        "Bias",
        lambda: explain_bias(model, tokenizer, text),
    )
    emotion_explanation = _run_component(
        "Emotion",
        lambda: explain_emotion(text, model, tokenizer),
    )

    shap_explanation = None
    if use_shap:
        shap_explanation = _run_component(
            "SHAP",
            lambda: explain_text(predict_fn, text),
        )

    lime_explanation = None
    if use_lime:
        lime_explanation = _run_component(
            "LIME",
            lambda: explain_prediction(predict_fn, text),
        )

    results = {
        "prediction": prediction,
        "bias_explanation": bias_explanation,
        "emotion_explanation": emotion_explanation,
        "shap_explanation": shap_explanation,
        "lime_explanation": lime_explanation,
    }

    logger.info("Unified explanation completed")
    return results


def explain_fast(
    text: str,
    predict_fn: PredictionFn,
) -> Dict[str, Any]:
    """Fast explanation path intended for low-latency endpoints."""
    prediction = predict_fn(text)
    lime_explanation = explain_prediction(predict_fn, text)

    return {
        "prediction": prediction,
        "lime_explanation": lime_explanation,
    }
