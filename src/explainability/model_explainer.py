"""
File Name: model_explainer.py
Module: Explainability - Unified Explanation Engine
Description:
    Unified explanation engine for TruthLens AI.

    This module orchestrates multiple explainability subsystems and produces
    a consolidated explanation object for model predictions.

    Supported explanation components:
        - Bias explanation
        - Emotion explanation
        - SHAP explanation
        - LIME explanation

    The module supports both full research-grade explainability pipelines and
    low-latency fast explanations for production APIs.


Dependencies:
    logging
    typing

Inputs:
    text : str
    predict_fn : Callable[[str], Dict[str, Any]]
    model : optional transformer model
    tokenizer : optional tokenizer

Outputs:
    Unified explanation dictionary
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from src.explainability.bias_explainer import explain_bias
from src.explainability.emotion_explainer import explain_emotion
from src.explainability.lime_explainer import explain_prediction as explain_lime_prediction
from src.explainability.shap_explainer import explain_text as explain_shap_text
from src.explainability.token_alignment import align_tokens
from src.explainability.utils_validation import validate_tokens_scores

logger = logging.getLogger(__name__)

PredictionFn = Callable[[str], Dict[str, Any]]


def _run_component(name: str, fn: Callable[[], Any]) -> Any:
    """
    Execute an explainability component safely.

    If the component fails, the error is logged and the system continues
    without interrupting the full explanation pipeline.
    """
    try:
        return fn()
    except Exception as exc:  # pragma: no cover
        logger.warning("%s explanation failed: %s", name, exc)
        return None


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

    This includes prediction outputs along with optional explainability
    components such as bias analysis, emotional manipulation detection,
    SHAP attribution, and LIME token explanations.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    if not callable(predict_fn):
        raise TypeError("predict_fn must be callable.")

    logger.info("Running unified model explanation pipeline")

    prediction = predict_fn(text)

    bias_explanation = None
    emotion_explanation = None

    if model is not None and tokenizer is not None:
        bias_explanation = _run_component(
            "Bias",
            lambda: explain_bias(model, tokenizer, text),
        )

        emotion_explanation = _run_component(
            "Emotion",
            lambda: explain_emotion(text, model, tokenizer),
        )
    else:
        logger.info(
            "Skipping bias/emotion explanations because "
            "model/tokenizer were not provided."
        )

    shap_explanation = None
    if use_shap:
        shap_explanation = _run_component(
            "SHAP",
            lambda: explain_shap_text(predict_fn, text),
        )
        if isinstance(shap_explanation, dict):
            token_items = shap_explanation.get("token_importance")
            if isinstance(token_items, list) and token_items:
                tokens = []
                scores = []
                for item in token_items:
                    if not isinstance(item, dict):
                        continue
                    token = item.get("token")
                    score = item.get("importance")
                    if isinstance(token, str) and isinstance(score, (int, float)):
                        tokens.append(token)
                        scores.append(score)
                if len(tokens) == len(scores) and tokens:
                    validate_tokens_scores(tokens, scores)
                    tokens, scores = align_tokens(tokens, scores)
                    shap_explanation["token_importance"] = [
                        {"token": t, "importance": float(s)}
                        for t, s in zip(tokens, scores)
                    ]

    lime_explanation = None
    if use_lime:
        lime_explanation = _run_component(
            "LIME",
            lambda: explain_lime_prediction(predict_fn, text),
        )
        if isinstance(lime_explanation, dict):
            items = lime_explanation.get("important_features")
            if isinstance(items, list) and items:
                tokens = []
                scores = []
                for item in items:
                    if not isinstance(item, (list, tuple)) or len(item) < 2:
                        continue
                    token, score = item[0], item[1]
                    if isinstance(token, str) and isinstance(score, (int, float)):
                        tokens.append(token)
                        scores.append(score)
                if len(tokens) == len(scores) and tokens:
                    validate_tokens_scores(tokens, scores)
                    tokens, scores = align_tokens(tokens, scores)
                    lime_explanation["important_features"] = list(zip(tokens, scores))

    results: Dict[str, Any] = {
        "prediction": prediction,
        "bias_explanation": bias_explanation,
        "emotion_explanation": emotion_explanation,
        "shap_explanation": shap_explanation,
        "lime_explanation": lime_explanation,
    }

    logger.info("Unified explanation pipeline completed")

    return results


def explain_fast(
    text: str,
    predict_fn: PredictionFn,
) -> Dict[str, Any]:
    """
    Fast explanation pipeline intended for low-latency API endpoints.

    Only prediction and LIME explanation are executed.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")

    if not callable(predict_fn):
        raise TypeError("predict_fn must be callable.")

    logger.info("Running fast explanation pipeline")

    prediction = predict_fn(text)

    lime_explanation = _run_component(
        "LIME",
        lambda: explain_lime_prediction(predict_fn=predict_fn, text=text),
    )

    return {
        "prediction": prediction,
        "lime_explanation": lime_explanation,
    }