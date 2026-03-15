"""
File: src/explainability/lime_explainer.py

Purpose
-------
Provide LIME explanations for model predictions.

LIME explains model decisions by perturbing text inputs
and identifying important tokens influencing predictions.

Outputs
-------
important_features
interactive HTML visualization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:  # pragma: no cover - environment-dependent
    LimeTextExplainer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_explainer: LimeTextExplainer | None = None


def _extract_fake_probability(result: Any) -> float:
    if not isinstance(result, dict) or "fake_probability" not in result:
        raise KeyError(
            "predict_fn(text) must return a dict with "
            "'fake_probability'."
        )

    fake_prob = float(result["fake_probability"])
    if fake_prob < 0.0 or fake_prob > 1.0:
        raise ValueError("fake_probability must be between 0 and 1.")
    return fake_prob


def get_explainer() -> LimeTextExplainer:
    """Lazily initialize and cache a LIME text explainer."""
    if LimeTextExplainer is None:
        raise ImportError(
            "LIME is not installed. Install dependency 'lime' to use "
            "explainability functions in src.explainability.lime_explainer."
        )

    global _explainer
    if _explainer is None:
        logger.info("Initializing LIME explainer")
        _explainer = LimeTextExplainer(class_names=["Real", "Fake"])

    return _explainer


def lime_predict_wrapper(
    texts: Sequence[str],
    predict_fn: Callable[[str], Dict[str, Any]],
) -> np.ndarray:
    """
    Convert prediction function output to a LIME-compatible probability matrix.
    """
    probs: list[list[float]] = []

    for text in texts:
        result = predict_fn(text)
        fake_prob = _extract_fake_probability(result)
        real_prob = 1.0 - fake_prob
        probs.append([real_prob, fake_prob])

    return np.asarray(probs, dtype=float)


def explain_prediction(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
    num_features: int = 10,
) -> Dict[str, Any]:
    """Generate a LIME explanation for one text sample."""
    if not text.strip():
        raise ValueError("text cannot be empty.")

    explainer = get_explainer()

    exp = explainer.explain_instance(
        text,
        lambda x: lime_predict_wrapper(x, predict_fn),
        num_features=num_features,
    )

    explanation = {
        "text": text,
        "important_features": exp.as_list(),
    }

    logger.info("LIME explanation generated")
    return explanation


def save_explanation_html(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
    output_path: str | Path = "reports/lime_explanation.html",
    num_features: int = 10,
) -> Path:
    """Save an interactive LIME explanation to HTML and return file path."""
    if not text.strip():
        raise ValueError("text cannot be empty.")

    explainer = get_explainer()

    exp = explainer.explain_instance(
        text,
        lambda x: lime_predict_wrapper(x, predict_fn),
        num_features=num_features,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp.save_to_file(str(output_path))

    logger.info("Saved LIME explanation: %s", output_path)
    return output_path
