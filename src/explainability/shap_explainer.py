"""
File: src/explainability/shap_explainer.py

Purpose
-------
Provide SHAP-based explanations for model predictions.

SHAP computes Shapley values for tokens,
revealing how each word contributes to predictions.

Outputs
-------
token importance
interactive visualization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np

try:
    import shap
except ImportError:  # pragma: no cover - environment-dependent
    shap = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_EXPLAINER_CACHE: dict[int, Any] = {}


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


def shap_predict_wrapper(
    texts: Sequence[str],
    predict_fn: Callable[[str], Dict[str, Any]],
) -> np.ndarray:
    """Convert prediction outputs to class probability matrix for SHAP."""
    outputs: list[list[float]] = []

    for text in texts:
        result = predict_fn(text)
        fake_prob = _extract_fake_probability(result)
        real_prob = 1.0 - fake_prob
        outputs.append([real_prob, fake_prob])

    return np.asarray(outputs, dtype=float)


def get_explainer(
    predict_fn: Callable[[str], Dict[str, Any]],
):
    """Create or reuse a SHAP text explainer per prediction function."""
    if shap is None:
        raise ImportError(
            "SHAP is not installed. Install dependency 'shap' to use "
            "explainability functions in src.explainability.shap_explainer."
        )

    cache_key = id(predict_fn)
    if cache_key not in _EXPLAINER_CACHE:
        logger.info("Initializing SHAP explainer")
        masker = shap.maskers.Text()
        _EXPLAINER_CACHE[cache_key] = shap.Explainer(
            lambda x: shap_predict_wrapper(x, predict_fn),
            masker,
        )

    return _EXPLAINER_CACHE[cache_key]


def explain_text(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
):
    """Generate SHAP explanation values for one text."""
    if not text.strip():
        raise ValueError("text cannot be empty.")

    explainer = get_explainer(predict_fn)
    shap_values = explainer([text])

    logger.info("SHAP explanation generated")
    return shap_values


def plot_explanation(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
) -> None:
    """Render SHAP text explanation in the active environment."""
    if shap is None:
        raise ImportError("SHAP is not installed.")

    shap_values = explain_text(predict_fn, text)
    shap.plots.text(shap_values[0])


def save_explanation_html(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
    output_path: str | Path = "reports/shap_explanation.html",
) -> Path:
    """Save SHAP text explanation as HTML and return path."""
    if shap is None:
        raise ImportError("SHAP is not installed.")

    shap_values = explain_text(predict_fn, text)
    html = shap.plots.text(shap_values[0], display=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(str(html))

    logger.info("Saved SHAP explanation: %s", output_path)
    return output_path
