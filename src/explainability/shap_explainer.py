"""
File Name: shap_explainer.py
Module: Explainability - SHAP
Description:
    Provides SHAP-based explanations for model predictions in the TruthLens AI
    system. SHAP (SHapley Additive exPlanations) computes token-level Shapley
    values, revealing how each word contributes to prediction outcomes.

    The module supports:
        • SHAP value computation
        • Token-level importance
        • Interactive visualization
        • HTML report generation
        • Explainer caching for performance


Dependencies:
    collections
    logging
    pathlib
    typing
    numpy
    shap

Inputs:
    predict_fn : Callable[[str], Dict[str, Any]]
    text : str

Outputs:
    SHAP explanation object
    HTML visualization (optional)
"""

from __future__ import annotations

from collections import OrderedDict
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore

logger = logging.getLogger(__name__)

_MAX_EXPLAINER_CACHE_SIZE = 8
_EXPLAINER_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()


def _extract_fake_probability(result: Any) -> float:
    """
    Extract fake news probability from prediction output.
    """

    if not isinstance(result, dict) or "fake_probability" not in result:
        raise KeyError(
            "predict_fn(text) must return a dict with 'fake_probability'."
        )

    fake_prob = float(result["fake_probability"])

    if fake_prob < 0.0 or fake_prob > 1.0:
        raise ValueError("fake_probability must be between 0 and 1.")

    return fake_prob


def shap_predict_wrapper(
    texts: Sequence[str],
    predict_fn: Callable[[str], Dict[str, Any]],
) -> np.ndarray:
    """
    Convert predictor output into probability matrix required by SHAP.
    """

    outputs: list[list[float]] = []

    for text in texts:
        result = predict_fn(text)

        fake_prob = _extract_fake_probability(result)

        real_prob = 1.0 - fake_prob

        outputs.append([real_prob, fake_prob])

    return np.asarray(outputs, dtype=float)


def _cache_key_for_predict_fn(
    predict_fn: Callable[[str], Dict[str, Any]],
) -> Tuple[Any, ...]:
    """
    Generate stable cache key for predictor callable.
    """

    module_name = getattr(predict_fn, "__module__", None)
    qual_name = getattr(predict_fn, "__qualname__", None)
    bound_instance = getattr(predict_fn, "__self__", None)

    if module_name and qual_name and "<lambda>" not in qual_name:

        if bound_instance is not None:
            return ("bound_method", module_name, qual_name, id(bound_instance))

        return ("function", module_name, qual_name)

    return ("ephemeral", id(predict_fn))


def _set_cache_entry(cache_key: Tuple[Any, ...], explainer: Any) -> None:
    """
    Insert explainer into LRU cache.
    """

    _EXPLAINER_CACHE[cache_key] = explainer

    _EXPLAINER_CACHE.move_to_end(cache_key)

    while len(_EXPLAINER_CACHE) > _MAX_EXPLAINER_CACHE_SIZE:
        evicted_key, _ = _EXPLAINER_CACHE.popitem(last=False)

        logger.debug("Evicted SHAP explainer cache key: %s", evicted_key)


def get_explainer(
    predict_fn: Callable[[str], Dict[str, Any]],
):
    """
    Create or reuse SHAP text explainer for prediction function.
    """

    if shap is None:
        raise ImportError(
            "SHAP is not installed. Install dependency 'shap' "
            "to use explainability features."
        )

    cache_key = _cache_key_for_predict_fn(predict_fn)

    if cache_key not in _EXPLAINER_CACHE:

        logger.info("Initializing SHAP explainer")

        masker = shap.maskers.Text()

        explainer = shap.Explainer(
            lambda x: shap_predict_wrapper(x, predict_fn),
            masker,
        )

        _set_cache_entry(cache_key, explainer)

    else:

        _EXPLAINER_CACHE.move_to_end(cache_key)

    return _EXPLAINER_CACHE[cache_key]


def explain_text(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
):
    """
    Generate SHAP explanation values for one text sample.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text cannot be empty.")

    explainer = get_explainer(predict_fn)

    shap_values = explainer([text])

    logger.info("SHAP explanation generated")

    return shap_values


def plot_explanation(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
) -> None:
    """
    Render SHAP text explanation in interactive environment.
    """

    if shap is None:
        raise ImportError("SHAP is not installed.")

    shap_values = explain_text(predict_fn, text)

    shap.plots.text(shap_values[0])


def save_explanation_html(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
    output_path: str | Path = "reports/shap_explanation.html",
) -> Path:
    """
    Save SHAP explanation visualization as HTML.
    """

    if shap is None:
        raise ImportError("SHAP is not installed.")

    shap_values = explain_text(predict_fn, text)

    html = shap.plots.text(shap_values[0], display=False)

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(str(html))

    logger.info("Saved SHAP explanation: %s", output_path)

    return output_path