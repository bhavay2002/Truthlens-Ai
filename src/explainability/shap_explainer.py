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
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from src.explainability.utils_validation import validate_tokens_scores

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore

logger = logging.getLogger(__name__)

_MAX_EXPLAINER_CACHE_SIZE = 8
_EXPLAINER_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
_LOCK = threading.RLock()

# Cache for computed SHAP Explanation objects so plot_explanation /
# save_explanation_html do not re-run the (expensive) perturbation loop.
_MAX_VALUE_CACHE_SIZE = 64
_VALUE_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()


def _process_shap_values(values):
    if isinstance(values, list):
        values = values[0]

    values = np.array(values, dtype=float)

    if values.ndim > 1:
        # For multi-class SHAP, take attributions for class index 1 (the
        # "fake" / positive class).  Averaging over classes produces near-zero
        # values for binary symmetric SHAP and destroys the attribution signal.
        values = values[:, 1]

    return values


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
    with _LOCK:
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


def _get_shap_values(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
) -> Any:
    """
    Return SHAP Explanation object for (predict_fn, text), using a
    bounded LRU cache to avoid recomputing across plot/save/explain calls.
    """
    cache_key = _cache_key_for_predict_fn(predict_fn) + (text,)
    with _LOCK:
        if cache_key in _VALUE_CACHE:
            _VALUE_CACHE.move_to_end(cache_key)
            return _VALUE_CACHE[cache_key]

    explainer = get_explainer(predict_fn)
    shap_values = explainer([text])

    with _LOCK:
        _VALUE_CACHE[cache_key] = shap_values
        _VALUE_CACHE.move_to_end(cache_key)
        while len(_VALUE_CACHE) > _MAX_VALUE_CACHE_SIZE:
            _VALUE_CACHE.popitem(last=False)

    return shap_values


def explain_text(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
) -> Dict[str, Any]:
    """
    Generate SHAP explanation values in a normalized, serializable schema.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text cannot be empty.")

    shap_values = _get_shap_values(predict_fn, text)
    tokens = list(shap_values.data[0])
    values = _process_shap_values(shap_values.values[0])
    min_len = min(len(tokens), len(values))
    tokens = tokens[:min_len]
    values = values[:min_len]

    filtered = [
        (t, v)
        for t, v in zip(tokens, values)
        if t not in {"[CLS]", "[SEP]", "<s>", "</s>"}
    ]

    if filtered:
        tokens, values = zip(*filtered)
        tokens = list(tokens)
        values = list(values)
        validate_tokens_scores(tokens, values)
    else:
        tokens, values = [], []

    token_importance = [
        {"token": str(tok), "importance": float(val)}
        for tok, val in zip(tokens, values)
    ]

    logger.info("SHAP explanation generated")

    return {
        "text": text,
        "token_importance": token_importance,
    }


def plot_explanation(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
) -> None:
    """
    Render SHAP text explanation in interactive environment.
    """

    if shap is None:
        raise ImportError("SHAP is not installed.")

    shap_values = _get_shap_values(predict_fn, text)

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

    shap_values = _get_shap_values(predict_fn, text)

    html = shap.plots.text(shap_values[0], display=False)

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(str(html))

    logger.info("Saved SHAP explanation: %s", output_path)

    return output_path