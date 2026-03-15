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


def _extract_fake_probabilities_from_batch(
    batch_result: Any,
    expected_size: int,
) -> list[float] | None:
    """Extract fake probabilities from batch-style predictor output."""
    if (
        not isinstance(batch_result, Sequence)
        or isinstance(batch_result, (str, bytes))
        or len(batch_result) != expected_size
    ):
        return None

    probs: list[float] = []
    for item in batch_result:
        try:
            probs.append(_extract_fake_probability(item))
        except Exception:
            return None
    return probs


def lime_predict_wrapper(
    texts: Sequence[str],
    predict_fn: Callable[[Any], Any],
) -> np.ndarray:
    """
    Convert predictor output to a LIME-compatible probability matrix.

    Supports both:
    - single-text predictors: predict_fn(text) -> {"fake_probability": ...}
    - batch predictors: predict_fn(list[str]) -> list[{"fake_probability": ...}]
    """
    text_list = [str(text) for text in texts]

    # Fast path: try batch prediction first.
    batch_fake_probs: list[float] | None = None
    try:
        batch_result = predict_fn(text_list)
        batch_fake_probs = _extract_fake_probabilities_from_batch(
            batch_result,
            expected_size=len(text_list),
        )
    except Exception:
        batch_fake_probs = None

    probs: list[list[float]] = []
    if batch_fake_probs is not None:
        for fake_prob in batch_fake_probs:
            probs.append([1.0 - fake_prob, fake_prob])
    else:
        for text in text_list:
            result = predict_fn(text)
            fake_prob = _extract_fake_probability(result)
            probs.append([1.0 - fake_prob, fake_prob])

    return np.asarray(probs, dtype=float)


def explain_prediction(
    predict_fn: Callable[[Any], Any],
    text: str,
    num_features: int = 10,
    num_samples: int = 256,
) -> Dict[str, Any]:
    """Generate a LIME explanation for one text sample."""
    if not text.strip():
        raise ValueError("text cannot be empty.")
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0.")

    explainer = get_explainer()

    exp = explainer.explain_instance(
        text,
        lambda x: lime_predict_wrapper(x, predict_fn),
        num_features=num_features,
        num_samples=num_samples,
    )

    explanation = {
        "text": text,
        "important_features": exp.as_list(),
    }

    logger.info("LIME explanation generated")
    return explanation


def save_explanation_html(
    predict_fn: Callable[[Any], Any],
    text: str,
    output_path: str | Path = "reports/lime_explanation.html",
    num_features: int = 10,
    num_samples: int = 256,
) -> Path:
    """Save an interactive LIME explanation to HTML and return file path."""
    if not text.strip():
        raise ValueError("text cannot be empty.")
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0.")

    explainer = get_explainer()

    exp = explainer.explain_instance(
        text,
        lambda x: lime_predict_wrapper(x, predict_fn),
        num_features=num_features,
        num_samples=num_samples,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp.save_to_file(str(output_path))

    logger.info("Saved LIME explanation: %s", output_path)
    return output_path
