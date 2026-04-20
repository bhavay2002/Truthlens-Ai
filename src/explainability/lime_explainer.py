"""
File Name: lime_explainer.py
Module: Explainability - LIME
Description:
    Provides Local Interpretable Model-Agnostic Explanations (LIME) for
    text classification predictions within the TruthLens AI system.

    This module supports generating token-level importance explanations
    by perturbing text inputs and observing changes in model predictions.
    It produces both structured explanation data and interactive HTML
    visualizations suitable for dashboards and reports.

Dependencies:
    logging
    pathlib
    typing
    numpy
    lime

Inputs:
    predict_fn : callable prediction function
    text : str input text

Outputs:
    explanation dictionary
    optional HTML visualization
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:  # pragma: no cover
    LimeTextExplainer = None  # type: ignore

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CACHE: Dict[str, LimeTextExplainer] = {}


def _extract_fake_probability(result: Any) -> float:
    """
    Extract fake probability from prediction result.
    """
    if not isinstance(result, dict) or "fake_probability" not in result:
        raise KeyError(
            "predict_fn(text) must return a dict containing 'fake_probability'."
        )

    prob = float(result["fake_probability"])

    if prob < 0.0 or prob > 1.0:
        raise ValueError("fake_probability must be between 0 and 1.")

    return prob


def get_explainer(model_id: str = "default") -> LimeTextExplainer:
    """
    Lazily initialize and cache a LimeTextExplainer instance.
    """
    if LimeTextExplainer is None:
        raise ImportError(
            "LIME is not installed. Install 'lime' to enable "
            "src.explainability.lime_explainer."
        )

    with _LOCK:
        if model_id not in _CACHE:
            logger.info("Initializing LIME text explainer")
            _CACHE[model_id] = LimeTextExplainer(class_names=["Real", "Fake"])

        return _CACHE[model_id]


def _extract_fake_probabilities_from_batch(
    batch_result: Any,
    expected_size: int,
) -> List[float] | None:
    """
    Extract probabilities from batch prediction output.
    """
    if (
        not isinstance(batch_result, Sequence)
        or isinstance(batch_result, (str, bytes))
        or len(batch_result) != expected_size
    ):
        return None

    probabilities: List[float] = []

    for item in batch_result:
        try:
            if isinstance(item, dict):
                probabilities.append(_extract_fake_probability(item))
            elif (
                isinstance(item, (list, tuple))
                and len(item) >= 2
            ):
                # predict_batch returns [prob_real, prob_fake, ...]
                probabilities.append(float(item[1]))
            elif hasattr(item, "__len__") and len(item) >= 2:
                probabilities.append(float(item[1]))
            else:
                return None
        except Exception:
            return None

    return probabilities


def lime_predict_wrapper(
    texts: Sequence[str],
    predict_fn: Callable[[Any], Any],
) -> np.ndarray:
    """
    Convert prediction outputs to LIME-compatible probability matrix.
    """
    text_list = [str(text) for text in texts]

    batch_probs: List[float] | None = None

    try:
        batch_result = predict_fn(text_list)

        batch_probs = _extract_fake_probabilities_from_batch(
            batch_result,
            expected_size=len(text_list),
        )

    except Exception:
        batch_probs = None

    probabilities: List[List[float]] = []

    if batch_probs is not None:
        for fake_prob in batch_probs:
            probabilities.append([1.0 - fake_prob, fake_prob])
    else:
        for text in text_list:
            result = predict_fn(text)
            fake_prob = _extract_fake_probability(result)
            probabilities.append([1.0 - fake_prob, fake_prob])

    return np.asarray(probabilities, dtype=float)


def explain_prediction(
    predict_fn: Callable[[Any], Any],
    text: str,
    num_features: int = 10,
    num_samples: int = 256,
) -> Dict[str, Any]:
    """
    Generate a LIME explanation for a single text input.
    """

    if not isinstance(text, str) or not text.strip():
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
    """
    Save interactive LIME explanation visualization to HTML.
    """

    if not isinstance(text, str) or not text.strip():
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

    logger.info("Saved LIME explanation to %s", output_path)

    return output_path