from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lime.lime_text import LimeTextExplainer
else:
    LimeTextExplainer = Any

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_MAX_CACHE_SIZE = 4
_CACHE: Dict[str, LimeTextExplainer] = OrderedDict()


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
    Lazily initialize and cache a LimeTextExplainer instance
    with LRU eviction to prevent unbounded growth.
    """
    if LimeTextExplainer is None:
        raise ImportError(
            "LIME is not installed. Install 'lime' to enable "
            "src.explainability.lime_explainer."
        )

    with _LOCK:
        if model_id in _CACHE:
            _CACHE.move_to_end(model_id)
            return _CACHE[model_id]

        logger.info("Initializing LIME text explainer (model_id=%s)", model_id)
        explainer = LimeTextExplainer(class_names=["Real", "Fake"])
        _CACHE[model_id] = explainer
        _CACHE.move_to_end(model_id)

        if len(_CACHE) > _MAX_CACHE_SIZE:
            evicted_id, _ = _CACHE.popitem(last=False)
            logger.debug("Evicted LIME explainer from cache (model_id=%s)", evicted_id)

        return explainer


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

    if not isinstance(probabilities, list) or len(probabilities) != expected_size:
        return None

    return probabilities


def lime_predict_wrapper(
    texts: Sequence[str],
    predict_fn: Callable[[Any], Any],
) -> np.ndarray:
    """
    Fully batch-aware LIME prediction wrapper.
    """
    text_list = [str(t) for t in texts]

    batch_fn = getattr(predict_fn, "batch_predict", None)
    if callable(batch_fn):
        try:
            results = batch_fn(text_list)
            probs: List[List[float]] = []
            for result in results:
                fake_prob = _extract_fake_probability(result)
                probs.append([1.0 - fake_prob, fake_prob])
            return np.asarray(probs, dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch_predict failed: %s", exc)

    try:
        batch_result = predict_fn(text_list)
        batch_probs = _extract_fake_probabilities_from_batch(
            batch_result,
            expected_size=len(text_list),
        )
        if batch_probs is not None:
            return np.asarray(
                [[1.0 - prob, prob] for prob in batch_probs],
                dtype=float,
            )
    except Exception:
        pass

    outputs: List[List[float]] = []
    for text in text_list:
        result = predict_fn(text)
        fake_prob = _extract_fake_probability(result)
        outputs.append([1.0 - fake_prob, fake_prob])

    return np.asarray(outputs, dtype=float)


def _batched_predict(
    texts: Sequence[str],
    predict_fn: Callable[[Any], Any],
    batch_size: int = 32,
) -> np.ndarray:
    results: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        chunk_preds = lime_predict_wrapper(chunk, predict_fn)
        results.append(chunk_preds)

    return np.vstack(results) if results else np.zeros((0, 2), dtype=float)


def _get_lime_predict_fn(
    predict_fn: Callable[[Any], Any],
) -> Callable[[Sequence[str]], np.ndarray]:
    return lambda x: _batched_predict(x, predict_fn)


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
    predictor = _get_lime_predict_fn(predict_fn)

    exp = explainer.explain_instance(
        text,
        predictor,
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
    predictor = _get_lime_predict_fn(predict_fn)

    exp = explainer.explain_instance(
        text,
        predictor,
        num_features=num_features,
        num_samples=num_samples,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp.save_to_file(str(output_path))

    logger.info("Saved LIME explanation to %s", output_path)

    return output_path


def clear_explainer_cache() -> None:
    with _LOCK:
        _CACHE.clear()
        logger.info("Cleared LIME explainer cache")


def cache_info() -> dict:
    with _LOCK:
        return {
            "size": len(_CACHE),
            "capacity": _MAX_CACHE_SIZE,
            "keys": list(_CACHE.keys()),
        }