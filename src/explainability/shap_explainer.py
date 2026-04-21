from __future__ import annotations

from collections import OrderedDict
import hashlib
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

SPECIAL_TOKENS = {
    "[CLS]", "[SEP]", "<s>", "</s>",
    "[PAD]", "<pad>", "[UNK]", "<unk>",
}


def _process_shap_values(values):
    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values, dtype=np.float32)

    if values.ndim == 3:
        values = values[:, :, -1]
    elif values.ndim == 2:
        if values.shape[1] == 1:
            values = values[:, 0]

    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

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

    batch_fn = getattr(predict_fn, "batch_predict", None)
    if callable(batch_fn):
        try:
            results = batch_fn(list(texts))
            outputs = []
            for result in results:
                fake_prob = _extract_fake_probability(result)
                outputs.append([1.0 - fake_prob, fake_prob])
            return np.asarray(outputs, dtype=float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Batch prediction failed, falling back: %s", exc)

    outputs: list[list[float]] = []

    for text in texts:
        result = predict_fn(text)

        fake_prob = _extract_fake_probability(result)

        real_prob = 1.0 - fake_prob

        outputs.append([real_prob, fake_prob])

    return np.asarray(outputs, dtype=float)


def _stable_predict_fn_key(
    predict_fn: Callable[[str], Dict[str, Any]],
) -> Tuple[Any, ...]:
    """
    Generate a stable cache key for predict_fn.

    Priority:
    1. Explicit attribute (recommended)
    2. Model metadata
    3. Fallback to module+name
    """

    stable_id = getattr(predict_fn, "__cache_key__", None)
    if isinstance(stable_id, str):
        return ("explicit", stable_id)

    bound = getattr(predict_fn, "__self__", None)
    if bound is not None:
        model_name = getattr(bound, "model_name", None)
        tokenizer_name = getattr(bound, "tokenizer_name", None)

        if model_name or tokenizer_name:
            return ("model", model_name, tokenizer_name)

    module_name = getattr(predict_fn, "__module__", None)
    qual_name = getattr(predict_fn, "__qualname__", None)

    if module_name and qual_name:
        return ("function", module_name, qual_name)

    return ("fallback", repr(predict_fn))


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

    cache_key = _stable_predict_fn_key(predict_fn)
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
    text_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    cache_key = _stable_predict_fn_key(predict_fn) + (text_hash,)
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


def get_shap_values(
    predict_fn: Callable[[str], Dict[str, Any]],
    text: str,
) -> Any:
    return _get_shap_values(predict_fn, text)


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
    data = getattr(shap_values, "data", None)
    if data is None or len(data) == 0:
        return {
            "text": text,
            "token_importance": [],
        }

    tokens = list(data[0])
    values = _process_shap_values(shap_values.values[0])

    if len(tokens) == 0 or len(values) == 0:
        return {
            "text": text,
            "token_importance": [],
        }

    min_len = min(len(tokens), len(values))
    tokens = tokens[:min_len]
    values = values[:min_len]

    if min_len == 0:
        return {
            "text": text,
            "token_importance": [],
        }

    filtered = [
        (t, v)
        for t, v in zip(tokens, values)
        if t not in SPECIAL_TOKENS
    ]

    if filtered:
        tokens, values = zip(*filtered)
        tokens = list(tokens)
        values = list(values)
        validate_tokens_scores(tokens, values)
    else:
        tokens, values = [], []

    if values:
        max_abs = max((abs(v) for v in values), default=1.0)
        if max_abs == 0:
            max_abs = 1.0
        values = [float(v / max_abs) for v in values]

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
    try:
        shap_values = _get_shap_values(predict_fn, text)
        shap.plots.text(shap_values[0])
    except Exception as e:  # noqa: BLE001
        logger.warning("SHAP plot failed: %s", e)


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
    html_str = str(html) if html is not None else "<p>No SHAP output</p>"

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_str)

    logger.info("Saved SHAP explanation: %s", output_path)

    return output_path