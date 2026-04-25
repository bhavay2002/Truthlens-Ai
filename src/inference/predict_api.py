"""
File: predict_api.py 
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Union

from src.inference.inference_engine import InferenceConfig, InferenceEngine
from src.inference.prediction_service import PredictionService
from src.inference.inference_logger import InferenceLogger
from src.inference.inference_cache import InferenceCache, InferenceCacheConfig
from src.inference.monitoring import InferenceMonitor
from src.inference.postprocessing import Postprocessor
from src.inference.result_formatter import ResultFormatter
from src.utils.input_validation import ensure_non_empty_text

# =========================================================
# GLOBAL SINGLETON
# =========================================================

_service: PredictionService | None = None
_lock = threading.Lock()


# =========================================================
# LOAD SERVICE
# =========================================================

def _get_service() -> PredictionService:
    global _service

    if _service is not None:
        return _service

    with _lock:
        if _service is None:

            # ---------------- ENGINE ----------------
            engine = InferenceEngine(
                InferenceConfig(
                    model_path="models",
                    device="auto",
                )
            )

            # ---------------- CACHE ----------------
            cache = InferenceCache(
                InferenceCacheConfig(
                    enable_memory_cache=True,
                    cache_version="v2"
                )
            )

            # ---------------- LOGGER ----------------
            logger = InferenceLogger()

            # ---------------- MONITOR ----------------
            monitor = InferenceMonitor()

            # ---------------- POSTPROCESSOR ----------------
            postprocessor = Postprocessor()

            # ---------------- FORMATTER ----------------
            formatter = ResultFormatter()

            # ---------------- SERVICE ----------------
            _service = PredictionService(
                engine=engine,
                cache=cache,
                logger_=logger,
                formatter=formatter,
            )

            # attach optional components
            _service.monitor = monitor
            _service.postprocessor = postprocessor

    return _service


# =========================================================
# INPUT VALIDATION
# =========================================================

def _ensure_list(texts: Union[str, List[str]]) -> List[str]:

    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be non-empty list")

    for t in texts:
        ensure_non_empty_text(t)

    return texts


# =========================================================
# 🔥 MAIN API (HUMAN FRIENDLY)
# =========================================================

def predict(text: str) -> Dict[str, Any]:

    service = _get_service()
    return service.predict(text)


# =========================================================
# 🔥 BATCH API
# =========================================================

def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:

    texts = _ensure_list(texts)
    service = _get_service()

    return service.predict_batch(texts)


# =========================================================
# 🔥 FULL PIPELINE (REPORT)
# =========================================================

def predict_full(text: str) -> Dict[str, Any]:

    service = _get_service()
    return service.predict_full(text)


# =========================================================
# 🔥 FORMATTED OUTPUT
# =========================================================

def predict_formatted(
    text: str,
    *,
    mode: str = "api",
) -> Dict[str, Any]:

    service = _get_service()
    return service.predict_formatted(text, mode=mode)


# =========================================================
# 🔥 EVALUATION ENTRYPOINT
# =========================================================

def predict_for_evaluation(texts: List[str]) -> Dict[str, Any]:

    texts = _ensure_list(texts)
    service = _get_service()

    return service.predict_for_evaluation(texts)


# =========================================================
# 🔥 UNCERTAINTY SUPPORT
# =========================================================

def predict_with_uncertainty(texts: List[str]) -> Dict[str, Any]:

    texts = _ensure_list(texts)
    service = _get_service()

    outputs = service.predict_for_evaluation(texts)

    results = {}

    import numpy as np

    for task, out in outputs.items():

        probs = out.get("probabilities")

        if probs is not None:
            entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        else:
            entropy = None

        results[task] = {
            **out,
            "entropy": entropy,
        }

    return results


# =========================================================
# 🔥 MONITORING ENDPOINT
# =========================================================

def get_metrics() -> Dict[str, Any]:

    service = _get_service()

    if hasattr(service, "monitor"):
        return service.monitor.snapshot()

    return {}


# =========================================================
# 🔥 RESET CACHE (OPTIONAL ADMIN)
# =========================================================

def clear_cache():

    service = _get_service()

    if service.cache:
        service.cache.clear()