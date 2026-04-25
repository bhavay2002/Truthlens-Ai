from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple, Iterable, Iterator, Any

import spacy
from spacy.language import Language
from spacy.util import is_package

from src.analysis.analysis_config import ANALYSIS_CONFIG

logger = logging.getLogger(__name__)


# =========================================================
# CACHE (THREAD-SAFE, MULTI-MODEL)
# =========================================================

_CacheKey = Tuple[str, Tuple[str, ...]]
_CACHE: Dict[_CacheKey, Language] = {}
_LOCK = threading.RLock()


# =========================================================
# CONFIG (CENTRALIZED)
# =========================================================

DEFAULT_MODEL = ANALYSIS_CONFIG.spacy.model
ENABLE_GPU = ANALYSIS_CONFIG.spacy.use_gpu
DEFAULT_BATCH_SIZE = ANALYSIS_CONFIG.spacy.batch_size
DEFAULT_N_PROCESS = ANALYSIS_CONFIG.spacy.n_process

TASK_DISABLE_MAP = ANALYSIS_CONFIG.spacy.task_disable_map


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _resolve_model(model: str) -> str:
    """
    Resolve model with safe fallback.
    """
    if is_package(model):
        return model

    logger.warning("[spaCy] Model not found: %s → using blank 'en'", model)
    return "en"


def _maybe_enable_gpu():
    if not ENABLE_GPU:
        return

    try:
        if spacy.prefer_gpu():
            logger.info("[spaCy] GPU enabled")
        else:
            logger.warning("[spaCy] GPU requested but not available")
    except Exception as e:
        logger.warning("[spaCy] GPU init failed: %s", e)


def _validate_pipeline(nlp: Language, disable: Tuple[str, ...]) -> None:
    active = set(nlp.pipe_names)

    for pipe in disable:
        if pipe in active:
            logger.warning(
                "[spaCy] Pipe '%s' expected disabled but still active",
                pipe,
            )


# =========================================================
# CORE LOADER
# =========================================================

def get_nlp(
    model: str = DEFAULT_MODEL,
    disable: Optional[Tuple[str, ...]] = None,
) -> Language:

    disable_tuple = tuple(disable or ())
    key: _CacheKey = (model, disable_tuple)

    if key in _CACHE:
        return _CACHE[key]

    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

        resolved_model = _resolve_model(model)
        _maybe_enable_gpu()

        logger.info(
            "[spaCy] Loading | model=%s | disable=%s",
            resolved_model,
            disable_tuple,
        )

        try:
            if resolved_model == "en":
                nlp = spacy.blank("en")
            else:
                nlp = spacy.load(resolved_model, disable=list(disable_tuple))
        except Exception as e:
            logger.exception("[spaCy] Load failed")
            raise RuntimeError(f"Failed to load spaCy model: {model}") from e

        nlp.max_length = 2_000_000

        _validate_pipeline(nlp, disable_tuple)

        _CACHE[key] = nlp
        return nlp


# =========================================================
# TASK-AWARE LOADER
# =========================================================

def get_task_nlp(task: str) -> Language:

    if task not in TASK_DISABLE_MAP:
        raise ValueError(f"Unknown task: {task}")

    disable = TASK_DISABLE_MAP[task]
    return get_nlp(DEFAULT_MODEL, disable=disable)


# =========================================================
# 🔥 SHARED DOC CACHE (CRITICAL OPTIMIZATION)
# =========================================================

def get_doc(context: Any, task: str):
    """
    Retrieve spaCy doc using shared cache.

    Ensures:
    - single NLP pass per task per context
    - reused across features
    """

    if not hasattr(context, "shared") or context.shared is None:
        context.shared = {}

    cache = context.shared.setdefault("spacy_docs", {})

    if task in cache:
        return cache[task]

    nlp = get_task_nlp(task)
    doc = nlp(context.text)

    cache[task] = doc
    return doc


# =========================================================
# STREAM PROCESSING (HIGH PERFORMANCE)
# =========================================================

def process_docs_stream(
    texts: Iterable[str],
    *,
    task: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_process: int = DEFAULT_N_PROCESS,
) -> Iterator:

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if n_process < 1:
        raise ValueError("n_process must be >= 1")

    nlp = get_task_nlp(task)

    logger.debug(
        "[spaCy] Stream | task=%s | batch=%d | proc=%d",
        task,
        batch_size,
        n_process,
    )

    try:
        yield from nlp.pipe(
            texts,
            batch_size=batch_size,
            n_process=n_process,
        )
    except Exception as e:
        logger.exception("[spaCy] Stream processing failed")
        raise RuntimeError("spaCy pipeline execution failed") from e


# =========================================================
# MATERIALIZED PROCESSING
# =========================================================

def process_docs(
    texts: Iterable[str],
    *,
    task: str,
) -> list:

    return list(process_docs_stream(texts, task=task))


# =========================================================
# WARMUP (LOW LATENCY)
# =========================================================

def warmup_all_tasks() -> None:

    logger.info("[spaCy] Warmup start")

    for task in TASK_DISABLE_MAP:
        try:
            nlp = get_task_nlp(task)
            _ = nlp("Warmup text.")
        except Exception:
            logger.exception("[spaCy] Warmup failed for task=%s", task)

    logger.info("[spaCy] Warmup complete")


# =========================================================
# CACHE CONTROL
# =========================================================

def clear_cache() -> None:

    with _LOCK:
        _CACHE.clear()
        logger.info("[spaCy] Cache cleared")


# =========================================================
# INTROSPECTION
# =========================================================

def get_loaded_models() -> Dict[str, Dict]:

    info = {}

    for (model, disable), nlp in _CACHE.items():
        key = f"{model}|disable={disable}"
        info[key] = {
            "pipes": list(nlp.pipe_names),
            "max_length": nlp.max_length,
        }

    return info