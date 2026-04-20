"""
File Name: _nlp.py
Module: Analysis - Shared spaCy Pipeline Cache

Description:
    Provides a thread-safe, in-process cache for spaCy Language pipelines so
    that each unique (model_name, disabled_components) combination is loaded
    only once per process lifetime.  All analyzer modules should obtain their
    pipeline via :func:`get_shared_nlp` (preferred) or :func:`get_nlp` rather
    than calling ``spacy.load`` directly.

    Caching strategy
    ----------------
    * Pipelines are keyed by ``(model_name, tuple(disable_components))``.
    * A ``threading.Lock`` guards the first-load path (double-checked locking),
      making the cache safe for typical multi-threaded web-service usage.
    * Subsequent accesses return the shared instance without acquiring the lock.

    Recommended usage (PERF-3)
    --------------------------
    All singleton analyzers in ``api/app.py`` should share ONE spaCy model
    instance.  Use :func:`get_shared_nlp` which always requests the fully-enabled
    ``en_core_web_sm`` pipeline (``disable=()``).  This means the entire process
    loads a single spaCy model regardless of how many analyzers are instantiated,
    avoiding the previous 4-instance proliferation.

Usage
-----
::

    from src.analysis._nlp import get_shared_nlp

    nlp = get_shared_nlp()
    doc = nlp("Some text to analyse.")
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple

import spacy
from spacy.language import Language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal cache
# ---------------------------------------------------------------------------

_CacheKey = Tuple[str, Tuple[str, ...]]

_CACHE: Dict[_CacheKey, Language] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_nlp(
    model: str = "en_core_web_sm",
    disable: Optional[Tuple[str, ...]] = None,
) -> Language:
    """Return a cached spaCy :class:`~spacy.language.Language` pipeline.

    If a pipeline with the given *model* and *disable* tuple has already been
    loaded in this process it is returned immediately (no lock required).
    On first load a :class:`threading.Lock` is acquired so that only one
    thread performs the expensive ``spacy.load`` call.

    Args:
        model:   spaCy model name, e.g. ``"en_core_web_sm"``.
        disable: Tuple of pipeline component names to disable, or ``None``
                 to enable all components.

    Returns:
        A fully initialised :class:`~spacy.language.Language` instance shared
        across all callers that request the same *(model, disable)* key.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    key: _CacheKey = (model, tuple(disable) if disable else ())

    # Fast path – no lock needed once cached
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    # Slow path – load under lock (double-checked locking pattern)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        disable_list = list(disable) if disable else []
        logger.info(
            "Loading spaCy pipeline: model=%s disable=%s",
            model,
            disable_list or "[]",
        )
        try:
            nlp = spacy.load(model, disable=disable_list)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load spaCy model '{model}': {exc}"
            ) from exc

        _CACHE[key] = nlp
        return nlp


_SHARED_MODEL = "en_core_web_sm"


def get_shared_nlp() -> Language:
    """Return the single shared spaCy pipeline used by all analyzers.

    Equivalent to ``get_nlp("en_core_web_sm", disable=())``.  All singleton
    analyzers in ``api/app.py`` should call this function so that exactly one
    spaCy model is held in memory for the entire process, regardless of how
    many analyzer classes are instantiated.

    Returns
    -------
    Language
        Fully-enabled ``en_core_web_sm`` pipeline shared process-wide.
    """
    return get_nlp(_SHARED_MODEL, disable=())


def clear_cache() -> None:
    """Remove all cached pipelines.  Intended for testing purposes only."""
    with _LOCK:
        _CACHE.clear()
