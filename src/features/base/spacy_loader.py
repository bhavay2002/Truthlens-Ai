"""Process-wide shared spaCy model loader.

Replaces the per-feature ``spacy.load("en_core_web_sm")`` calls scattered
across ``src/features/`` (3 sites: ``emotion_target_features``,
``narrative_role_features``, ``text/syntactic_features``).

Each ``spacy.load`` materialises a ~50MB pipeline; loading it once per
extractor instance means the same model is held in memory N times and the
import-time cost is paid N times. The loader here returns a single shared
``Language`` instance per model name, behind a lock so concurrent first
loads do not race.

If the model is not installed, the loader records the failure once and
returns ``None``. Callers must handle ``None`` (fall back to regex/blank
behaviour) — never assume the model is available.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_cache: Dict[str, Any] = {}
_failed: Dict[str, bool] = {}


def get_shared_nlp(model_name: str = "en_core_web_sm") -> Optional[Any]:
    """Return a shared spaCy ``Language`` for ``model_name``, or ``None``.

    The model is loaded the first time it is requested and cached for the
    lifetime of the process. ``None`` is returned (and remembered) when
    spaCy itself is unavailable or the model is not installed, so the
    second call does not re-attempt the failed import.
    """
    cached = _cache.get(model_name)
    if cached is not None:
        return cached
    if _failed.get(model_name):
        return None

    with _lock:
        # re-check after acquiring lock
        cached = _cache.get(model_name)
        if cached is not None:
            return cached
        if _failed.get(model_name):
            return None

        try:
            import spacy  # noqa: WPS433 (deferred import is intentional)
        except Exception as exc:
            logger.warning("spaCy not importable: %s", exc)
            _failed[model_name] = True
            return None

        try:
            nlp = spacy.load(model_name)
        except Exception as exc:
            logger.warning(
                "spaCy model '%s' unavailable; using fallback. (%s)",
                model_name, exc,
            )
            _failed[model_name] = True
            return None

        _cache[model_name] = nlp
        logger.info("Shared spaCy model loaded: %s", model_name)
        return nlp


def reset_shared_nlp() -> None:
    """Drop all cached models. Test-only."""
    with _lock:
        _cache.clear()
        _failed.clear()
