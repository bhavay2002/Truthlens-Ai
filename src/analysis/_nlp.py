from __future__ import annotations

import logging
import threading
from typing import Dict, Optional, Tuple, Iterable, List

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
# Configuration
# ---------------------------------------------------------------------------

_SHARED_MODEL = "en_core_web_sm"

# 🔥 SAFE DEFAULT (DO NOT BREAK ANALYZERS)
SAFE_DISABLE = ()

# ⚡ FAST MODE (ONLY IF YOU KNOW WHAT YOU'RE DOING)
FAST_DISABLE = ("parser",)  # keep tagger + lemmatizer


DEFAULT_BATCH_SIZE = 32
DEFAULT_N_PROCESS = 1


# ---------------------------------------------------------------------------
# Core Loader
# ---------------------------------------------------------------------------

def get_nlp(
    model: str = _SHARED_MODEL,
    disable: Optional[Tuple[str, ...]] = None,
) -> Language:

    key: _CacheKey = (model, tuple(disable) if disable else ())

    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        disable_list = list(disable) if disable else []

        logger.info(
            "[spaCy] Loading model=%s | disable=%s",
            model,
            disable_list or "[]",
        )

        try:
            nlp = spacy.load(model, disable=disable_list)

            # 🔥 Safety + performance
            nlp.max_length = 2_000_000

        except Exception as exc:
            raise RuntimeError(
                f"Failed to load spaCy model '{model}': {exc}"
            ) from exc

        _CACHE[key] = nlp
        return nlp


# ---------------------------------------------------------------------------
# Shared NLP
# ---------------------------------------------------------------------------

def get_shared_nlp(mode: str = "safe") -> Language:
    """
    mode:
        - "safe" → full pipeline (recommended)
        - "fast" → disables parser only
    """

    if mode == "fast":
        disable = FAST_DISABLE
    else:
        disable = SAFE_DISABLE

    return get_nlp(_SHARED_MODEL, disable=disable)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------

def warmup() -> None:
    logger.info("[spaCy] Warmup start")

    nlp = get_shared_nlp()

    _ = nlp("Warmup text for pipeline initialization.")

    logger.info("[spaCy] Warmup complete")


# ---------------------------------------------------------------------------
# Batch Config (CENTRALIZED)
# ---------------------------------------------------------------------------

def get_batch_config():
    return {
        "batch_size": DEFAULT_BATCH_SIZE,
        "n_process": DEFAULT_N_PROCESS,
    }


# ---------------------------------------------------------------------------
# Batch Processing (PIPE ONLY)
# ---------------------------------------------------------------------------

def process_docs(
    texts: Iterable[str],
    *,
    batch_size: Optional[int] = None,
    n_process: Optional[int] = None,
) -> List:
    """
    Returns spaCy Docs ONLY.
    Use inside pipeline.run_batch().
    """

    nlp = get_shared_nlp()

    config = get_batch_config()

    return list(
        nlp.pipe(
            texts,
            batch_size=batch_size or config["batch_size"],
            n_process=n_process or config["n_process"],
        )
    )


# ---------------------------------------------------------------------------
# Cache Control
# ---------------------------------------------------------------------------

def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()