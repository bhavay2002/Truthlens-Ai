"""Canonical text tokenization for the features layer.

A single tokenizer used by every feature extractor. Replaces the ~19 ad-hoc
`_tokenize` / `_simple_tokenize` helpers that had:

* Inconsistent regexes (`\\b\\w+\\b` vs `[A-Za-z']+`).
* ASCII-only behaviour (`[A-Za-z']+` silently strips accented characters
  like ``café`` -> ``caf``, ``résumé`` -> ``rsum``).
* Per-feature recompilation and re-tokenization of the same text.

The canonical pattern matches Unicode letter runs with optional contraction
suffixes (``don't``, ``it's``). Digits and underscores are excluded because
they are not linguistic tokens for the bias/emotion/propaganda features.

Pipeline-level callers should populate ``FeatureContext.tokens_word`` once
at the top of the request via ``ensure_tokens_word(ctx)``; downstream
features then read ``ctx.tokens_word`` for free.
"""

from __future__ import annotations

import re
from typing import List, Optional

# ``[^\W\d_]`` = "word character minus digits and underscore" = Unicode letter.
# Apostrophe-suffix is optional (``don't``, ``it's``).  ``re.UNICODE`` is the
# default in Python 3 for ``str`` patterns; named explicitly for clarity.
_WORD_TOKEN_RE = re.compile(
    r"[^\W\d_]+(?:'[^\W\d_]+)*",
    re.UNICODE,
)


def tokenize_words(text: str) -> List[str]:
    """Return lowercased Unicode word tokens.

    >>> tokenize_words("Café au lait — don't stop!")
    ['café', 'au', 'lait', "don't", 'stop']
    """
    if not text:
        return []
    return _WORD_TOKEN_RE.findall(text.lower())


def ensure_tokens_word(context, text: Optional[str] = None) -> List[str]:
    """Return ``context.tokens_word``, computing it once on first access.

    Falls back to the legacy ``context.tokens`` field for backward
    compatibility, then to a fresh tokenization. The result is cached on
    ``context.tokens_word`` so subsequent extractors in the same request
    do not re-tokenize.

    Lazy initialisation here means a feature can call this even when the
    pipeline did not pre-populate it — at most one tokenization per
    request still holds.
    """
    cached = getattr(context, "tokens_word", None)
    if cached is not None:
        return cached

    legacy = getattr(context, "tokens", None)
    if legacy is not None:
        try:
            context.tokens_word = list(legacy)
        except Exception:
            context.tokens_word = legacy
        return context.tokens_word

    src = text if text is not None else getattr(context, "text", "") or ""
    context.tokens_word = tokenize_words(src)
    return context.tokens_word
