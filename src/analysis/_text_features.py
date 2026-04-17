"""
File Name: _text_features.py
Module: Analysis - Shared Text Feature Utilities

Description:
    Reusable helpers for token extraction, frequency counting, and phrase/term
    matching used across the analysis sub-package.  Centralising these helpers
    eliminates duplicated Counter-rebuild patterns and replaces naive substring
    checks (``phrase in text_lower``) with word-boundary-aware matching.

    Phrase matching strategy
    ------------------------
    Two strategies are provided:

    1. **Baseline** (always available): Uses compiled regular expressions with
       ``\\b`` word-boundary anchors for single tokens and ``(?<!\\w)…(?!\\w)``
       lookarounds for multi-word phrases.  This avoids false positives such as
       matching "war" inside "Warsaw" or "award".

    2. **Aho-Corasick acceleration** (optional): If the ``ahocorasick`` package
       is installed the automaton is built once per call and provides O(n+m)
       matching for large lexicons.  Results still go through word-boundary
       validation, so correctness is preserved.  Falls back gracefully to the
       baseline when the package is absent.

Usage
-----
::

    from src.analysis._text_features import (
        extract_alpha_lemmas,
        build_counter,
        term_ratio,
        phrase_match_count,
    )
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Collection, List, Set

from spacy.tokens import Doc


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


def extract_alpha_lemmas(doc: Doc) -> List[str]:
    """Return lowercase lemmas for every alpha token in *doc*.

    This is the canonical token extraction pattern used across all analysis
    modules to ensure a consistent denominator for ratio features.

    Args:
        doc: A processed spaCy :class:`~spacy.tokens.Doc`.

    Returns:
        List of ``token.lemma_.lower()`` strings for tokens where
        ``token.is_alpha`` is ``True``.
    """
    return [token.lemma_.lower() for token in doc if token.is_alpha]


# ---------------------------------------------------------------------------
# Counter helpers
# ---------------------------------------------------------------------------


def build_counter(tokens: List[str]) -> Counter:
    """Wrap :class:`collections.Counter` for a list of string tokens.

    Args:
        tokens: Pre-extracted token list (e.g. from :func:`extract_alpha_lemmas`).

    Returns:
        A :class:`~collections.Counter` mapping token → frequency.
    """
    return Counter(tokens)


def word_count(doc: Doc) -> int:
    """Return the number of alpha tokens in *doc*.

    Used as a consistent denominator across ratio features so that all modules
    use the same definition of "document length".

    Args:
        doc: A processed spaCy :class:`~spacy.tokens.Doc`.

    Returns:
        Count of tokens where ``token.is_alpha`` is ``True``.
    """
    return sum(1 for token in doc if token.is_alpha)


# ---------------------------------------------------------------------------
# Term-ratio helper
# ---------------------------------------------------------------------------


def term_ratio(
    token_counts: Counter,
    n_tokens: int,
    lexicon: Collection[str],
) -> float:
    """Compute the ratio of lexicon hits to total alpha-token count.

    Args:
        token_counts: Counter built from :func:`build_counter`.
        n_tokens:     Total number of alpha tokens (``len(tokens)``).
        lexicon:      Collection of target terms/lemmas to look up.

    Returns:
        ``hits / n_tokens`` as ``float``, or ``0.0`` when *n_tokens* is zero.
    """
    if n_tokens == 0:
        return 0.0
    hits = sum(token_counts[t] for t in lexicon if t in token_counts)
    return float(hits / n_tokens)


# ---------------------------------------------------------------------------
# Phrase matching
# ---------------------------------------------------------------------------


def phrase_match_count(
    text_lower: str,
    phrases: Collection[str],
    *,
    word_boundary: bool = True,
) -> int:
    """Count how many *phrases* appear in *text_lower* as whole words.

    When *word_boundary* is ``True`` (default) the function avoids partial-word
    false positives by using regex word-boundary anchors.  When ``False`` it
    falls back to plain substring search (faster but less accurate).

    If the optional ``ahocorasick`` package is installed and *word_boundary* is
    ``True``, the Aho-Corasick automaton is used for O(n+m) scanning of large
    lexicons (still validated with boundary anchors afterwards).

    Args:
        text_lower:    Input text already converted to lower-case.
        phrases:       Iterable of phrases/terms to search for.
        word_boundary: Whether to enforce word-boundary matching.

    Returns:
        Count of distinct phrases that match in *text_lower*.
    """
    if not text_lower or not phrases:
        return 0

    if not word_boundary:
        return sum(1 for phrase in phrases if phrase in text_lower)

    # Try Aho-Corasick acceleration when available
    try:
        import ahocorasick  # type: ignore[import]
        return _phrase_match_aho(text_lower, phrases, ahocorasick)
    except ImportError:
        pass

    return _phrase_match_baseline(text_lower, phrases)


def _phrase_match_baseline(
    text_lower: str,
    phrases: Collection[str],
) -> int:
    """Word-boundary regex matching (no external dependencies)."""
    count = 0
    for phrase in phrases:
        if " " in phrase:
            # Multi-word: lookaround anchors prevent embedding in other words
            pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        else:
            pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text_lower):
            count += 1
    return count


def _phrase_match_aho(
    text_lower: str,
    phrases: Collection[str],
    ahocorasick_mod: object,
) -> int:  # pragma: no cover – only exercised when ahocorasick is installed
    """Aho-Corasick accelerated matching with word-boundary post-validation."""
    A = ahocorasick_mod.Automaton()  # type: ignore[attr-defined]
    for idx, phrase in enumerate(phrases):
        A.add_word(phrase, (idx, phrase))
    A.make_automaton()

    seen: Set[str] = set()
    count = 0
    for _, (_, phrase) in A.iter(text_lower):
        if phrase in seen:
            continue
        if " " in phrase:
            pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        else:
            pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, text_lower):
            count += 1
            seen.add(phrase)
    return count
