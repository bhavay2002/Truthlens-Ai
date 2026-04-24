# src/analysis/_text_features.py

from __future__ import annotations

import re
from collections import Counter
from typing import Collection, List

from spacy.tokens import Doc

# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------


def extract_alpha_lemmas(doc: Doc) -> List[str]:
    return [token.lemma_.lower() for token in doc if token.is_alpha]


# ---------------------------------------------------------------------------
# Counter helpers
# ---------------------------------------------------------------------------


def build_counter(tokens: List[str]) -> Counter:
    return Counter(tokens)


def word_count_from_tokens(tokens: List[str]) -> int:
    return len(tokens)


# ---------------------------------------------------------------------------
# Term-ratio helper
# ---------------------------------------------------------------------------


def term_ratio(
    token_counts: Counter,
    n_tokens: int,
    lexicon: Collection[str],
) -> float:
    if n_tokens == 0:
        return 0.0

    hits = sum(token_counts.get(t, 0) for t in lexicon)
    return float(hits / n_tokens)


# ---------------------------------------------------------------------------
# Phrase matching (OPTIMIZED)
# ---------------------------------------------------------------------------

_REGEX_CACHE = {}


def _compile_patterns(phrases):
    key = tuple(sorted(phrases))

    if key in _REGEX_CACHE:
        return _REGEX_CACHE[key]

    compiled = []
    for phrase in phrases:
        if " " in phrase:
            pattern = re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")
        else:
            pattern = re.compile(r"\b" + re.escape(phrase) + r"\b")
        compiled.append(pattern)

    _REGEX_CACHE[key] = compiled
    return compiled


def phrase_match_count(
    text_lower: str,
    phrases: Collection[str],
    *,
    word_boundary: bool = True,
) -> int:

    if not text_lower or not phrases:
        return 0

    if not word_boundary:
        return sum(1 for phrase in phrases if phrase in text_lower)

    patterns = _compile_patterns(phrases)

    return sum(1 for pattern in patterns if pattern.search(text_lower))


def normalize_lexicon_terms(terms: set[str]) -> set[str]:
    return {t.replace("_", " ").strip().lower() for t in terms if t}