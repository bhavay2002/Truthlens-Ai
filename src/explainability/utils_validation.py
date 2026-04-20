from __future__ import annotations

import math
from typing import Sequence


def validate_tokens_scores(tokens: Sequence[str], scores: Sequence[float]) -> None:
    """
    Validate token-score pairs used across explainability modules.
    """
    if not isinstance(tokens, Sequence) or isinstance(tokens, (str, bytes)):
        raise TypeError("tokens must be a sequence of strings")

    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        raise TypeError("scores must be a sequence of numeric values")

    if len(tokens) == 0 or len(scores) == 0:
        raise ValueError("Empty tokens or scores")

    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must match length")

    for token in tokens:
        if not isinstance(token, str):
            raise TypeError("all tokens must be strings")

    for score in scores:
        if not isinstance(score, (int, float)):
            raise TypeError("all scores must be numeric")
        if not math.isfinite(float(score)):
            raise ValueError("scores must be finite values")
