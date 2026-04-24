# src/analysis/feature_context.py

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import List

from spacy.tokens import Doc

from src.analysis._text_features import extract_alpha_lemmas


@dataclass
class FeatureContext:
    """
    Shared feature container computed ONCE per document.
    Passed to all analyzers.
    """

    doc: Doc
    text_lower: str
    tokens: List[str]
    token_counts: Counter
    n_tokens: int

    @classmethod
    def from_doc(cls, doc: Doc) -> "FeatureContext":
        tokens = extract_alpha_lemmas(doc)

        return cls(
            doc=doc,
            text_lower=doc.text.lower(),
            tokens=tokens,
            token_counts=Counter(tokens),
            n_tokens=len(tokens),
        )