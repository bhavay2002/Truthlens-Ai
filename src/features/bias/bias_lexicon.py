"""
Wrapper module providing the compute_bias_features interface expected by the API.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Any

from src.features.base.base_feature import FeatureContext
from src.features.bias.bias_lexicon_features import BiasLexiconFeatures


_TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

_BIAS_TERMS = {
    "allegedly", "reportedly", "claims", "insists", "radical", "extreme",
    "massive", "shocking", "outrageous", "disgusting", "terrible", "horrible",
    "wonderful", "amazing", "incredible", "devastating", "alarming",
    "explosive", "bombshell", "stunning", "catastrophic", "disastrous",
}


@dataclass
class BiasResult:
    bias_score: float
    media_bias: str
    biased_tokens: List[str]
    sentence_heatmap: List[Dict[str, Any]]


def compute_bias_features(text: str) -> BiasResult:
    """Compute bias features for the given text."""
    extractor = BiasLexiconFeatures()
    context = FeatureContext(text=text)
    features = extractor.extract(context)

    bias_score = features.get("bias_lexicon_density", 0.0)

    tokens = _TOKEN_PATTERN.findall(text.lower())
    biased_tokens = [t for t in tokens if t in _BIAS_TERMS]

    if bias_score < 0.05:
        media_bias = "center"
    elif bias_score < 0.15:
        media_bias = "lean"
    else:
        media_bias = "strong"

    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_heatmap = []
    for sent in sentences:
        sent_tokens = _TOKEN_PATTERN.findall(sent.lower())
        sent_bias_count = sum(1 for t in sent_tokens if t in _BIAS_TERMS)
        sent_score = sent_bias_count / max(len(sent_tokens), 1)
        sentence_heatmap.append({"sentence": sent, "bias_score": round(sent_score, 4)})

    return BiasResult(
        bias_score=round(bias_score, 4),
        media_bias=media_bias,
        biased_tokens=biased_tokens,
        sentence_heatmap=sentence_heatmap,
    )
