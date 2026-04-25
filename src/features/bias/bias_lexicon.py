from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np

from src.features.base.base_feature import FeatureContext
from src.features.bias.bias_lexicon_features import BiasLexiconFeatures


# ---------------------------------------------------------
# Result schema
# ---------------------------------------------------------

@dataclass
class BiasResult:
    bias_score: float
    media_bias: str
    bias_intensity: float
    bias_entropy: float
    bias_subjectivity: float
    bias_certainty: float
    biased_tokens: List[str]
    sentence_heatmap: List[Dict[str, Any]]


# ---------------------------------------------------------
# Tokenization
# ---------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def compute_bias_features(text: str) -> BiasResult:

    extractor = BiasLexiconFeatures()

    tokens = _TOKEN_PATTERN.findall(text.lower())

    context = FeatureContext(
        text=text,
        tokens=tokens,
    )

    features = extractor.extract(context)

    # -------------------------
    # Core signals
    # -------------------------

    bias_score = features.get("bias_density", 0.0)
    intensity = features.get("bias_intensity", 0.0)
    entropy = features.get("bias_entropy", 0.0)
    subjectivity = features.get("bias_subjectivity", 0.0)
    certainty = features.get("bias_certainty", 0.0)

    # -------------------------
    # Classification (IMPROVED)
    # -------------------------

    composite = 0.5 * bias_score + 0.3 * intensity + 0.2 * subjectivity

    if composite < 0.05:
        media_bias = "center"
    elif composite < 0.15:
        media_bias = "lean"
    else:
        media_bias = "strong"

    # -------------------------
    # Biased tokens (derived from features)
    # -------------------------

    biased_tokens = [
        t for t in tokens
        if features.get("bias_eval_ratio", 0) > 0
        or features.get("bias_assertive_ratio", 0) > 0
    ]

    # -------------------------
    # Sentence heatmap (CONSISTENT)
    # -------------------------

    sentences = [
        s.strip() for s in re.split(r"[.!?]+", text) if s.strip()
    ]

    sentence_heatmap = []

    for sent in sentences:

        sent_tokens = _TOKEN_PATTERN.findall(sent.lower())

        if not sent_tokens:
            score = 0.0
        else:
            # reuse extractor logic locally
            sent_ctx = FeatureContext(text=sent, tokens=sent_tokens)
            sent_feat = extractor.extract(sent_ctx)

            score = sent_feat.get("bias_density", 0.0)

        sentence_heatmap.append({
            "sentence": sent,
            "bias_score": round(score, 4),
        })

    return BiasResult(
        bias_score=round(bias_score, 4),
        media_bias=media_bias,
        bias_intensity=round(intensity, 4),
        bias_entropy=round(entropy, 4),
        bias_subjectivity=round(subjectivity, 4),
        bias_certainty=round(certainty, 4),
        biased_tokens=biased_tokens,
        sentence_heatmap=sentence_heatmap,
    )