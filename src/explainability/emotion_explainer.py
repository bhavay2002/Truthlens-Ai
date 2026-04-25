from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.features.emotion.emotion_schema import EMOTION_TERMS

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# LOOKUP
# =========================================================

WORD_TO_EMOTION = {
    w.lower(): e
    for e, words in EMOTION_TERMS.items()
    for w in words
}


INTENSIFIERS = {
    "very","extremely","highly","incredibly","really","so","too",
    "completely","totally","deeply","strongly"
}


# =========================================================
# DATA MODEL
# =========================================================

@dataclass
class EmotionExplanation:
    tokens: List[str]

    lexicon_intensity: List[float]
    gradient_importance: List[float]

    fused_importance: List[float]

    sentence_scores: List[Dict[str, float]]
    emotion_distribution: Dict[str, float]

    intensity_score: float


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text: str):
    return re.findall(r"\b[a-z]+\b", text.lower())


def sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


# =========================================================
# LEXICON
# =========================================================

def compute_lexicon(tokens):

    values = []

    for t in tokens:
        val = 0.0

        if t in WORD_TO_EMOTION:
            val += 1.0

        if t in INTENSIFIERS:
            val += 0.5

        values.append(val)

    return np.asarray(values, dtype=float)


# =========================================================
# GRADIENT (SIMPLIFIED SAFE)
# =========================================================

def compute_gradients(model, tokenizer, text):

    inputs = tokenizer(text, return_tensors="pt")

    emb = model.get_input_embeddings()(inputs["input_ids"]).detach()
    emb.requires_grad_(True)

    out = model(inputs_embeds=emb)
    out.logits.max().backward()

    grads = emb.grad.abs().sum(dim=-1)[0].detach().cpu().numpy()

    return grads


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0)
    return x / (np.sum(x) + EPS)


# =========================================================
# FUSION
# =========================================================

def fuse(lexicon, gradients):

    lexicon = normalize(lexicon)
    gradients = normalize(gradients) if gradients is not None else None

    if gradients is None:
        return lexicon

    return normalize(0.6 * lexicon + 0.4 * gradients)


# =========================================================
# SENTENCE LEVEL
# =========================================================

def compute_sentence_scores(text):

    results = []

    for s in sentences(text):
        toks = tokenize(s)
        vals = compute_lexicon(toks)

        score = float(np.mean(vals)) if len(vals) else 0.0

        results.append({
            "sentence": s,
            "emotion_intensity": score
        })

    return results


# =========================================================
# DISTRIBUTION
# =========================================================

def emotion_distribution(tokens):

    counts = {}

    for t in tokens:
        if t in WORD_TO_EMOTION:
            e = WORD_TO_EMOTION[t]
            counts[e] = counts.get(e, 0) + 1

    total = sum(counts.values()) + EPS

    return {k: v / total for k, v in counts.items()}


# =========================================================
# MAIN
# =========================================================

def explain_emotion(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:

    tokens = tokenize(text)

    lexicon_vals = compute_lexicon(tokens)

    gradients = None
    if model and tokenizer:
        try:
            gradients = compute_gradients(model, tokenizer, text)
        except Exception:
            logger.warning("Gradient failed")

    fused = fuse(lexicon_vals, gradients)

    return EmotionExplanation(
        tokens=tokens,

        lexicon_intensity=normalize(lexicon_vals).tolist(),
        gradient_importance=normalize(gradients).tolist() if gradients is not None else [],

        fused_importance=fused.tolist(),

        sentence_scores=compute_sentence_scores(text),
        emotion_distribution=emotion_distribution(tokens),

        intensity_score=float(np.mean(fused)),
    ).__dict__