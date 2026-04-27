# src/features/emotion/emotion_intensity_features.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import normalized_entropy
from src.features.base.tokenization import ensure_tokens_word

from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    EMOTION_TERMS,
)

logger = logging.getLogger(__name__)

EPS = 1e-8
MAX_CLIP = 1.0


# =========================================================
# TRANSFORMER SETUP (SAFE)
# =========================================================

TRANSFORMER_AVAILABLE = False
_tokenizer = None
_model = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    _model.eval()

    TRANSFORMER_AVAILABLE = True

    TRANSFORMER_LABELS = [
        "anger", "disgust", "fear",
        "joy", "neutral", "sadness", "surprise"
    ]

except Exception as e:
    logger.warning(
        "Transformer not available, using lexicon fallback | %s", str(e)
    )


# =========================================================
# REVERSE LOOKUP (LEXICON)
# =========================================================

WORD_TO_EMOTION = {
    word: emotion
    for emotion, words in EMOTION_TERMS.items()
    for word in words
}


# =========================================================
# LEXICON EMOTION DETECTOR
# =========================================================

def _lexicon_emotions(tokens):

    counts = {emotion: 0 for emotion in EMOTION_LABELS}

    for token in tokens:
        emo = WORD_TO_EMOTION.get(token)
        if emo:
            counts[emo] += 1

    total_hits = sum(counts.values())
    total_tokens = len(tokens)

    return counts, total_hits, total_tokens


# =========================================================
# FEATURE EXTRACTOR
# =========================================================

@dataclass
@register_feature
class EmotionIntensityFeatures(BaseFeature):

    name: str = "emotion_intensity_features"
    group: str = "emotion"
    description: str = "Robust emotion intensity + hybrid modeling"

    # -----------------------------------------------------

    def _transformer_emotions(self, text: str) -> Dict[str, float]:

        if not TRANSFORMER_AVAILABLE or _tokenizer is None or _model is None:
            return {emotion: 0.0 for emotion in EMOTION_LABELS}

        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = _model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()

        scores = {emotion: 0.0 for emotion in EMOTION_LABELS}

        for label, prob in zip(TRANSFORMER_LABELS, probs):
            if label in scores:
                scores[label] = float(prob)

        return scores

    # -----------------------------------------------------

    def _hybrid_emotions(self, text: str, tokens):

        # -------- Lexicon --------
        counts, hits, n_lex_tokens = _lexicon_emotions(tokens)

        lex_scores = (
            np.array([counts[e] for e in EMOTION_LABELS], dtype=np.float32)
            / (hits + EPS)
            if hits > 0 else np.zeros(len(EMOTION_LABELS))
        )

        # -------- Transformer --------
        if TRANSFORMER_AVAILABLE:
            t_scores = np.array(
                list(self._transformer_emotions(text).values()),
                dtype=np.float32,
            )
        else:
            t_scores = np.zeros(len(EMOTION_LABELS))

        # -------- Adaptive fusion --------
        alpha = 0.7 if t_scores.sum() > 0 else 0.0

        scores = alpha * t_scores + (1 - alpha) * lex_scores

        total = scores.sum()
        if total > 0:
            scores = scores / total

        return scores, hits, n_lex_tokens

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        tokens = ensure_tokens_word(context, text)
        scores, hits, token_count = self._hybrid_emotions(text, tokens)

        token_count = max(token_count, 1)

        # -------------------------
        # Coverage
        # -------------------------

        coverage = hits / token_count

        # -------------------------
        # Core statistics
        # -------------------------

        max_val = float(np.max(scores))
        mean_val = float(np.mean(scores))
        std_val = float(np.std(scores))
        range_val = float(np.max(scores) - np.min(scores))

        # Strong intensity signal
        l2_intensity = float(np.linalg.norm(scores))

        # -------------------------
        # Entropy (normalized)
        # -------------------------

        entropy = normalized_entropy(scores)

        # -------------------------
        # Output
        # -------------------------

        return {
            "emotion_intensity_max": self._safe(max_val),
            "emotion_intensity_mean": self._safe(mean_val),
            "emotion_intensity_std": self._safe(std_val),
            "emotion_intensity_range": self._safe(range_val),

            "emotion_intensity_l2": self._safe(l2_intensity),
            "emotion_intensity_entropy": self._safe(entropy),

            "emotion_coverage": self._safe(coverage),
        }

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))