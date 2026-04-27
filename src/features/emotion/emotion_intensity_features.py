# src/features/emotion/emotion_intensity_features.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

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

        # Single-text path delegates to the batched implementation so
        # the .cpu().numpy() copy and softmax accounting are owned in
        # exactly one place (audit fix §6.2).
        batched = self._transformer_emotions_batch([text])
        return batched[0]

    # -----------------------------------------------------

    def _transformer_emotions_batch(
        self, texts: List[str]
    ) -> List[Dict[str, float]]:
        """Batched HF inference. Audit fix §6.2 — the single-sample
        ``_transformer_emotions`` path issued one ``model(**inputs)`` +
        one ``.cpu().numpy()`` round-trip *per* document, which made
        ``extract_batch`` linear in batch size with the worst possible
        constant. This batched variant tokenizes the whole list once,
        runs a single forward pass, and copies the full
        ``(B, num_labels)`` softmax matrix back to host memory in one
        ``.cpu().numpy()`` call.
        """
        if not texts:
            return []

        empty_default = [
            {emotion: 0.0 for emotion in EMOTION_LABELS} for _ in texts
        ]

        if not TRANSFORMER_AVAILABLE or _tokenizer is None or _model is None:
            return empty_default

        inputs = _tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = _model(**inputs)

        # ONE host copy for the entire batch.
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        results: List[Dict[str, float]] = []
        for row in probs:
            scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
            for label, prob in zip(TRANSFORMER_LABELS, row):
                if label in scores:
                    scores[label] = float(prob)
            results.append(scores)

        return results

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
            t_scores_dict = self._transformer_emotions(text)
            t_scores = np.array(
                [t_scores_dict[e] for e in EMOTION_LABELS],
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
            return self._empty()

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

            # Audit fix §11 — explicit indicator. Downstream models
            # were unable to tell whether a near-zero intensity row
            # came from a genuinely flat document or from the
            # transformer being unavailable (lexicon-only fallback).
            "emotion_transformer_available": (
                1.0 if TRANSFORMER_AVAILABLE else 0.0
            ),
        }

    # -----------------------------------------------------
    # BATCH (audit fix §6.2)
    #
    # Default ``BaseFeature.extract_batch`` calls ``extract`` per sample,
    # which in turn called the transformer once per text. This override
    # tokenizes + softmaxes the entire batch in a single forward pass and
    # only does the cheap lexicon / stats work in the per-sample loop.
    # -----------------------------------------------------

    def extract_batch(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:

        if not contexts:
            return []

        # Pre-tokenise + collect batch text in one pass.
        texts: List[str] = []
        token_lists = []
        active_idx: List[int] = []

        results: List[Dict[str, float]] = [self._empty() for _ in contexts]

        for i, ctx in enumerate(contexts):
            text = (ctx.text or "").strip()
            if not text:
                continue
            tokens = ensure_tokens_word(ctx, text)
            token_lists.append(tokens)
            texts.append(text)
            active_idx.append(i)

        if not texts:
            return results

        # ONE transformer forward pass for the whole batch.
        if TRANSFORMER_AVAILABLE:
            t_batch = self._transformer_emotions_batch(texts)
        else:
            t_batch = [
                {emotion: 0.0 for emotion in EMOTION_LABELS}
                for _ in texts
            ]

        for j, dst_i in enumerate(active_idx):
            text = texts[j]
            tokens = token_lists[j]

            counts, hits, n_lex_tokens = _lexicon_emotions(tokens)
            lex_scores = (
                np.array(
                    [counts[e] for e in EMOTION_LABELS], dtype=np.float32
                )
                / (hits + EPS)
                if hits > 0
                else np.zeros(len(EMOTION_LABELS), dtype=np.float32)
            )

            t_scores = np.array(
                [t_batch[j][e] for e in EMOTION_LABELS], dtype=np.float32
            )

            alpha = 0.7 if t_scores.sum() > 0 else 0.0
            scores = alpha * t_scores + (1 - alpha) * lex_scores
            total = scores.sum()
            if total > 0:
                scores = scores / total

            token_count = max(n_lex_tokens, 1)
            coverage = hits / token_count

            max_val = float(np.max(scores))
            mean_val = float(np.mean(scores))
            std_val = float(np.std(scores))
            range_val = float(np.max(scores) - np.min(scores))
            l2_intensity = float(np.linalg.norm(scores))
            entropy = normalized_entropy(scores)

            results[dst_i] = {
                "emotion_intensity_max": self._safe(max_val),
                "emotion_intensity_mean": self._safe(mean_val),
                "emotion_intensity_std": self._safe(std_val),
                "emotion_intensity_range": self._safe(range_val),
                "emotion_intensity_l2": self._safe(l2_intensity),
                "emotion_intensity_entropy": self._safe(entropy),
                "emotion_coverage": self._safe(coverage),
                "emotion_transformer_available": (
                    1.0 if TRANSFORMER_AVAILABLE else 0.0
                ),
            }

        return results

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        # Audit fix §11 — fixed-key sentinel so a degenerate input
        # never produces a missing-column row downstream.
        return {
            "emotion_intensity_max": 0.0,
            "emotion_intensity_mean": 0.0,
            "emotion_intensity_std": 0.0,
            "emotion_intensity_range": 0.0,
            "emotion_intensity_l2": 0.0,
            "emotion_intensity_entropy": 0.0,
            "emotion_coverage": 0.0,
            "emotion_transformer_available": (
                1.0 if TRANSFORMER_AVAILABLE else 0.0
            ),
        }

    def _fallback(self) -> Dict[str, float]:
        return self._empty()

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))