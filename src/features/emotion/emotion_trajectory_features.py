"""
File Name: emotion_trajectory_features.py
Module: Feature Engineering - Emotion Trajectory Features
Description:
    Computes features describing how emotional signals evolve across
    the progression of a text. The module splits text into segments
    (sentences or fixed windows) and estimates emotion scores per segment.
    It then derives trajectory statistics such as trend, volatility,
    peak position, and emotional shifts.

    Transformer-based emotion models are used when available. Otherwise,
    a lexicon-based fallback is applied to estimate segment-level emotions.

Dependencies:
    dataclasses
    typing
    logging
    numpy
    re
    transformers (optional)
    torch (optional)

Inputs:
    FeatureContext containing input text

Outputs:
    Dict[str, float] representing emotion trajectory statistics
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Optional Transformer Emotion Model
# ---------------------------------------------------------------------

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.eval()

    TRANSFORMER_AVAILABLE = True

except Exception:  # noqa: BLE001
    TRANSFORMER_AVAILABLE = False
    _tokenizer = None
    _model = None
    logger.warning("Transformer emotion model unavailable. Using lexicon fallback.")

# ---------------------------------------------------------------------
# Lexicon fallback
# ---------------------------------------------------------------------

LEXICON = {
    "anger": {"angry", "rage", "furious", "hate"},
    "fear": {"fear", "terror", "panic", "scared"},
    "joy": {"happy", "joy", "delight", "smile"},
    "sadness": {"sad", "cry", "grief", "sorrow"},
}

EMOTION_VOCAB = set().union(*LEXICON.values())


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _lexicon_score(text: str) -> float:
    """Compute simple emotional intensity score using lexicon."""
    tokens = re.findall(r"\b\w+\b", text.lower())

    if not tokens:
        return 0.0

    emotion_count = sum(1 for t in tokens if t in EMOTION_VOCAB)

    return emotion_count / len(tokens)


def _transformer_score(text: str) -> float:
    """Compute emotional intensity using transformer probabilities."""
    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()

    return float(np.max(probs) - np.mean(probs))


@dataclass
@register_feature
class EmotionTrajectoryFeatures(BaseFeature):
    """
    Captures how emotion evolves through the text.

    Output features include:
        emotion_traj_mean
        emotion_traj_std
        emotion_traj_slope
        emotion_traj_peak_position
        emotion_traj_volatility
    """

    name: str = "emotion_trajectory_features"
    description: str = "Emotion evolution and trajectory statistics"

    def _segment_scores(self, text: str) -> List[float]:
        sentences = _split_sentences(text)

        if not sentences:
            return [0.0]

        scores: List[float] = []

        for s in sentences:
            if TRANSFORMER_AVAILABLE:
                score = _transformer_score(s)
            else:
                score = _lexicon_score(s)

            scores.append(score)

        return scores

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        scores = np.array(self._segment_scores(context.text), dtype=np.float32)

        if len(scores) == 1:
            scores = np.append(scores, scores)

        mean_val = float(np.mean(scores))
        std_val = float(np.std(scores))

        # Linear trend
        x = np.arange(len(scores))
        slope = float(np.polyfit(x, scores, 1)[0])

        peak_position = float(np.argmax(scores) / len(scores))

        volatility = float(np.mean(np.abs(np.diff(scores))))

        features: Dict[str, float] = {
            "emotion_traj_mean": mean_val,
            "emotion_traj_std": std_val,
            "emotion_traj_slope": slope,
            "emotion_traj_peak_position": peak_position,
            "emotion_traj_volatility": volatility,
        }

        logger.debug(
            "Emotion trajectory features extracted | mean=%.4f slope=%.4f",
            mean_val,
            slope,
        )

        return features