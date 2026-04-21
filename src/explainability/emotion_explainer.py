from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    EMOTION_TERMS,
)

logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# Reverse lookup (case-normalized)
# -------------------------------------------------------------

WORD_TO_EMOTION: Dict[str, str] = {}

for emotion, words in EMOTION_TERMS.items():
    for word in words:
        WORD_TO_EMOTION[word.lower()] = emotion


# -------------------------------------------------------------
# Intensifier vocabulary (deduplicated)
# -------------------------------------------------------------

INTENSIFIER_ADVERBS = set({
    "very","extremely","highly","incredibly","remarkably","exceptionally",
    "unbelievably","extraordinarily","immensely","intensely","profoundly","dramatically",
    "really","so","too","quite","pretty","fairly","rather","especially","particularly",
    "notably","significantly",
    "totally","absolutely","completely","entirely","fully","perfectly","truly","genuinely","seriously",
    "deeply","strongly","wildly","terribly","awfully","desperately","badly",
    "massively","hugely","greatly","vastly","considerably","substantially","overwhelmingly",
    "utterly","decidedly","undeniably","indisputably","unquestionably",
})


# -------------------------------------------------------------
# Data container
# -------------------------------------------------------------

@dataclass
class EmotionExplanation:
    emotion_tokens: List[Dict[str, Any]]
    emotion_heatmap: List[Dict[str, Any]]
    sentence_heatmap: List[Dict[str, Any]]
    gradient_attribution: List[Dict[str, Any]]
    heatmap_matrix: List[List[float]]
    ui_highlights: List[Dict[str, Any]]


# -------------------------------------------------------------
# Tokenization
# -------------------------------------------------------------

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b[a-z]+\b", text.lower())


def tokenize_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


# -------------------------------------------------------------
# Emotion detection
# -------------------------------------------------------------

def detect_emotion_tokens(tokens: List[str]) -> List[Dict[str, Any]]:
    return [
        {"token": t, "emotion": WORD_TO_EMOTION[t], "position": i}
        for i, t in enumerate(tokens)
        if t in WORD_TO_EMOTION
    ]


# -------------------------------------------------------------
# Token intensity
# -------------------------------------------------------------

def compute_token_intensity(tokens: List[str]) -> List[Dict[str, Any]]:
    heatmap = []

    for i, t in enumerate(tokens):
        intensity = 0.0

        if t in WORD_TO_EMOTION:
            intensity += 1.0

        if t in INTENSIFIER_ADVERBS:
            intensity += 0.5

        heatmap.append({
            "token": t,
            "intensity": round(float(intensity), 3),
            "position": i,
        })

    return heatmap


# -------------------------------------------------------------
# Sentence heatmap
# -------------------------------------------------------------

def compute_sentence_heatmap(text: str) -> List[Dict[str, Any]]:
    results = []

    for sentence in tokenize_sentences(text):
        tokens = tokenize_words(sentence)
        scores = compute_token_intensity(tokens)

        total = sum(x["intensity"] for x in scores)
        normalized = total / max(len(tokens), 1)

        results.append({
            "sentence": sentence,
            "emotion_intensity": round(float(normalized), 4),
        })

    return results


# -------------------------------------------------------------
# Device helper
# -------------------------------------------------------------

def _resolve_device(model: Any) -> Optional[torch.device]:
    try:
        return next(model.parameters()).device
    except Exception:
        return None


# -------------------------------------------------------------
# Proper Integrated Gradients (multi-step)
# -------------------------------------------------------------

def compute_integrated_gradients(
    model: Any,
    tokenizer: Any,
    text: str,
    steps: int = 16,
) -> List[Dict[str, Any]]:
    """
    Safe Integrated Gradients:
    - No model.zero_grad()
    - No parameter grad accumulation (uses autograd.grad)
    - Model runs in eval() (restored afterward)
    """

    was_training = model.training
    model.eval()

    try:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        embedding_layer = model.get_input_embeddings()

        with torch.no_grad():
            input_emb = embedding_layer(input_ids)
            base_emb = torch.zeros_like(input_emb)

        total_grads = torch.zeros_like(input_emb)

        alphas = torch.linspace(0.0, 1.0, steps, device=input_emb.device)

        for alpha in alphas:
            emb = (base_emb + alpha * (input_emb - base_emb)).detach()
            emb.requires_grad_(True)

            outputs = model(
                inputs_embeds=emb,
                attention_mask=attention_mask,
            )

            target = outputs.logits.max()

            (grad,) = torch.autograd.grad(
                outputs=target,
                inputs=emb,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

            total_grads += grad

        avg_grads = total_grads / max(steps, 1)
        attributions = (input_emb - base_emb) * avg_grads

        scores = attributions.abs().sum(dim=-1)[0].detach().cpu().numpy()
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        scores = np.maximum(scores, 0.0)
        total = float(scores.sum())
        if total > 0:
            scores /= total

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu())

        return [
            {"token": tok, "importance": float(score)}
            for tok, score in zip(tokens, scores)
        ]

    finally:
        if was_training:
            model.train()


# -------------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------------

def generate_heatmap_matrix(tokens, heatmap):
    return [[float(x["intensity"])] for x in heatmap]


def generate_ui_highlights(heatmap):
    return [
        {"token": x["token"], "strength": x["intensity"]}
        for x in heatmap if x["intensity"] > 0
    ]


# -------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------

def explain_emotion(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    tokens = tokenize_words(text)

    emotion_tokens = detect_emotion_tokens(tokens)
    emotion_heatmap = compute_token_intensity(tokens)
    sentence_heatmap = compute_sentence_heatmap(text)

    gradient_attr: List[Dict[str, Any]] = []

    if model and tokenizer:
        try:
            gradient_attr = compute_integrated_gradients(model, tokenizer, text)
        except Exception as exc:
            logger.warning("Gradient attribution failed: %s", exc)

    explanation = EmotionExplanation(
        emotion_tokens=emotion_tokens,
        emotion_heatmap=emotion_heatmap,
        sentence_heatmap=sentence_heatmap,
        gradient_attribution=gradient_attr,
        heatmap_matrix=generate_heatmap_matrix(tokens, emotion_heatmap),
        ui_highlights=generate_ui_highlights(emotion_heatmap),
    )

    return explanation.__dict__