"""
File Name: bias_explainer.py
Module: Explainability - Bias Analysis
Description:
    Provides interpretability utilities for bias detection outputs within
    the TruthLens AI system. The module analyzes text using lexical bias
    signals, SHAP token importance, integrated gradients, and transformer
    attention scores to generate human-readable bias explanations.

Purpose:
    Provide interpretable explanations for bias predictions using
    multiple attribution techniques.

Capabilities:
    - SHAP token attribution
    - Integrated gradients
    - Attention attribution
    - Lexicon bias fallback
    - Sentence bias scoring
    - Subword merging
    - Bias heatmap generation

Dependencies:
    logging
    re
    dataclasses
    typing
    numpy
    torch
    shap (optional)

Inputs:
    model : transformer model
    tokenizer : tokenizer compatible with the model
    text : input text

Outputs:
    Structured explanation dictionary with token importance, sentence bias
    scores, attention weights, and visualization-ready heatmaps.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import shap
except ImportError:  # optional
    shap = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Bias Lexicon Fallback
# ---------------------------------------------------------

BIAS_TERMS = {

    # ----------------------------------------------------
    # Ideological labeling
    # ----------------------------------------------------
    "radical","extremist","far_left","far_right",
    "socialist","communist","fascist","elitist",
    "ultra","hardline","reactionary","authoritarian",

    # ----------------------------------------------------
    # Delegitimization language
    # ----------------------------------------------------
    "corrupt","crooked","rigged","fraud","fraudulent",
    "illegitimate","dishonest","deceptive","manipulative",
    "propaganda","brainwashing","indoctrination",

    # ----------------------------------------------------
    # Media / information delegitimization
    # ----------------------------------------------------
    "fake","fake_news","misleading","fabricated",
    "distorted","biased","partisan","agenda_driven",
    "media_bias","spin","coverup",

    # ----------------------------------------------------
    # Elite vs people rhetoric
    # ----------------------------------------------------
    "elite","establishment","bureaucrat","globalist",
    "oligarch","corporate_elite","power_elite",
    "political_elite","technocrat",

    # ----------------------------------------------------
    # Emotional / loaded descriptors
    # ----------------------------------------------------
    "disgraceful","outrageous","shocking",
    "terrible","horrible","evil","dangerous",
    "disgusting","absurd","ridiculous",

    # ----------------------------------------------------
    # Conspiracy framing
    # ----------------------------------------------------
    "conspiracy","scheme","plot","agenda",
    "hidden_agenda","secret_plan","manipulated",
    "controlled","staged","engineered"
}
# ---------------------------------------------------------
# Data Structures
# ---------------------------------------------------------

@dataclass
class BiasExplanation:

    token_importance: List[Dict[str, Any]]
    integrated_gradients: List[Dict[str, Any]]
    attention_scores: List[Dict[str, Any]]

    biased_tokens: List[str]
    sentence_bias_scores: List[Dict[str, Any]]

    bias_intensity: float
    bias_heatmap: List[Dict[str, Any]]


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------

def tokenize_sentences(text: str) -> List[str]:

    sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if s.strip()]


def normalize_scores(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values

    min_v = values.min()
    max_v = values.max()

    if max_v - min_v < 1e-9:
        return values

    return (values - min_v) / (max_v - min_v)


# ---------------------------------------------------------
# Subword Token Merge
# ---------------------------------------------------------

def merge_subwords(tokens, scores):

    merged_tokens = []
    merged_scores = []

    buffer_token = ""
    buffer_scores = []

    for token, score in zip(tokens, scores):

        if token.startswith("##"):

            buffer_token += token[2:]
            buffer_scores.append(score)

        else:

            if buffer_token:
                merged_tokens.append(buffer_token)
                merged_scores.append(np.mean(buffer_scores))

            buffer_token = token
            buffer_scores = [score]

    if buffer_token:
        merged_tokens.append(buffer_token)
        merged_scores.append(np.mean(buffer_scores))

    return merged_tokens, merged_scores


# ---------------------------------------------------------
# Lexicon Bias Detection
# ---------------------------------------------------------

def compute_lexicon_bias(text: str):

    tokens = re.findall(r"\b[a-z]+\b", text.lower())

    matched = [t for t in tokens if t in BIAS_TERMS]

    unique_tokens = list(set(matched))

    score = len(matched) / max(len(tokens), 1)

    return score, unique_tokens


# ---------------------------------------------------------
# Sentence Bias
# ---------------------------------------------------------

def compute_sentence_bias(text):

    sentences = tokenize_sentences(text)

    results = []

    for sentence in sentences:

        score, tokens = compute_lexicon_bias(sentence)

        results.append(
            {
                "sentence": sentence,
                "bias_score": score,
                "biased_tokens": tokens,
            }
        )

    return results


# ---------------------------------------------------------
# SHAP Attribution
# ---------------------------------------------------------

def compute_shap_importance(model, tokenizer, text):

    if shap is None:
        raise ImportError("SHAP not installed")

    def predict(texts):

        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = model(**encodings)

        return outputs.logits.detach().cpu().numpy()

    explainer = shap.Explainer(predict, tokenizer)

    shap_values = explainer([text])

    tokens = list(shap_values.data[0])
    values = shap_values.values[0]

    values = np.mean(values, axis=-1)

    return [
        {"token": t, "importance": float(v)}
        for t, v in zip(tokens, values)
    ]


# ---------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------

def compute_integrated_gradients(model, tokenizer, text):

    inputs = tokenizer(text, return_tensors="pt")

    model.zero_grad()

    embeddings = model.get_input_embeddings()(inputs["input_ids"])

    embeddings = embeddings.detach().requires_grad_(True)

    outputs = model(inputs_embeds=embeddings)

    target = outputs.logits.max()

    target.backward()

    grads = embeddings.grad.abs().sum(dim=-1)[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return [
        {"token": t, "importance": float(g)}
        for t, g in zip(tokens, grads.detach().cpu().numpy())
    ]


# ---------------------------------------------------------
# Attention Attribution
# ---------------------------------------------------------

def compute_attention_scores(model, tokenizer, text):

    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():

        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions[-1]

    matrix = attentions.mean(dim=1)[0]

    token_scores = matrix.mean(dim=0)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return [
        {"token": t, "attention": float(a)}
        for t, a in zip(tokens, token_scores.detach().cpu().numpy())
    ]


# ---------------------------------------------------------
# Bias Intensity
# ---------------------------------------------------------

def compute_bias_intensity(token_importance):

    values = [abs(t["importance"]) for t in token_importance]

    if not values:
        return 0.0

    return float(np.mean(values))


# ---------------------------------------------------------
# Biased Tokens
# ---------------------------------------------------------

def extract_biased_tokens(token_importance, threshold=0.05):

    return [
        t["token"]
        for t in token_importance
        if abs(t["importance"]) >= threshold
    ]


# ---------------------------------------------------------
# Heatmap
# ---------------------------------------------------------

def generate_bias_heatmap(token_importance):

    return [
        {
            "token": t["token"],
            "bias_strength": abs(float(t["importance"]))
        }
        for t in token_importance
    ]


# ---------------------------------------------------------
# Main Explain Function
# ---------------------------------------------------------

def explain_bias(model, tokenizer, text):

    if not text.strip():
        raise ValueError("text must not be empty")

    integrated_gradients_result: List[Dict[str, Any]] = []

    try:

        shap_importance = compute_shap_importance(model, tokenizer, text)

    except Exception as e:

        logger.warning("SHAP failed: %s", e)

        integrated_gradients_result = compute_integrated_gradients(
            model,
            tokenizer,
            text,
        )

        shap_importance = integrated_gradients_result

    tokens = [t["token"] for t in shap_importance]
    scores = [t["importance"] for t in shap_importance]

    scores = normalize_scores(scores)

    tokens, scores = merge_subwords(tokens, scores)

    token_importance = [
        {"token": t, "importance": float(s)}
        for t, s in zip(tokens, scores)
    ]

    biased_tokens = extract_biased_tokens(token_importance)

    sentence_scores = compute_sentence_bias(text)

    attention_scores = compute_attention_scores(
        model,
        tokenizer,
        text,
    )

    bias_intensity = compute_bias_intensity(token_importance)

    heatmap = generate_bias_heatmap(token_importance)

    if not integrated_gradients_result:
        integrated_gradients_result = compute_integrated_gradients(
            model,
            tokenizer,
            text,
        )

    explanation = BiasExplanation(
        token_importance=token_importance,
        integrated_gradients=integrated_gradients_result,
        attention_scores=attention_scores,
        biased_tokens=biased_tokens,
        sentence_bias_scores=sentence_scores,
        bias_intensity=bias_intensity,
        bias_heatmap=heatmap,
    )

    return explanation.__dict__