"""
TruthLens AI
Bias Explainability Engine

Provides explainability for bias predictions using
multiple attribution techniques.

Explanation Methods
-------------------
• SHAP token attribution
• Integrated Gradients
• RoBERTa attention visualization
• Sentence-level bias scoring
• Bias heatmap generation

Outputs
-------
token_importance
biased_token_highlights
sentence_bias_scores
attention_scores
bias_heatmap
"""

from __future__ import annotations

from typing import Dict, List
from dataclasses import dataclass
import numpy as np
import shap
import re
import torch

from src.features.bias.bias_lexicon import compute_bias_features


# ---------------------------------------------------------
# Data Structure
# ---------------------------------------------------------

@dataclass
class BiasExplanation:
    token_importance: List[Dict]
    biased_token_highlights: List[str]
    sentence_bias_scores: List[Dict]
    attention_scores: List[Dict]
    bias_heatmap: List[Dict]


# ---------------------------------------------------------
# Sentence Tokenization
# ---------------------------------------------------------

def tokenize_sentences(text: str) -> List[str]:

    sentences = re.split(r"[.!?]+", text)

    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------
# Sentence-Level Bias Scoring
# ---------------------------------------------------------

def compute_sentence_bias(text: str) -> List[Dict]:

    sentences = tokenize_sentences(text)

    results = []

    for sentence in sentences:

        bias_result = compute_bias_features(sentence)

        results.append({
            "sentence": sentence,
            "bias_score": bias_result.bias_score,
            "biased_tokens": bias_result.biased_tokens
        })

    return results


# ---------------------------------------------------------
# SHAP Attribution
# ---------------------------------------------------------

def compute_shap_importance(model, tokenizer, text: str) -> List[Dict]:

    def predict(texts):

        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encodings)

        logits = outputs.logits.detach().numpy()

        return logits

    explainer = shap.Explainer(predict, tokenizer)

    shap_values = explainer([text])

    tokens = shap_values.data[0]
    values = shap_values.values[0]

    results = []

    for token, value in zip(tokens, values):

        results.append({
            "token": token,
            "importance": float(value)
        })

    return results


# ---------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------

def compute_integrated_gradients(model, tokenizer, text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    embedding_layer = model.get_input_embeddings()
    input_ids = inputs["input_ids"]
    input_embeddings = embedding_layer(input_ids).detach().requires_grad_(True)

    model_kwargs = {
        "inputs_embeds": input_embeddings,
        "attention_mask": inputs.get("attention_mask")
    }
    if "token_type_ids" in inputs:
        model_kwargs["token_type_ids"] = inputs["token_type_ids"]

    outputs = model(**model_kwargs)

    target = outputs.logits.max()

    target.backward()

    gradients = input_embeddings.grad

    importance = gradients.abs().sum(dim=-1).detach().cpu().numpy()[0]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    results = []

    for token, score in zip(tokens, importance):

        results.append({
            "token": token,
            "importance": float(score)
        })

    return results


# ---------------------------------------------------------
# Attention Visualization
# ---------------------------------------------------------

def compute_attention_scores(model, tokenizer, text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions[-1]

    attention_matrix = attentions.mean(dim=1)[0].detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    token_scores = attention_matrix.mean(axis=0)

    results = []

    for token, score in zip(tokens, token_scores):

        results.append({
            "token": token,
            "attention": float(score)
        })

    return results


# ---------------------------------------------------------
# Extract Important Tokens
# ---------------------------------------------------------

def extract_biased_tokens(token_importance: List[Dict], threshold: float = 0.05):

    highlights = []

    for item in token_importance:

        if abs(item["importance"]) >= threshold:
            highlights.append(item["token"])

    return highlights


# ---------------------------------------------------------
# Bias Heatmap
# ---------------------------------------------------------

def generate_bias_heatmap(token_importance: List[Dict]):

    heatmap = []

    for item in token_importance:

        score = abs(item["importance"])

        heatmap.append({
            "token": item["token"],
            "bias_strength": round(score, 4)
        })

    return heatmap


# ---------------------------------------------------------
# Main Explanation Pipeline
# ---------------------------------------------------------

def explain_bias(model, tokenizer, text: str) -> Dict:

    # -----------------------------------------------------
    # SHAP explanation
    # -----------------------------------------------------

    shap_importance = compute_shap_importance(model, tokenizer, text)

    biased_tokens = extract_biased_tokens(shap_importance)

    # -----------------------------------------------------
    # Integrated Gradients
    # -----------------------------------------------------

    ig_importance = compute_integrated_gradients(model, tokenizer, text)

    # -----------------------------------------------------
    # Attention scores
    # -----------------------------------------------------

    attention_scores = compute_attention_scores(model, tokenizer, text)

    # -----------------------------------------------------
    # Sentence bias
    # -----------------------------------------------------

    sentence_scores = compute_sentence_bias(text)

    # -----------------------------------------------------
    # Bias heatmap
    # -----------------------------------------------------

    heatmap = generate_bias_heatmap(shap_importance)

    return {
        "token_importance": shap_importance,
        "integrated_gradients": ig_importance,
        "biased_token_highlights": biased_tokens,
        "sentence_bias_scores": sentence_scores,
        "attention_scores": attention_scores,
        "bias_heatmap": heatmap
    }


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------

if __name__ == "__main__":

    example_text = """
    The corrupt regime has once again shocked the nation.
    Citizens are furious about the disastrous decision.
    """

    # Model + tokenizer would be loaded elsewhere
    # explanation = explain_bias(model, tokenizer, example_text)

    # print(explanation)