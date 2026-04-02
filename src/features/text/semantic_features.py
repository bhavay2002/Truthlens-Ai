"""
File Name: semantic_features.py
Module: Text Feature Engineering - Semantic Features
Description:
    Extracts semantic-level features from text using transformer embeddings
    and embedding-based similarity statistics. These features capture the
    semantic content and representational structure of the input text.

    The module supports optional integration with HuggingFace Transformers.
    If transformers are unavailable, a lightweight fallback embedding
    representation based on token hashing is used to ensure the feature
    pipeline remains operational.

Author: TruthLens Engineering Team
Date: 2026-04-02
Dependencies:
    dataclasses
    typing
    logging
    numpy
    transformers (optional)
    torch (optional)

Inputs:
    FeatureContext containing text and optional embeddings

Outputs:
    Dict[str, float] containing semantic representation statistics
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModel

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME)

    _model.eval()

    TRANSFORMER_AVAILABLE = True
except Exception:  # noqa: BLE001
    TRANSFORMER_AVAILABLE = False
    _tokenizer = None
    _model = None
    logger.warning("Transformers not available. Using fallback semantic features.")


def _fallback_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Lightweight deterministic embedding fallback using hashing.
    """

    vector = np.zeros(dim, dtype=np.float32)

    tokens = text.lower().split()

    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % dim
        vector[idx] += 1.0

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector /= norm

    return vector


def _transformer_embedding(text: str) -> np.ndarray:
    """
    Generate embedding using transformer model.
    """

    with torch.no_grad():
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        outputs = _model(**inputs)

        token_embeddings = outputs.last_hidden_state

        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)

        mean_embedding = summed / counts

        return mean_embedding.squeeze(0).cpu().numpy()


@dataclass
@register_feature
class SemanticFeatures(BaseFeature):
    """
    Extract semantic embedding statistics.

    Example features:
    - embedding_norm
    - embedding_mean
    - embedding_std
    - embedding_max
    - embedding_min
    """

    name: str = "semantic_features"
    description: str = "Transformer-based semantic representation features"

    def _compute_embedding(self, context: FeatureContext) -> np.ndarray:
        """
        Compute embedding from context or text.
        """

        if context.embeddings is not None:
            return np.asarray(context.embeddings)

        if not context.text:
            raise ValueError("FeatureContext.text cannot be empty")

        if TRANSFORMER_AVAILABLE:
            return _transformer_embedding(context.text)

        return _fallback_embedding(context.text)

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract semantic embedding statistics.
        """

        embedding = self._compute_embedding(context)

        if embedding.ndim != 1:
            embedding = embedding.flatten()

        features = {
            "embedding_norm": float(np.linalg.norm(embedding)),
            "embedding_mean": float(np.mean(embedding)),
            "embedding_std": float(np.std(embedding)),
            "embedding_max": float(np.max(embedding)),
            "embedding_min": float(np.min(embedding)),
        }

        logger.debug(
            "Semantic features extracted | dim=%d norm=%.4f",
            embedding.shape[0],
            features["embedding_norm"],
        )

        return features