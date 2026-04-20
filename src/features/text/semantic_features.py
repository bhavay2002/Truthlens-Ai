"""
File Name: semantic_features.py
Module: Text Feature Engineering - Semantic Features
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np

from src.features.base.base_feature import  BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature

logger = logging.getLogger(__name__)


# =========================================================
# GLOBAL MODEL CACHE (PROCESS-LEVEL SINGLETON)
# =========================================================

_SEMANTIC_MODEL = None
_SEMANTIC_TOKENIZER = None
_SEMANTIC_TORCH = None
_SEMANTIC_DEVICE = None


def _fallback_embedding(text: str, dim: int = 128) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()

    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        vector[h % dim] += 1.0

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


@dataclass
@register_feature
class SemanticFeatures(BaseFeature):
    name: str = "semantic_features"
    description: str = "Transformer-based semantic representation features"

    _torch: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _device: Any = field(default=None, init=False, repr=False)
    _model_on_device: bool = field(default=False, init=False, repr=False)
    _transformer_available: bool = field(default=False, init=False, repr=False)

    def initialize(self) -> None:
        global _SEMANTIC_MODEL, _SEMANTIC_TOKENIZER, _SEMANTIC_TORCH, _SEMANTIC_DEVICE

        if _SEMANTIC_MODEL is not None and _SEMANTIC_TOKENIZER is not None:
            self._model = _SEMANTIC_MODEL
            self._tokenizer = _SEMANTIC_TOKENIZER
            self._torch = _SEMANTIC_TORCH
            self._device = _SEMANTIC_DEVICE
            self._transformer_available = True
            self._model_on_device = True
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            model_name = "sentence-transformers/all-MiniLM-L6-v2"

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            _SEMANTIC_MODEL = model
            _SEMANTIC_TOKENIZER = tokenizer
            _SEMANTIC_TORCH = torch
            _SEMANTIC_DEVICE = device

            self._model = model
            self._tokenizer = tokenizer
            self._torch = torch
            self._device = device

            self._transformer_available = True
            self._model_on_device = True

            logger.info("Semantic transformer loaded once and cached")

        except Exception:  # noqa: BLE001
            self._tokenizer = None
            self._model = None
            self._torch = None
            self._device = None
            self._transformer_available = False
            self._model_on_device = False
            logger.warning("Transformers not available. Using fallback semantic features.")

    def _transformer_embedding(self, text: str) -> np.ndarray:
        if not self._transformer_available or self._tokenizer is None or self._model is None:
            return _fallback_embedding(text)

        torch = self._torch
        if not self._model_on_device:
            self._model.to(self._device)
            self._model_on_device = True
        with torch.no_grad():
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_embedding = summed / counts
            emb = mean_embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return np.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=0.0)

    def _transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 128), dtype=np.float32)
        if not self._transformer_available or self._tokenizer is None or self._model is None:
            return np.stack([_fallback_embedding(t) for t in texts], axis=0)

        torch = self._torch
        if not self._model_on_device:
            self._model.to(self._device)
            self._model_on_device = True

        with torch.no_grad():
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            outputs = self._model(**inputs)

            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_embeddings = summed / counts

            emb = mean_embeddings.detach().cpu().numpy().astype(np.float32)
            return np.nan_to_num(emb, nan=0.0, posinf=1.0, neginf=0.0)

    def _compute_embedding(self, context: FeatureContext) -> np.ndarray:
        if context.embeddings is not None:
            return np.asarray(context.embeddings)

        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")
        if not context.text.strip():
            return np.zeros(128, dtype=np.float32)

        self.initialize()
        if self._transformer_available:
            return self._transformer_embedding(context.text)

        return _fallback_embedding(context.text)

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        embedding = self._compute_embedding(context)

        embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=0.0)

        if embedding.size == 0:
            return {}

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

    def extract_batch(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        if not contexts:
            return []

        self.initialize()

        embeddings: List[np.ndarray] = [
            np.asarray([], dtype=np.float32) for _ in contexts
        ]
        text_contexts: List[str] = []
        text_indices: List[int] = []

        for index, context in enumerate(contexts):
            if context.embeddings is not None:
                embeddings[index] = np.asarray(context.embeddings)
            elif isinstance(context.text, str) and context.text.strip():
                text_contexts.append(context.text)
                text_indices.append(index)
            else:
                embeddings[index] = np.asarray([], dtype=np.float32)

        if text_contexts:
            if self._transformer_available:
                batch_embeddings = self._transformer_embeddings(text_contexts)
            else:
                batch_embeddings = np.stack([_fallback_embedding(t) for t in text_contexts], axis=0)

            for index, emb in zip(text_indices, batch_embeddings):
                embeddings[index] = emb

        return [self.extract(FeatureContext(text="", embeddings=emb)) for emb in embeddings]