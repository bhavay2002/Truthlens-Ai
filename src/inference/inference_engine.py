"""
File Name: inference_engine.py
Module: Inference Engine
Description:
    Production-grade inference engine responsible for loading trained models,
    executing predictions, managing device placement, validating inputs,
    and returning structured prediction outputs for downstream systems.

    Designed for scalable ML systems similar to those used in large-scale
    research and production environments.

Dependencies:
    logging
    typing
    dataclasses
    torch
    transformers
    pathlib
    json

Inputs:
    Raw text or batch of texts for prediction

Outputs:
    Structured prediction results including probabilities and predicted labels
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """
    Configuration for inference engine.
    """
    model_path: str
    tokenizer_path: Optional[str]
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 8
    return_probabilities: bool = True


@dataclass
class PredictionResult:
    """
    Structured prediction result.
    """
    text: str
    predicted_label: Union[int, str]
    probabilities: Optional[List[float]] = None
    logits: Optional[List[float]] = None


class InferenceEngine:
    """
    High-level inference engine used for model loading and prediction.

    Responsibilities:
    - Model loading
    - Tokenization
    - Device management
    - Batch inference
    - Output formatting
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = None
        self.tokenizer = None
        self.label_map: Optional[Dict[int, str]] = None

        self._load_model()

    def _resolve_device(self, device: str) -> torch.device:
        """
        Resolve device configuration.
        """
        if device == "auto":
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            resolved = torch.device(device)

        logger.info("Using device: %s", resolved)
        return resolved

    def _load_model(self) -> None:
        """
        Load model and tokenizer from disk.
        """
        model_path = Path(self.config.model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        tokenizer_path = (
            self.config.tokenizer_path
            if self.config.tokenizer_path
            else self.config.model_path
        )

        try:
            logger.info("Loading tokenizer from %s", tokenizer_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            logger.info("Loading model from %s", model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            )

            self.model.to(self.device)
            self.model.eval()

            self._load_label_map(model_path)

        except Exception as exc:
            logger.exception("Failed to load model")
            raise RuntimeError("Model loading failed") from exc

    def _load_label_map(self, model_path: Path) -> None:
        """
        Load label mapping if available.
        """
        label_map_path = model_path / "label_map.json"

        if label_map_path.exists():
            try:
                with open(label_map_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)

                self.label_map = {int(k): v for k, v in raw_map.items()}
                logger.info("Loaded label map")

            except Exception as exc:
                logger.warning("Failed to load label map: %s", exc)

    def _validate_input(self, texts: Union[str, List[str]]) -> List[str]:
        """
        Validate input texts.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, list):
            raise TypeError("Input must be string or list of strings")

        if not texts:
            raise ValueError("Input text list is empty")

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("All inputs must be strings")

        return texts

    def _batchify(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """
        Split items into batches.
        """
        return [
            items[i : i + batch_size]
            for i in range(0, len(items), batch_size)
        ]

    def predict(
        self,
        texts: Union[str, List[str]]
    ) -> List[PredictionResult]:
        """
        Perform batch prediction.
        """
        validated_texts = self._validate_input(texts)
        batches = self._batchify(validated_texts, self.config.batch_size)

        results: List[PredictionResult] = []

        with torch.no_grad():
            for batch in batches:
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )

                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                outputs = self.model(**encoded)
                logits = outputs.logits

                probabilities = torch.softmax(logits, dim=-1)

                predicted_indices = torch.argmax(probabilities, dim=-1)

                for i, text in enumerate(batch):
                    label_idx = predicted_indices[i].item()

                    label = (
                        self.label_map[label_idx]
                        if self.label_map
                        else label_idx
                    )

                    probs = probabilities[i].tolist()
                    logit_values = logits[i].tolist()

                    result = PredictionResult(
                        text=text,
                        predicted_label=label,
                        probabilities=probs if self.config.return_probabilities else None,
                        logits=logit_values,
                    )

                    results.append(result)

        return results

    def predict_single(self, text: str) -> PredictionResult:
        """
        Convenience method for single prediction.
        """
        results = self.predict([text])
        return results[0]

    def warmup(self) -> None:
        """
        Perform a dummy inference for warm-up.
        """
        logger.info("Running inference warm-up")

        dummy_text = "Warmup inference example."

        try:
            _ = self.predict_single(dummy_text)
        except Exception as exc:
            logger.warning("Warm-up failed: %s", exc)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about loaded model.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        return {
            "model_path": self.config.model_path,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "num_trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
        }