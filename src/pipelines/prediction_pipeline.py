"""
File Name: predictor.py
Module: TruthLens Inference - Prediction Pipeline
Description:
    Implements the production-grade prediction pipeline for the TruthLens AI
    system. This module loads trained model assets, prepares text inputs,
    performs optional feature transformation, runs model inference using
    PyTorch, and returns structured prediction outputs.

    The module supports:
        • single text prediction
        • batch prediction
        • GPU / CPU device management
        • tokenizer-based input preparation
        • optional engineered feature transformation
        • structured prediction output with probabilities

Dependencies:
    logging
    typing
    pandas
    torch

Inputs:
    Raw text string or list of text strings

Outputs:
    Prediction dictionary containing predicted label, confidence score,
    and probability distribution
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

from src.features.pipelines.feature_pipeline import transform_feature_pipeline
from src.utils.input_validation import (
    ensure_non_empty_text,
    ensure_non_empty_text_list,
)
from src.utils.settings import load_settings


logger = logging.getLogger(__name__)

SETTINGS = load_settings()

MAX_LENGTH: int = SETTINGS.model.max_length

DEFAULT_ID2LABEL: Dict[int, str] = {0: "REAL", 1: "FAKE"}


class Predictor:
    """
    Production inference class for TruthLens model predictions.
    """

    def __init__(self) -> None:
        """
        Initialize prediction assets.
        """

        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._vectorizer: Any | None = None

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "Predictor initialized",
            extra={"device": str(self._device)},
        )

    def _get_assets(self) -> Tuple[Any, Any, Any]:
        """
        Load model assets lazily.

        Returns
        -------
        Tuple[model, tokenizer, vectorizer]
        """

        if self._model is None or self._tokenizer is None:
            try:
                from src.models.registry.model_registry import ModelRegistry

                assets = ModelRegistry.load_model()

                self._model = assets["model"]
                self._tokenizer = assets["tokenizer"]
                self._vectorizer = assets.get("vectorizer")

                self._model.to(self._device)
                self._model.eval()

                logger.info("Model assets loaded successfully")

            except Exception as exc:
                logger.exception("Failed to load model assets")
                raise RuntimeError("Model loading failed") from exc

        return self._model, self._tokenizer, self._vectorizer

    def _resolve_id2label(self, model: Any) -> Dict[int, str]:
        """
        Resolve label mapping from model configuration.
        """

        id2label = getattr(model.config, "id2label", None) or {}
        resolved: Dict[int, str] = {}

        for key, value in id2label.items():
            try:
                resolved[int(key)] = str(value).upper()
            except (TypeError, ValueError):
                continue

        if resolved:
            return resolved

        label2id = getattr(model.config, "label2id", None) or {}

        for label, idx in label2id.items():
            try:
                resolved[int(idx)] = str(label).upper()
            except (TypeError, ValueError):
                continue

        return resolved or dict(DEFAULT_ID2LABEL)

    @staticmethod
    def _label_for_index(index: int, mapping: Dict[int, str]) -> str:
        """
        Resolve label string from class index.
        """

        return mapping.get(index, f"LABEL_{index}")

    def _prepare_model_text(self, text: str, vectorizer: Any) -> str:
        """
        Optionally transform text using feature engineering pipeline.
        """

        if vectorizer is None:
            return text

        try:
            df = pd.DataFrame({"text": [text]})

            transformed_df = transform_feature_pipeline(
                df,
                vectorizer=vectorizer,
                text_column="text",
            )

            return str(transformed_df["engineered_text"].iloc[0])

        except Exception as exc:
            logger.exception("Feature transformation failed")
            raise RuntimeError("Feature transformation failed") from exc

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run prediction pipeline on a single text.

        Parameters
        ----------
        text : str

        Returns
        -------
        Dict[str, Any]
        """

        ensure_non_empty_text(text)

        logger.info("Running prediction pipeline")

        model, tokenizer, vectorizer = self._get_assets()

        model_text = self._prepare_model_text(text, vectorizer)

        try:
            inputs = tokenizer(
                model_text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            model_inputs = {
                key: value.to(self._device) for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = model(**model_inputs)

                logits = outputs.logits

                probs = torch.softmax(logits, dim=1)

                confidence, pred_class = torch.max(probs, dim=1)

        except Exception as exc:
            logger.exception("Model inference failed")
            raise RuntimeError("Inference failed") from exc

        id2label = self._resolve_id2label(model)

        prediction = self._label_for_index(int(pred_class.item()), id2label)

        confidence_value = float(confidence.item())

        probabilities: Dict[str, float] = {}

        for i in range(int(probs.shape[1])):
            label = self._label_for_index(i, id2label)

            probabilities[label] = float(probs[0][i].item())

        logger.info(
            "Prediction completed | class=%s | confidence=%.3f",
            prediction,
            confidence_value,
        )

        return {
            "prediction": prediction,
            "confidence": confidence_value,
            "probabilities": probabilities,
            "model_version": getattr(model.config, "model_version", None),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Run predictions on multiple texts.

        Parameters
        ----------
        texts : List[str]

        Returns
        -------
        List[Dict[str, Any]]
        """

        normalized = ensure_non_empty_text_list(texts)

        results: List[Dict[str, Any]] = []

        for idx, text in enumerate(normalized):
            ensure_non_empty_text(text, name=f"texts[{idx}]")

            results.append(self.predict(text))

        return results


_model: Any | None = None
_tokenizer: Any | None = None
_vectorizer: Any | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_assets() -> Tuple[Any, Any, Any]:
    """
    Load and cache model assets for module-level prediction helpers.
    """

    global _model, _tokenizer, _vectorizer

    if _model is None or _tokenizer is None:
        from src.models.registry.model_registry import ModelRegistry

        assets = ModelRegistry.load_model()
        _model = assets["model"]
        _tokenizer = assets["tokenizer"]
        _vectorizer = assets.get("vectorizer")

        _model.to(_device)
        _model.eval()

    return _model, _tokenizer, _vectorizer


def _resolve_id2label(model: Any) -> Dict[int, str]:
    id2label = getattr(model.config, "id2label", None) or {}
    resolved: Dict[int, str] = {}

    for key, value in id2label.items():
        try:
            resolved[int(key)] = str(value).upper()
        except (TypeError, ValueError):
            continue

    if resolved:
        return resolved

    label2id = getattr(model.config, "label2id", None) or {}
    for label, idx in label2id.items():
        try:
            resolved[int(idx)] = str(label).upper()
        except (TypeError, ValueError):
            continue

    return resolved or dict(DEFAULT_ID2LABEL)


def _label_for_index(index: int, mapping: Dict[int, str]) -> str:
    return mapping.get(index, f"LABEL_{index}")


def _prepare_model_text(text: str, vectorizer: Any) -> str:
    if vectorizer is None:
        return text

    df = pd.DataFrame({"text": [text]})
    transformed_df = transform_feature_pipeline(
        df,
        vectorizer=vectorizer,
        text_column="text",
    )
    return str(transformed_df["engineered_text"].iloc[0])


def predict_text(text: str) -> Dict[str, Any]:
    """
    Run full prediction pipeline on a single text input.
    """

    ensure_non_empty_text(text)

    model, tokenizer, vectorizer = _get_assets()
    model_text = _prepare_model_text(text, vectorizer)

    inputs = tokenizer(
        model_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    model_inputs = {key: value.to(_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    id2label = _resolve_id2label(model)
    prediction = _label_for_index(int(pred_class.item()), id2label)
    confidence_value = float(confidence.item())

    probabilities: Dict[str, float] = {}
    for i in range(int(probs.shape[1])):
        label = _label_for_index(i, id2label)
        probabilities[label] = float(probs[0][i].item())

    return {
        "prediction": prediction,
        "confidence": confidence_value,
        "probabilities": probabilities,
    }


def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Run prediction pipeline for a list of texts.
    """

    normalized = ensure_non_empty_text_list(texts)
    for idx, text in enumerate(normalized):
        ensure_non_empty_text(text, name=f"texts[{idx}]")

    return [predict_text(text) for text in normalized]
