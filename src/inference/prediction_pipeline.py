"""
File Name: prediction_pipeline.py
Module: Prediction Pipeline
Description:
    Executes trained ML models to produce structured predictions for multiple
    analytical tasks including bias detection, ideology classification,
    propaganda detection, emotion analysis, and credibility estimation.

    The pipeline coordinates feature preparation, device placement, model
    inference, and result aggregation. It is designed to ensure deterministic
    inference behavior consistent with the training pipeline.

Dependencies:
    logging
    typing
    dataclasses
    numpy
    torch

Inputs:
    Prepared feature vectors or tensors.

Outputs:
    Structured prediction dictionary containing outputs from multiple models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PredictionPipelineConfig:
    """
    Configuration for prediction pipeline.
    """
    device: str = "cpu"
    apply_softmax: bool = True
    credibility_weight_bias: float = 0.25
    credibility_weight_propaganda: float = 0.35
    credibility_weight_emotion: float = 0.15
    credibility_weight_ideology: float = 0.25


class PredictionPipeline:
    """
    Production-grade prediction pipeline responsible for executing trained
    ML models and producing structured predictions.
    """

    def __init__(
        self,
        config: PredictionPipelineConfig,
        bias_model: Optional[torch.nn.Module] = None,
        ideology_model: Optional[torch.nn.Module] = None,
        propaganda_model: Optional[torch.nn.Module] = None,
        emotion_model: Optional[torch.nn.Module] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device if device else config.device)

        self.bias_model = bias_model
        self.ideology_model = ideology_model
        self.propaganda_model = propaganda_model
        self.emotion_model = emotion_model

        logger.info("PredictionPipeline initialized on device: %s", self.device)

    def _predict_logits(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run model forward pass safely.
        """
        try:
            model.eval()
            with torch.no_grad():
                logits = model(features)
                return logits
        except Exception as exc:
            logger.exception("Model inference failed")
            raise RuntimeError("Prediction failed") from exc

    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply softmax to logits.
        """
        return torch.softmax(logits, dim=-1)

    def _predict_class(self, logits: torch.Tensor) -> int:
        """
        Return predicted class index.
        """
        return torch.argmax(logits, dim=-1).item()

    def _predict_probability(self, logits: torch.Tensor) -> float:
        """
        Return probability for binary classification tasks.
        """
        probs = self._softmax(logits)
        return probs[:, 1].item()

    def _predict_bias(self, features: torch.Tensor) -> Optional[str]:
        """
        Predict political bias.
        """
        if self.bias_model is None:
            return None

        logits = self._predict_logits(self.bias_model, features)

        if self.config.apply_softmax:
            logits = self._softmax(logits)

        label_idx = self._predict_class(logits)

        bias_labels = {0: "left", 1: "center", 2: "right"}
        return bias_labels.get(label_idx, "unknown")

    def _predict_ideology(self, features: torch.Tensor) -> Optional[str]:
        """
        Predict ideological leaning.
        """
        if self.ideology_model is None:
            return None

        logits = self._predict_logits(self.ideology_model, features)

        if self.config.apply_softmax:
            logits = self._softmax(logits)

        label_idx = self._predict_class(logits)

        ideology_labels = {0: "left", 1: "center", 2: "right"}
        return ideology_labels.get(label_idx, "unknown")

    def _predict_propaganda(self, features: torch.Tensor) -> Optional[float]:
        """
        Predict propaganda probability.
        """
        if self.propaganda_model is None:
            return None

        logits = self._predict_logits(self.propaganda_model, features)

        probability = self._predict_probability(logits)
        return float(probability)

    def _predict_emotion(self, features: torch.Tensor) -> Optional[Dict[str, float]]:
        """
        Predict emotion distribution.
        """
        if self.emotion_model is None:
            return None

        logits = self._predict_logits(self.emotion_model, features)

        probs = self._softmax(logits).cpu().numpy()[0]

        emotion_labels = [
            "anger",
            "fear",
            "joy",
            "sadness",
            "surprise",
        ]

        return {emotion_labels[i]: float(probs[i]) for i in range(len(emotion_labels))}

    def _compute_credibility_score(
        self,
        bias: Optional[str],
        propaganda_prob: Optional[float],
        emotion: Optional[Dict[str, float]],
        ideology: Optional[str],
    ) -> float:
        """
        Compute credibility score from model signals.
        """
        bias_score = 0.5
        ideology_score = 0.5
        propaganda_score = 1.0
        emotion_score = 0.5

        if bias == "center":
            bias_score = 1.0
        elif bias in {"left", "right"}:
            bias_score = 0.6

        if ideology == "center":
            ideology_score = 1.0
        elif ideology in {"left", "right"}:
            ideology_score = 0.6

        if propaganda_prob is not None:
            propaganda_score = 1.0 - propaganda_prob

        if emotion:
            emotion_intensity = max(emotion.values())
            emotion_score = 1.0 - emotion_intensity

        score = (
            bias_score * self.config.credibility_weight_bias
            + propaganda_score * self.config.credibility_weight_propaganda
            + emotion_score * self.config.credibility_weight_emotion
            + ideology_score * self.config.credibility_weight_ideology
        )

        return float(np.clip(score, 0.0, 1.0))

    def predict(self, features: torch.Tensor) -> Dict[str, Any]:
        """
        Execute all models and return structured predictions.
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a torch.Tensor")

        features = features.to(self.device)

        bias = self._predict_bias(features)
        ideology = self._predict_ideology(features)
        propaganda_prob = self._predict_propaganda(features)
        emotion = self._predict_emotion(features)

        credibility_score = self._compute_credibility_score(
            bias=bias,
            propaganda_prob=propaganda_prob,
            emotion=emotion,
            ideology=ideology,
        )

        result = {
            "bias": bias,
            "ideology": ideology,
            "propaganda_probability": propaganda_prob,
            "emotion": emotion,
            "credibility_score": credibility_score,
        }

        logger.debug("Prediction result: %s", result)

        return result