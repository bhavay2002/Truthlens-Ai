"""
File Name: prediction_pipeline.py
Module: Prediction Pipeline
Description:
    Executes trained ML models to produce structured predictions for
    TruthLens analytical tasks including:

    - bias detection
    - ideology classification
    - propaganda detection
    - emotion analysis
    - credibility estimation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.features.emotion.emotion_schema import EMOTION_LABELS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class PredictionPipelineConfig:

    device: str = "cpu"

    credibility_weight_bias: float = 0.25
    credibility_weight_propaganda: float = 0.35
    credibility_weight_emotion: float = 0.15
    credibility_weight_ideology: float = 0.25


# ---------------------------------------------------------------------
# Prediction Pipeline
# ---------------------------------------------------------------------

class PredictionPipeline:

    def __init__(
        self,
        config: PredictionPipelineConfig,
        bias_model: Optional[torch.nn.Module] = None,
        ideology_model: Optional[torch.nn.Module] = None,
        propaganda_model: Optional[torch.nn.Module] = None,
        emotion_model: Optional[torch.nn.Module] = None,
    ) -> None:

        self.config = config
        self.device = torch.device(config.device)

        self.bias_model = bias_model
        self.ideology_model = ideology_model
        self.propaganda_model = propaganda_model
        self.emotion_model = emotion_model

        # Move models to device
        for model in [
            self.bias_model,
            self.ideology_model,
            self.propaganda_model,
            self.emotion_model,
        ]:
            if model is not None:
                model.to(self.device)
                model.eval()

        logger.info("PredictionPipeline initialized on device: %s", self.device)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def _predict_logits(
        self,
        model: torch.nn.Module,
        features: torch.Tensor,
    ) -> torch.Tensor:

        with torch.no_grad():

            outputs = model(features)

            if isinstance(outputs, dict):

                if "logits" in outputs:
                    return outputs["logits"]

                if "probabilities" in outputs:
                    return torch.log(outputs["probabilities"] + 1e-9)

            return outputs

    # -------------------------------------------------------------

    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:

        return torch.softmax(logits, dim=-1)

    # -------------------------------------------------------------

    def _sigmoid(self, logits: torch.Tensor) -> torch.Tensor:

        return torch.sigmoid(logits)

    # -------------------------------------------------------------

    def _predict_class(self, probs: torch.Tensor) -> int:

        return int(torch.argmax(probs, dim=-1).item())

    # -------------------------------------------------------------

    def _prediction_confidence(self, probs: torch.Tensor) -> float:

        max_prob = torch.max(probs)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9))
        confidence = max_prob * torch.exp(-entropy)

        return float(confidence.item())

    # -----------------------------------------------------------------
    # Bias Prediction
    # -----------------------------------------------------------------

    def _predict_bias(self, features: torch.Tensor) -> Optional[str]:

        if self.bias_model is None:
            return None

        logits = self._predict_logits(self.bias_model, features)
        probs = self._softmax(logits)

        label_idx = self._predict_class(probs)

        bias_labels = {
            0: "non_bias",
            1: "bias",
        }

        return bias_labels.get(label_idx, "unknown")

    # -----------------------------------------------------------------
    # Ideology Prediction
    # -----------------------------------------------------------------

    def _predict_ideology(self, features: torch.Tensor) -> Optional[str]:

        if self.ideology_model is None:
            return None

        logits = self._predict_logits(self.ideology_model, features)
        probs = self._softmax(logits)

        label_idx = self._predict_class(probs)

        ideology_labels = {
            0: "left",
            1: "center",
            2: "right",
        }

        return ideology_labels.get(label_idx, "unknown")

    # -----------------------------------------------------------------
    # Propaganda Prediction
    # -----------------------------------------------------------------

    def _predict_propaganda(self, features: torch.Tensor) -> Optional[float]:

        if self.propaganda_model is None:
            return None

        logits = self._predict_logits(self.propaganda_model, features)
        probs = self._softmax(logits)

        return float(probs[:, 1].item())

    # -----------------------------------------------------------------
    # Emotion Prediction
    # -----------------------------------------------------------------

    def _predict_emotion(
        self,
        features: torch.Tensor,
    ) -> Optional[Dict[str, float]]:

        if self.emotion_model is None:
            return None

        logits = self._predict_logits(self.emotion_model, features)

        probs = self._sigmoid(logits).cpu().numpy()[0]

        emotion_distribution: Dict[str, float] = {}

        for i, emotion in enumerate(EMOTION_LABELS):

            if i < len(probs):
                emotion_distribution[emotion] = float(probs[i])
            else:
                emotion_distribution[emotion] = 0.0

        return emotion_distribution

    # -----------------------------------------------------------------
    # Credibility Score
    # -----------------------------------------------------------------

    def _compute_credibility_score(
        self,
        bias: Optional[str],
        propaganda_prob: Optional[float],
        emotion: Optional[Dict[str, float]],
        ideology: Optional[str],
    ) -> tuple[float, Dict[str, float]]:

        bias_score = 0.5
        ideology_score = 0.5
        propaganda_score = 1.0
        emotion_score = 0.5

        # Bias component
        if bias == "non_bias":
            bias_score = 1.0
        elif bias == "bias":
            bias_score = 0.5

        # Ideology component
        if ideology == "center":
            ideology_score = 1.0
        elif ideology in {"left", "right"}:
            ideology_score = 0.6

        # Propaganda component
        if propaganda_prob is not None:
            propaganda_score = 1.0 - propaganda_prob

        # Emotion component
        if emotion:

            values = np.array(list(emotion.values()))
            max_intensity = float(np.max(values))

            eps = 1e-9
            entropy = -np.sum(values * np.log(values + eps))

            n = len(values)
            normalized_entropy = entropy / np.log(n) if n > 1 else 0.0

            emotion_score = (
                0.5 * (1.0 - max_intensity)
                + 0.5 * normalized_entropy
            )

        score = (
            bias_score * self.config.credibility_weight_bias
            + propaganda_score * self.config.credibility_weight_propaganda
            + emotion_score * self.config.credibility_weight_emotion
            + ideology_score * self.config.credibility_weight_ideology
        )

        credibility = float(np.clip(score, 0.0, 1.0))

        explanation = {
            "bias_component": bias_score,
            "propaganda_component": propaganda_score,
            "emotion_component": emotion_score,
            "ideology_component": ideology_score,
        }

        return credibility, explanation

    # -----------------------------------------------------------------
    # Main Prediction
    # -----------------------------------------------------------------

    def predict(self, features: torch.Tensor) -> Dict[str, Any]:

        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a torch.Tensor")

        features = features.to(self.device)

        bias = self._predict_bias(features)
        ideology = self._predict_ideology(features)
        propaganda_prob = self._predict_propaganda(features)
        emotion = self._predict_emotion(features)

        credibility_score, explanation = self._compute_credibility_score(
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
            "credibility_explanation": explanation,
        }

        logger.debug("Prediction result: %s", result)

        return result