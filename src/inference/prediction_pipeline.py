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

    Integrates with the explainability subsystem to enrich predictions
    with token-level attribution, attention rollout, explanation
    aggregation, consistency metrics, and optional caching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.features.emotion.emotion_schema import EMOTION_LABELS
from src.inference.feature_preparer import FeaturePreparer
from src.models.inference.prediction_output import PredictionOutput

from src.explainability.attention_rollout import AttentionRollout
from src.explainability.attention_visualizer import AttentionVisualizer
from src.explainability.explanation_aggregator import (
    ExplanationAggregator,
    AggregationWeights,
)
from src.explainability.explanation_cache import ExplanationCache
from src.explainability.explanation_consistency import ExplanationConsistency
from src.explainability.explanation_metrics import ExplanationMetrics
from src.explainability.explanation_visualizer import ExplanationVisualizer
from src.explainability.model_explainer import explain_prediction_full, explain_fast
from src.explainability.propaganda_explainer import PropagandaExplainer
from src.explainability.token_alignment import align_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class PredictionPipelineConfig:

    device: str = "cpu"
    return_probabilities: bool = True

    credibility_weight_bias: float = 0.25
    credibility_weight_propaganda: float = 0.35
    credibility_weight_emotion: float = 0.15
    credibility_weight_ideology: float = 0.25


@dataclass
class ExplainabilityConfig:
    """
    Configuration for the explainability layer attached to the
    prediction pipeline.
    """

    enabled: bool = True
    use_lime: bool = True
    use_shap: bool = False
    use_attention_rollout: bool = True
    use_propaganda_explainer: bool = True
    use_aggregation: bool = True
    use_consistency: bool = True
    use_explanation_metrics: bool = False
    cache_enabled: bool = True
    cache_max_size: int = 128
    cache_dir: Optional[str] = None
    aggregation_weights: AggregationWeights = field(
        default_factory=AggregationWeights
    )


# ---------------------------------------------------------------------
# Explainability Layer
# ---------------------------------------------------------------------

class ExplainabilityLayer:
    """
    Orchestrates all explainability components for the prediction pipeline.

    Holds instances of:
        - ExplanationCache      -- LRU cache for explanation results
        - align_tokens          -- subword-to-word token merging
        - AttentionRollout      -- cumulative attention-flow attribution
        - ExplanationAggregator -- weighted combination of explanation signals
        - ExplanationConsistency -- pairwise correlation between methods
        - ExplanationMetrics    -- faithfulness / comprehensiveness metrics
        - ExplanationVisualizer -- matplotlib visualizations (optional)
        - AttentionVisualizer   -- attention heatmap (requires model)
        - PropagandaExplainer   -- gradient-based token attribution (requires model)
    """

    def __init__(
        self,
        config: ExplainabilityConfig,
        propaganda_model: Optional[torch.nn.Module] = None,
        attention_model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.config = config

        self.cache = ExplanationCache(
            max_size=config.cache_max_size,
            cache_dir=config.cache_dir,
        ) if config.cache_enabled else None

        self.attention_rollout = AttentionRollout()
        self.aggregator = ExplanationAggregator(
            weights=config.aggregation_weights
        ) if config.use_aggregation else None
        self.consistency = ExplanationConsistency() if config.use_consistency else None
        self.metrics = ExplanationMetrics() if config.use_explanation_metrics else None
        self.visualizer = ExplanationVisualizer()

        self.propaganda_explainer: Optional[PropagandaExplainer] = None
        if config.use_propaganda_explainer and propaganda_model is not None:
            self.propaganda_explainer = PropagandaExplainer(propaganda_model)

        self.attention_visualizer: Optional[AttentionVisualizer] = None
        if attention_model is not None:
            self.attention_visualizer = AttentionVisualizer(attention_model)

        logger.info("ExplainabilityLayer initialized")

    def explain(
        self,
        text: str,
        predict_fn: Callable[[str], Dict[str, Any]],
        tokens: Optional[List[str]] = None,
        attentions: Optional[List[torch.Tensor]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete explanation package for a prediction.

        Parameters
        ----------
        text : str
            Raw article text being explained.
        predict_fn : Callable
            Function that accepts text and returns a prediction dict.
        tokens : list of str, optional
            Pre-tokenized tokens aligned with the model input.
        attentions : list of torch.Tensor, optional
            Per-layer attention tensors for attention rollout.
        input_ids : torch.Tensor, optional
            Token id tensor for propaganda explainer.
        attention_mask : torch.Tensor, optional
            Attention mask tensor for propaganda explainer.
        model : optional
            Transformer model for bias/emotion explanation.
        tokenizer : optional
            Tokenizer for bias/emotion explanation.

        Returns
        -------
        Dict containing explanation signals and aggregated result.
        """

        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None:
                logger.debug("Explanation cache hit")
                return cached

        explanation: Dict[str, Any] = {}

        model_explanation = explain_prediction_full(
            text=text,
            predict_fn=predict_fn,
            model=model,
            tokenizer=tokenizer,
            use_lime=self.config.use_lime,
            use_shap=self.config.use_shap,
        )
        explanation["model_explanation"] = model_explanation

        lime_explanation = model_explanation.get("lime_explanation")
        shap_explanation = model_explanation.get("shap_explanation")

        rollout_result: Optional[Dict[str, Any]] = None
        if self.config.use_attention_rollout and attentions and tokens:
            try:
                rollout_result = self.attention_rollout.compute_rollout(
                    attentions=attentions,
                    tokens=tokens,
                )
                aligned_tokens, aligned_scores = align_tokens(
                    rollout_result["tokens"],
                    rollout_result["rollout_scores"],
                )
                rollout_result["aligned_tokens"] = aligned_tokens
                rollout_result["aligned_scores"] = aligned_scores
                explanation["attention_rollout"] = rollout_result
            except Exception as exc:
                logger.warning("Attention rollout failed: %s", exc)

        propaganda_token_scores: Optional[Dict[str, float]] = None
        if (
            self.propaganda_explainer is not None
            and input_ids is not None
            and attention_mask is not None
            and tokens is not None
        ):
            try:
                propaganda_token_scores = self.propaganda_explainer.explain(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokens=tokens,
                )
                explanation["propaganda_token_scores"] = propaganda_token_scores
                explanation["propaganda_intensity"] = (
                    self.propaganda_explainer.propaganda_intensity(
                        propaganda_token_scores
                    )
                )
            except Exception as exc:
                logger.warning("Propaganda explainer failed: %s", exc)

        if self.aggregator is not None:
            try:
                shap_items = (
                    shap_explanation
                    if isinstance(shap_explanation, list)
                    else None
                )
                lime_items = (
                    lime_explanation
                    if isinstance(lime_explanation, list)
                    else None
                )
                attention_items: Optional[List[Dict]] = None
                if rollout_result and "aligned_tokens" in rollout_result:
                    attention_items = [
                        {"token": t, "attention": s}
                        for t, s in zip(
                            rollout_result["aligned_tokens"],
                            rollout_result["aligned_scores"],
                        )
                    ]
                aggregated = self.aggregator.aggregate(
                    shap_importance=shap_items,
                    attention_scores=attention_items,
                    lime_importance=lime_items,
                )
                explanation["aggregated_explanation"] = aggregated
            except Exception as exc:
                logger.warning("Explanation aggregation failed: %s", exc)

        if self.consistency is not None:
            try:
                shap_items_c = (
                    shap_explanation
                    if isinstance(shap_explanation, list)
                    else None
                )
                lime_items_c = (
                    lime_explanation
                    if isinstance(lime_explanation, list)
                    else None
                )
                attention_items_c: Optional[List[Dict]] = None
                if rollout_result and "aligned_tokens" in rollout_result:
                    attention_items_c = [
                        {"token": t, "attention": s}
                        for t, s in zip(
                            rollout_result["aligned_tokens"],
                            rollout_result["aligned_scores"],
                        )
                    ]
                consistency_scores = self.consistency.compute(
                    shap_importance=shap_items_c,
                    attention_scores=attention_items_c,
                    lime_importance=lime_items_c,
                )
                explanation["consistency_metrics"] = consistency_scores
            except Exception as exc:
                logger.warning("Explanation consistency failed: %s", exc)

        if self.cache is not None:
            self.cache.set(text, explanation)

        return explanation

    def clear_cache(self) -> None:
        """Clear both in-memory and disk explanation caches."""
        if self.cache is not None:
            self.cache.clear_memory()
            self.cache.clear_disk()


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
        explainability_layer: Optional[ExplainabilityLayer] = None,
        aggregation_pipeline: Optional[AggregationPipeline] = None,
    ) -> None:

        self.config = config
        self.device = torch.device(config.device)

        self.bias_model = bias_model
        self.ideology_model = ideology_model
        self.propaganda_model = propaganda_model
        self.emotion_model = emotion_model
        self.explainability_layer = explainability_layer
        self.aggregation_pipeline = aggregation_pipeline or AggregationPipeline()

        for model in [
            self.bias_model,
            self.ideology_model,
            self.propaganda_model,
            self.emotion_model,
        ]:
            if model is not None:
                model.to(self.device)
                model.eval()

        if self.device.type == "cuda":
            for model in [
                self.bias_model,
                self.ideology_model,
                self.propaganda_model,
                self.emotion_model,
            ]:
                if model is not None:
                    model.half()

        if self.device.type == "cuda":
            for model in [
                self.bias_model,
                self.ideology_model,
                self.propaganda_model,
                self.emotion_model,
            ]:
                if model is not None:
                    try:
                        compiled = torch.compile(model, mode="max-autotune")
                        if model is self.bias_model:
                            self.bias_model = compiled
                        elif model is self.ideology_model:
                            self.ideology_model = compiled
                        elif model is self.propaganda_model:
                            self.propaganda_model = compiled
                        elif model is self.emotion_model:
                            self.emotion_model = compiled
                    except Exception:
                        logger.warning("torch.compile failed")

        logger.info("PredictionPipeline initialized on device: %s", self.device)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def _forward_all(self, features: torch.Tensor) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            if self.bias_model is not None:
                outputs["bias"] = self.bias_model(features)

            if self.ideology_model is not None:
                outputs["ideology"] = self.ideology_model(features)

            if self.propaganda_model is not None:
                outputs["propaganda"] = self.propaganda_model(features)

            if self.emotion_model is not None:
                outputs["emotion"] = self.emotion_model(features)

        return outputs

    def _extract_logits(self, outputs: Any) -> torch.Tensor:
        if isinstance(outputs, dict):
            if "logits" in outputs:
                return outputs["logits"]
            if "probabilities" in outputs:
                return torch.log(outputs["probabilities"] + 1e-9)
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, torch.Tensor):
            return outputs
        raise RuntimeError("Model output missing logits")

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

    def _softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def _sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def _predict_class(self, probs: torch.Tensor) -> int:
        return int(torch.argmax(probs, dim=-1).item())

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

        bias_labels = {0: "non_bias", 1: "bias"}
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

        ideology_labels = {0: "left", 1: "center", 2: "right"}
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

        if bias == "non_bias":
            bias_score = 1.0
        elif bias == "bias":
            bias_score = 0.5

        if ideology == "center":
            ideology_score = 1.0
        elif ideology in {"left", "right"}:
            ideology_score = 0.6

        if propaganda_prob is not None:
            propaganda_score = 1.0 - propaganda_prob

        if emotion:
            values = np.array(list(emotion.values()))
            max_intensity = float(np.max(values))
            eps = 1e-9
            entropy = -np.sum(values * np.log(values + eps))
            n = len(values)
            normalized_entropy = entropy / np.log(n) if n > 1 else 0.0
            emotion_score = (
                0.5 * (1.0 - max_intensity) + 0.5 * normalized_entropy
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

    def _to_label_from_prediction_tensor(
        self,
        prediction_tensor: Optional[torch.Tensor],
        label_map: Dict[int, str],
    ) -> Optional[str]:
        if not isinstance(prediction_tensor, torch.Tensor):
            return None
        if prediction_tensor.numel() == 0:
            return None
        idx = int(prediction_tensor.reshape(-1)[0].item())
        return label_map.get(idx, "unknown")

    def _to_probability_from_task(
        self,
        task: Any,
        positive_index: int,
    ) -> Optional[float]:
        probs = getattr(task, "probabilities", None)
        if not isinstance(probs, torch.Tensor):
            return None
        if probs.numel() == 0:
            return None
        if probs.dim() == 1:
            if positive_index >= probs.size(0):
                return None
            return float(probs[positive_index].item())
        if positive_index >= probs.size(-1):
            return None
        return float(probs.reshape(-1, probs.size(-1))[0, positive_index].item())

    def _to_emotion_distribution_from_task(
        self,
        task: Any,
    ) -> Optional[Dict[str, float]]:
        probs = getattr(task, "probabilities", None)
        if not isinstance(probs, torch.Tensor):
            return None

        probs_flat = probs.reshape(-1, probs.size(-1))[0] if probs.dim() > 1 else probs

        emotions: Dict[str, float] = {}
        for idx, label in enumerate(EMOTION_LABELS):
            if idx < probs_flat.size(0):
                emotions[label] = float(probs_flat[idx].item())
            else:
                emotions[label] = 0.0
        return emotions

    def _adapt_structured_output(
        self,
        structured_output: PredictionOutput,
    ) -> Dict[str, Any]:
        bias_task = structured_output.tasks.get("bias")
        ideology_task = structured_output.tasks.get("ideology")
        propaganda_task = structured_output.tasks.get("propaganda")
        emotion_task = structured_output.tasks.get("emotion")

        bias = self._to_label_from_prediction_tensor(
            getattr(bias_task, "predictions", None),
            {0: "non_bias", 1: "bias"},
        )
        ideology = self._to_label_from_prediction_tensor(
            getattr(ideology_task, "predictions", None),
            {0: "left", 1: "center", 2: "right"},
        )

        propaganda_prob = self._to_probability_from_task(
            propaganda_task,
            positive_index=1,
        )
        emotion = self._to_emotion_distribution_from_task(emotion_task)

        credibility_score, explanation = self._compute_credibility_score(
            bias=bias,
            propaganda_prob=propaganda_prob,
            emotion=emotion,
            ideology=ideology,
        )

        return {
            "bias": bias,
            "ideology": ideology,
            "propaganda_probability": propaganda_prob,
            "emotion": emotion,
            "credibility_score": credibility_score,
            "credibility_explanation": explanation,
            "structured_prediction": structured_output.to_dict(),
        }

    def _build_aggregation_profile(
        self,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.aggregation_pipeline.build_profile_from_prediction(prediction)

    def predict(self, features: torch.Tensor) -> Dict[str, Any]:

        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a torch.Tensor")

        if features.device.type == "cpu":
            features = features.pin_memory()
        features = features.to(self.device, non_blocking=True)

        with torch.inference_mode():
            outputs = self._forward_all(features)

            bias = None
            if "bias" in outputs:
                logits = self._extract_logits(outputs["bias"])
                preds = torch.argmax(logits, dim=-1)
                preds_cpu = preds.detach().cpu().numpy()
                bias = ["non_bias" if p == 0 else "bias" for p in preds_cpu.tolist()]

            ideology = None
            if "ideology" in outputs:
                logits = self._extract_logits(outputs["ideology"])
                preds = torch.argmax(logits, dim=-1)
                mapping = ["left", "center", "right"]
                preds_cpu = preds.detach().cpu().numpy()
                ideology = [mapping[p] for p in preds_cpu.tolist()]

            propaganda_prob = None
            if "propaganda" in outputs:
                if self.config.return_probabilities:
                    logits = self._extract_logits(outputs["propaganda"])
                    if logits.size(-1) == 2:
                        propaganda_prob = torch.sigmoid(logits[:, 1] - logits[:, 0])
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        propaganda_prob = probs[:, 1]
                    propaganda_prob = propaganda_prob.detach()

            emotion = None
            if "emotion" in outputs:
                if self.config.return_probabilities:
                    logits = self._extract_logits(outputs["emotion"])
                    probs = torch.sigmoid(logits)
                    emotion = [
                        dict(zip(EMOTION_LABELS, row.tolist()))
                        for row in probs
                    ]

            credibility_scores: List[float] = []
            explanations: List[Dict[str, float]] = []

            batch_size = int(features.size(0))
            for i in range(batch_size):
                score, exp = self._compute_credibility_score(
                    bias=bias[i] if bias else None,
                    propaganda_prob=(
                        float(propaganda_prob[i]) if propaganda_prob is not None else None
                    ),
                    emotion=emotion[i] if emotion else None,
                    ideology=ideology[i] if ideology else None,
                )
                credibility_scores.append(score)
                explanations.append(exp)

        result: Dict[str, Any] = {
            "bias": bias,
            "ideology": ideology,
            "propaganda_probability": (
                propaganda_prob.cpu().tolist() if propaganda_prob is not None else None
            ),
            "emotion": emotion,
            "credibility_score": credibility_scores,
            "credibility_explanation": explanations,
        }

        logger.debug("Prediction result: %s", result)
        return result

    def predict_single(self, features: torch.Tensor) -> Dict[str, Any]:
        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a torch.Tensor")
        if features.ndim == 1:
            features = features.unsqueeze(0)
        out = self.predict(features)
        return {
            "bias": out["bias"][0] if out.get("bias") else None,
            "ideology": out["ideology"][0] if out.get("ideology") else None,
            "propaganda_probability": out["propaganda_probability"][0] if out.get("propaganda_probability") else None,
            "emotion": out["emotion"][0] if out.get("emotion") else None,
            "credibility_score": out["credibility_score"][0] if out.get("credibility_score") else None,
            "credibility_explanation": out["credibility_explanation"][0] if out.get("credibility_explanation") else None,
        }

    def export_onnx(self, path: str) -> None:
        model = self.bias_model or self.ideology_model or self.propaganda_model or self.emotion_model
        if model is None:
            raise RuntimeError("No model available for ONNX export")

        dummy = torch.randn(1, 768, device=self.device)

        torch.onnx.export(
            model,
            dummy,
            path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}},
            opset_version=17,
        )

        logger.info("ONNX export completed: %s", path)

    # -----------------------------------------------------------------
    # Prediction with Explanation
    # -----------------------------------------------------------------

    def predict_with_explanation(
        self,
        features: torch.Tensor,
        text: str,
        predict_fn: Callable[[str], Dict[str, Any]],
        analysis_modules: Optional[Dict[str, Any]] = None,
        tokens: Optional[List[str]] = None,
        attentions: Optional[List[torch.Tensor]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run prediction and attach a full explainability report.

        Combines the model's structured prediction output with token-level
        attributions from LIME, SHAP, attention rollout, and propaganda
        gradient attribution. Results are aggregated and consistency-checked
        before being returned alongside the base prediction.

        Parameters
        ----------
        features : torch.Tensor
            Prepared feature tensor for the task models.
        text : str
            Raw article text (used by LIME, SHAP, and cache).
        predict_fn : Callable
            Text-in / prediction-dict-out function (used by explainers).
        tokens : list of str, optional
            Pre-tokenized token list aligned with model inputs.
        attentions : list of torch.Tensor, optional
            Per-layer attention tensors for attention rollout.
        input_ids : torch.Tensor, optional
            Token id tensor for propaganda explainer.
        attention_mask : torch.Tensor, optional
            Attention mask tensor for propaganda explainer.
        model : optional
            Transformer model for bias/emotion explanation.
        tokenizer : optional
            Tokenizer for bias/emotion explanation.

        Returns
        -------
        Dict with prediction output merged with 'explainability' key.
        """

        prediction = self.predict_with_aggregation(
            features,
            text=text,
            analysis_modules=analysis_modules,
        )

        if self.explainability_layer is None:
            logger.debug("No ExplainabilityLayer configured; returning prediction only")
            return prediction

        try:
            explanation = self.explainability_layer.explain(
                text=text,
                predict_fn=predict_fn,
                tokens=tokens,
                attentions=attentions,
                input_ids=input_ids,
                attention_mask=attention_mask,
                model=model,
                tokenizer=tokenizer,
            )
            prediction["explainability"] = explanation
        except Exception as exc:
            logger.warning("Explainability layer failed: %s", exc)
            prediction["explainability"] = {}

        return prediction

    # -----------------------------------------------------------------
    # Prediction with Aggregation
    # -----------------------------------------------------------------

    def predict_with_aggregation(
        self,
        features: torch.Tensor,
        *,
        text: Optional[str] = None,
        analysis_modules: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run prediction and enrich with full aggregation pipeline output.

        Calls ``predict()`` to obtain raw task outputs, assembles a minimal
        feature profile, then runs ``AggregationPipeline.run()`` to produce
        weighted scores, categorical risk levels, and score explanations.
        The result merges the base prediction with an ``aggregation`` key
        containing the full pipeline output.

        Parameters
        ----------
        features : torch.Tensor
            Prepared feature tensor for the task models.
        text : str, optional
            Raw article text forwarded to ``AggregationPipeline.run()``
            so that analysis modules can be resolved when *analysis_modules*
            is not provided.
        analysis_modules : dict, optional
            Pre-computed analysis-module outputs to inject into the profile
            before aggregation.

        Returns
        -------
        Dict merging prediction output with an ``aggregation`` key containing
        ``scores``, ``raw_scores``, ``risks``, ``explanations``, and
        ``analysis_modules``.
        """

        prediction = self.predict(features)
        profile = self._build_aggregation_profile(prediction)

        try:
            agg_result = self.aggregation_pipeline.run(
                profile,
                text=text,
                analysis_modules=analysis_modules,
            )
        except Exception as exc:
            logger.warning("AggregationPipeline.run() failed: %s", exc)
            agg_result = {}

        result = dict(prediction)
        result["aggregation"] = agg_result

        logger.debug(
            "predict_with_aggregation completed | scores=%s",
            list(agg_result.get("scores", {}).keys()),
        )
        return result

    # -----------------------------------------------------------------
    # Feature-dict entry point
    # -----------------------------------------------------------------

    def predict_from_feature_dict(
        self,
        feature_dict: Dict[str, Any],
        *,
        feature_preparer: FeaturePreparer,
    ) -> Dict[str, Any]:
        """
        Integrate FeaturePreparer path directly into prediction pipeline.
        """

        prepared = feature_preparer.prepare_single(feature_dict)

        if isinstance(prepared, torch.Tensor):
            tensor = prepared
        else:
            tensor = torch.tensor(prepared, dtype=torch.float32)

        text = feature_dict.get("text") if isinstance(feature_dict, dict) else None
        analysis_modules = (
            feature_dict.get("analysis_modules")
            if isinstance(feature_dict, dict)
            else None
        )

        if isinstance(text, str) and text.strip():
            return self.predict_with_aggregation(
                tensor,
                text=text,
                analysis_modules=(
                    analysis_modules
                    if isinstance(analysis_modules, dict)
                    else None
                ),
            )

        return self.predict(tensor)

    def predict_from_structured_output(
        self,
        structured_output: PredictionOutput | Dict[str, Any],
        *,
        text: Optional[str] = None,
        analysis_modules: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if isinstance(structured_output, dict):
            structured = PredictionOutput.from_raw_outputs(structured_output)
        elif isinstance(structured_output, PredictionOutput):
            structured = structured_output
        else:
            raise TypeError(
                "structured_output must be PredictionOutput or raw output dict"
            )

        prediction = self._adapt_structured_output(structured)
        profile = self._build_aggregation_profile(prediction)

        try:
            aggregation = self.aggregation_pipeline.run(
                profile,
                text=text,
                analysis_modules=analysis_modules,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("AggregationPipeline.run() failed for structured output: %s", exc)
            aggregation = {}

        result = dict(prediction)
        result["aggregation"] = aggregation
        return result
