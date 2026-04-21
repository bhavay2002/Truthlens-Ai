from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.features.emotion.emotion_schema import EMOTION_LABELS
from src.inference.feature_preparer import FeaturePreparer
from src.models.inference.prediction_output import PredictionOutput

from src.explainability.explanation_aggregator import AggregationWeights
from src.explainability.orchestrator import (
    ExplainabilityConfig,
    ExplainabilityOrchestrator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class PredictionPipelineConfig:

    device: str = "cpu"
    return_probabilities: bool = True 


# Backward-compatibility alias.
# ExplainabilityLayer has been consolidated into ExplainabilityOrchestrator.
# New code should import ExplainabilityOrchestrator directly from
# src.explainability.orchestrator.
ExplainabilityLayer = ExplainabilityOrchestrator


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
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.bias_model = bias_model
        self.ideology_model = ideology_model
        self.propaganda_model = propaganda_model
        self.emotion_model = emotion_model
        self.explainability_layer = explainability_layer
        self.aggregation_pipeline = aggregation_pipeline or AggregationPipeline()

        for attr in ["bias_model", "ideology_model", "propaganda_model", "emotion_model"]:
            model = getattr(self, attr)

            if model is None:
                continue

            model = model.to(self.device)
            model.eval()

            if self.device.type == "cuda" and torch.cuda.is_available():
                model = model.half()
                try:
                    compiled_model = torch.compile(model, mode="reduce-overhead")
                    model = compiled_model
                except Exception:
                    logger.debug("torch.compile skipped for %s", attr)

            setattr(self, attr, model)

        logger.info("PredictionPipeline initialized on device: %s", self.device)

    # -----------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------

    def _forward_all(self, features: torch.Tensor) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {}

        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            with ctx:
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
                return outputs["probabilities"]
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, torch.Tensor):
            return outputs
        raise RuntimeError("Model output missing logits")

    def _safe_tensor(self, t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)

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

        return {
            "bias": bias,
            "ideology": ideology,
            "propaganda_probability": propaganda_prob,
            "emotion": emotion,
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

        features = features.to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            features = features.to(dtype=torch.float16, non_blocking=True)
        else:
            features = features.to(dtype=torch.float32)

        with torch.inference_mode():
            outputs = self._forward_all(features)

            bias_preds = None
            ideology_preds = None
            propaganda_prob = None
            emotion_probs = None

            if "bias" in outputs:
                logits = self._extract_logits(outputs["bias"])
                bias_preds = torch.argmax(logits, dim=-1)

            if "ideology" in outputs:
                logits = self._extract_logits(outputs["ideology"])
                ideology_preds = torch.argmax(logits, dim=-1)

            if "propaganda" in outputs and self.config.return_probabilities:
                raw = self._extract_logits(outputs["propaganda"])
                if raw.dim() == 1:
                    raw = raw.unsqueeze(0)

                probs = torch.softmax(raw, dim=-1)

                if probs.size(-1) >= 2:
                    propaganda_prob = probs[:, 1]
                else:
                    propaganda_prob = probs[:, 0]

                propaganda_prob = self._safe_tensor(propaganda_prob)

            if "emotion" in outputs and self.config.return_probabilities:
                logits = self._extract_logits(outputs["emotion"])
                probs = torch.sigmoid(logits)
                probs = self._safe_tensor(probs)
                emotion_probs = probs.reshape(-1, probs.size(-1))

            batch_size = int(features.shape[0])

            def _check(t: Optional[torch.Tensor]) -> None:
                if t is not None and t.shape[0] != batch_size:
                    raise RuntimeError("Batch size mismatch in model outputs")

            _check(bias_preds)
            _check(ideology_preds)
            _check(propaganda_prob)
            _check(emotion_probs)

        bias: List[str] = []
        ideology: List[str] = []
        emotion: List[Dict[str, float]] = []
        propaganda_list = None

        if bias_preds is not None:
            bias_list = bias_preds.detach().cpu().numpy().reshape(-1)
            bias = ["non_bias" if p == 0 else "bias" for p in bias_list]

        if ideology_preds is not None:
            ideology_list = ideology_preds.detach().cpu().numpy().reshape(-1)
            mapping = ["left", "center", "right"]
            ideology = [
                mapping[p] if 0 <= int(p) < len(mapping) else "unknown"
                for p in ideology_list
            ]

        if propaganda_prob is not None:
            propaganda_list = propaganda_prob.detach().cpu().numpy().reshape(-1).tolist()

        if emotion_probs is not None:
            probs_list = emotion_probs.detach().cpu().tolist()
            emotion = [
                {
                    label: float(row[i]) if i < len(row) else 0.0
                    for i, label in enumerate(EMOTION_LABELS)
                }
                for row in probs_list
            ]

        result: Dict[str, Any] = {
            "bias": bias,
            "ideology": ideology,
            "propaganda_probability": propaganda_list,
            "emotion": emotion,
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
        }

    def predict_batch_safe(
        self,
        features: torch.Tensor,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        if not isinstance(features, torch.Tensor):
            raise TypeError("Features must be a torch.Tensor")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if features.ndim == 1:
            features = features.unsqueeze(0)

        outputs: List[Dict[str, Any]] = []
        total = int(features.shape[0])
        for i in range(0, total, batch_size):
            batch = features[i : i + batch_size]
            outputs.append(self.predict(batch))
        return outputs

    def export_onnx(self, path: str) -> None:
        model = self.bias_model or self.ideology_model or self.propaganda_model or self.emotion_model
        if model is None:
            raise RuntimeError("No model available for ONNX export")

        input_dim = next(model.parameters()).shape[-1]
        dummy = torch.randn(1, input_dim, device=self.device)

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

        scores = agg_result.get("scores") if isinstance(agg_result, dict) else None

        if not isinstance(scores, dict):
            scores = {}

        result["credibility_score"] = float(
            scores.get("truthlens_credibility_score", 0.0)
        )
        result["manipulation_risk"] = float(
            scores.get("truthlens_manipulation_risk", 0.0)
        )
        result["final_score"] = float(
            scores.get("truthlens_final_score", 0.0)
        )

        logger.debug(
            "predict_with_aggregation completed | scores=%s",
            list(scores.keys()) if isinstance(scores, dict) else [],
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
