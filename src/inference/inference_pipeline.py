from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.config.task_config import TASK_CONFIG
from src.inference.postprocessing import Postprocessor
from src.explainability.orchestrator import ExplainabilityOrchestrator
from src.graph.graph_pipeline import GraphPipeline, get_default_pipeline

from src.monitoring.feature_logger import (
    log_request_latency,
    log_failure,
    time_block,
)

import time

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class PredictionPipelineConfig:
    device: str = "cpu"
    return_probabilities: bool = True


ExplainabilityLayer = ExplainabilityOrchestrator


# =========================================================
# PIPELINE
# =========================================================

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

        self.postprocessor = Postprocessor()

        # 🔥 NEW — G-R1: share the process-wide singleton.
        self.graph_pipeline = get_default_pipeline()

        for name in ["bias_model", "ideology_model", "propaganda_model", "emotion_model"]:
            model = getattr(self, name)
            if model:
                model.to(self.device)
                model.eval()

        logger.info("PredictionPipeline initialized")

    # =====================================================
    # CORE FORWARD
    # =====================================================

    def _forward_all(self, features: torch.Tensor) -> Dict[str, Any]:

        outputs = {}

        ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with torch.no_grad():
            with ctx:

                if self.bias_model:
                    outputs["bias"] = self.bias_model(features)

                if self.ideology_model:
                    outputs["ideology"] = self.ideology_model(features)

                if self.propaganda_model:
                    outputs["propaganda"] = self.propaganda_model(features)

                if self.emotion_model:
                    outputs["emotion"] = self.emotion_model(features)

        return outputs

    def _extract_logits(self, out):

        if isinstance(out, dict) and "logits" in out:
            return out["logits"]

        if hasattr(out, "logits"):
            return out.logits

        if isinstance(out, torch.Tensor):
            return out

        raise RuntimeError("Invalid model output")

    # =====================================================
    # MULTI-TASK
    # =====================================================
    #
    # CRIT-3: per-task output type is driven by ``TASK_CONFIG`` (i.e. the
    # ``tasks:`` block in ``config/config.yaml``) — never by hardcoded
    # constants in this file. The previous ``_BINARY_TASKS = {"propaganda"}``
    # contradicted the YAML (which marks propaganda as ``multiclass``) and
    # silently produced wrong predictions.

    _MULTILABEL_THRESHOLD = 0.5

    def _resolve_task_type(self, task: str) -> str:
        try:
            return str(TASK_CONFIG[task]["type"])
        except (KeyError, TypeError):
            logger.warning("No task type registered for %s; defaulting to multiclass", task)
            return "multiclass"

    def predict_multitask(self, features: torch.Tensor) -> Dict[str, Any]:

        outputs = self._forward_all(features)

        results = {}

        for task, out in outputs.items():

            logits = self._extract_logits(out)

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            task_type = self._resolve_task_type(task)

            if task_type == "multilabel":
                probs = torch.sigmoid(logits)
                preds = (probs >= self._MULTILABEL_THRESHOLD).int()
            elif task_type == "binary":
                probs = torch.sigmoid(logits)
                if logits.shape[-1] == 1:
                    preds = (probs >= self._MULTILABEL_THRESHOLD).int().squeeze(-1)
                else:
                    preds = torch.argmax(probs, dim=-1)
            else:
                # multiclass
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            results[task] = {
                "logits": logits.detach().cpu().numpy(),
                "probabilities": probs.detach().cpu().numpy(),
                "predictions": preds.detach().cpu().numpy(),
            }

        return results

    # =====================================================
    # POSTPROCESSING
    # =====================================================

    def predict_with_postprocessing(
        self,
        features: torch.Tensor,
        *,
        task_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:

        raw = self.predict_multitask(features)

        # CRIT-8: drive the postprocessor's task types from the task
        # registry whenever the caller does not override them, so the
        # multilabel/multiclass branches stay in lockstep with the YAML.
        if task_types is None:
            task_types = {task: self._resolve_task_type(task) for task in raw}

        return self.postprocessor.process(
            raw,
            task_types=task_types,
        )

    # =====================================================
    # MAIN PREDICT
    # =====================================================

    def predict(self, features: torch.Tensor) -> Dict[str, Any]:

        processed = self.predict_with_postprocessing(features)

        batch_size = int(features.shape[0])

        result = {
            "bias": [],
            "ideology": [],
            "propaganda_probability": [],
            "emotion": [],
        }

        for i in range(batch_size):

            for task, out in processed.items():

                if task == "bias":
                    result["bias"].append(out["labels"][i])

                elif task == "ideology":
                    result["ideology"].append(out["labels"][i])

                elif task == "propaganda":
                    prob = out["probabilities"][i]
                    result["propaganda_probability"].append(float(prob[1]))

                elif task == "emotion":
                    result["emotion"].append(out["probabilities"][i].tolist())

        return result

    # =====================================================
    #  NEW FULL OUTPUT (SERVICE READY)
    # =====================================================

    def predict_full(
        self,
        features: torch.Tensor,
        *,
        text: Optional[str] = None,
        predict_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
       ) -> Dict[str, Any]:
    
        start_total = time.time()
    
        try:
            # ---------------- BASE PREDICTION ----------------
            with time_block("model_prediction", task="inference"):
                prediction = self.predict(features)
    
            # ---------------- AGGREGATION ----------------
            with time_block("aggregation", task="inference"):
                profile = self.aggregation_pipeline.build_profile_from_prediction(prediction)
                aggregation = self.aggregation_pipeline.run(profile, text=text)
    
            scores = aggregation.get("raw_scores", {})
    
            # ---------------- GRAPH ----------------
            graph_output = None
            if text:
                try:
                    with time_block("graph_pipeline", task="inference"):
                        graph_output = self.graph_pipeline.run(text)
                except Exception as e:
                    log_failure(e, context={"stage": "graph_pipeline"})
                    logger.warning("Graph pipeline failed: %s", e)
    
            # ---------------- EXPLAINABILITY ----------------
            explanation = {}
            if self.explainability_layer and text and predict_fn:
                try:
                    with time_block("explainability", task="inference"):
                        explanation = self.explainability_layer.explain(
                            text=text,
                            predict_fn=predict_fn,
                        )
                except Exception as e:
                    log_failure(e, context={"stage": "explainability"})
                    logger.warning("Explainability failed: %s", e)
    
            # ---------------- TOTAL LATENCY ----------------
            total_latency = time.time() - start_total
    
            log_request_latency(
                total_latency,
                task="full_inference",
            )
    
            # ---------------- FINAL OUTPUT ----------------
            return {
                "prediction": prediction,
                "scores": scores,
                "analysis_modules": {
                    "graph": graph_output,
                    "graph_explanation": explanation.get("graph_explanation"),
                },
                "explanation": explanation,
                "meta": {
                    "total_latency": round(total_latency, 4),
                },
            }
    
        except Exception as e:
            log_failure(
                e,
                context={
                    "stage": "predict_full",
                    "has_text": text is not None,
                },
            )
            raise