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

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.inference.feature_preparer import FeaturePreparer
from src.inference.prediction_pipeline import PredictionPipeline
from src.inference.report_generator import ReportGenerator
from src.models.calibration import IsotonicCalibrator, TemperatureScaler
from src.utils import (
    ensure_file_exists,
    ensure_non_empty_text_list,
    get_device,
    load_json,
    move_to_device,
)

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
    use_amp: bool = True


@dataclass
class PredictionResult:
    """
    Structured prediction result.
    """
    text: str
    predicted_label: Union[int, str]
    probabilities: Optional[List[float]] = None
    calibrated_probabilities: Optional[List[float]] = None
    ensemble_probabilities: Optional[List[float]] = None
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

    def __init__(
        self,
        config: InferenceConfig,
        *,
        feature_preparer: Optional[FeaturePreparer] = None,
        prediction_pipeline: Optional[PredictionPipeline] = None,
        report_generator: Optional[ReportGenerator] = None,
        article_analyzer: Optional[Any] = None,
        temperature_scaler: Optional[TemperatureScaler] = None,
        isotonic_calibrator: Optional[IsotonicCalibrator] = None,
        ensemble_model: Optional[torch.nn.Module] = None,
    ) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        self.use_amp = (self.device.type == "cuda") and getattr(config, "use_amp", True)
        if self.device.type == "cuda":
            self.amp_dtype = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )
        else:
            self.amp_dtype = torch.float32
        self.use_compile = True
        self.model = None
        self.tokenizer = None
        self.label_map: Optional[Dict[int, str]] = None
        self.feature_preparer = feature_preparer
        self.prediction_pipeline = prediction_pipeline
        self.report_generator = report_generator or ReportGenerator()
        self.article_analyzer = article_analyzer
        self.temperature_scaler = temperature_scaler
        self.isotonic_calibrator = isotonic_calibrator
        self.ensemble_model = ensemble_model
        if self.ensemble_model is not None:
            self.ensemble_model.to(self.device)
            self.ensemble_model.eval()

        self._load_model()

    def _resolve_device(self, device: str) -> torch.device:
        """
        Resolve device configuration.
        """
        if device == "auto":
            resolved = get_device(prefer_gpu=True)
        else:
            resolved = torch.device(device)

        logger.info("Using device: %s", resolved)
        return resolved

    def _load_model(self) -> None:
        """
        Load model and tokenizer from disk.
        """
        model_path = Path(self.config.model_path)

        ensure_file_exists(model_path)

        tokenizer_path = (
            self.config.tokenizer_path
            if self.config.tokenizer_path
            else self.config.model_path
        )

        try:
            logger.info("Loading tokenizer from %s", tokenizer_path)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

            logger.info("Loading model from %s", model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=self.amp_dtype if self.device.type == "cuda" else None,
                low_cpu_mem_usage=True,
            )

            if self.device.type == "cuda":
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(True)
                if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                    torch.backends.cuda.enable_mem_efficient_sdp(True)

            self.model.to(self.device, non_blocking=True)

            if self.use_compile:
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                except Exception:
                    logger.debug("torch.compile skipped")
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
                raw_map = load_json(label_map_path)

                self.label_map = {int(k): v for k, v in raw_map.items()}
                logger.info("Loaded label map")

            except Exception as exc:
                logger.warning("Failed to load label map: %s", exc)

    def _validate_input(self, texts: Union[str, List[str]]) -> List[str]:
        """
        Validate input texts.
        """
        if isinstance(texts, str):
            return ensure_non_empty_text_list([texts], name="texts")
        return ensure_non_empty_text_list(texts, name="texts")

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

        with torch.inference_mode():
            for batch in batches:
                encoded = self.tokenizer(
                    batch,
                    padding="longest",
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )

                if self.device.type == "cuda":
                    for key in encoded:
                        encoded[key] = encoded[key].pin_memory().to(
                            self.device,
                            non_blocking=True,
                        )
                else:
                    for key in encoded:
                        encoded[key] = encoded[key].to(self.device)

                if self.use_amp:
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                        enabled=True,
                    ):
                        logits = self._compute_logits(encoded)
                else:
                    logits = self._compute_logits(encoded)

                needs_probs = (
                    self.config.return_probabilities
                    or self.temperature_scaler is not None
                    or self.isotonic_calibrator is not None
                    or self.ensemble_model is not None
                )

                if needs_probs:
                    probabilities = torch.softmax(logits, dim=-1)
                    calibrated_probabilities = self._apply_calibration(logits, probabilities)
                else:
                    probabilities = None
                    calibrated_probabilities = None

                ensemble_probabilities = self._apply_ensemble(encoded)
                if ensemble_probabilities is not None:
                    predicted_indices = torch.argmax(ensemble_probabilities, dim=-1)
                elif needs_probs:
                    predicted_indices = torch.argmax(calibrated_probabilities, dim=-1)
                else:
                    predicted_indices = torch.argmax(logits, dim=-1)

                batch_logits_cpu = logits.detach().cpu()
                batch_preds_cpu = predicted_indices.detach().cpu()

                batch_probs_cpu = (
                    probabilities.detach().cpu() if probabilities is not None else None
                )
                batch_calibrated_cpu = (
                    calibrated_probabilities.detach().cpu()
                    if calibrated_probabilities is not None
                    else None
                )
                batch_ensemble_cpu = (
                    ensemble_probabilities.detach().cpu()
                    if ensemble_probabilities is not None
                    else None
                )

                for i, text in enumerate(batch):
                    label_idx = int(batch_preds_cpu[i].item())

                    label = (
                        self.label_map[label_idx]
                        if self.label_map
                        else label_idx
                    )

                    probs = (
                        batch_probs_cpu[i].tolist()
                        if batch_probs_cpu is not None and self.config.return_probabilities
                        else None
                    )
                    calibrated_probs = (
                        batch_calibrated_cpu[i].tolist()
                        if batch_calibrated_cpu is not None and self.config.return_probabilities
                        else None
                    )
                    ensemble_probs = (
                        batch_ensemble_cpu[i].tolist()
                        if batch_ensemble_cpu is not None and self.config.return_probabilities
                        else None
                    )
                    logit_values = batch_logits_cpu[i].tolist()

                    result = PredictionResult(
                        text=text,
                        predicted_label=label,
                        probabilities=probs if self.config.return_probabilities else None,
                        calibrated_probabilities=(
                            calibrated_probs if self.config.return_probabilities else None
                        ),
                        ensemble_probabilities=(
                            ensemble_probs if self.config.return_probabilities else None
                        ),
                        logits=logit_values,
                    )

                    results.append(result)

        return results

    def export_onnx(self, output_path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        if not output_path:
            raise ValueError("output_path must be non-empty")

        dummy = self.tokenizer(
            ["ONNX export"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.config.max_length,
        )

        input_ids = dummy["input_ids"].to(self.device)
        attention_mask = dummy.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        args = (input_ids,) if attention_mask is None else (input_ids, attention_mask)

        torch.onnx.export(
            self.model,
            args=args,
            f=output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
            },
            opset_version=17,
        )

        logger.info("ONNX model exported to %s", output_path)

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

    def predict_from_feature_dict(
        self,
        features: Dict[str, Any],
        *,
        article_text: Optional[str] = None,
        title: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Integrate FeaturePreparer + src.inference.prediction_pipeline + ReportGenerator.
        """
        if self.feature_preparer is None:
            raise RuntimeError("FeaturePreparer not configured")
        if self.prediction_pipeline is None:
            raise RuntimeError("PredictionPipeline not configured")

        prepared = self.feature_preparer.prepare_single(features)
        if isinstance(prepared, torch.Tensor):
            tensor = prepared
        else:
            tensor = torch.tensor(prepared, dtype=torch.float32)

        prediction = self.prediction_pipeline.predict(tensor)

        report = self.report_generator.generate_report(
            article_text=article_text or str(features.get("text", "")),
            title=title,
            source=source,
            bias_analysis={"bias": prediction.get("bias")},
            emotion_analysis={"emotion": prediction.get("emotion")},
            narrative_structure={},
            entity_graph={},
            credibility_score=prediction.get("credibility_score"),
        )

        return {
            "prediction": prediction,
            "report": report,
        }

    def analyze_text(
        self,
        text: str,
        *,
        title: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Integrate src.inference.analyze_article + ReportGenerator via engine.
        """
        if self.article_analyzer is None:
            raise RuntimeError("ArticleAnalyzer not configured")

        analysis = self.article_analyzer.analyze(text)

        report = self.report_generator.generate_report(
            article_text=text,
            title=title,
            source=source,
            bias_analysis=analysis.get("bias_features", {}),
            emotion_analysis=analysis.get("emotion_features", {}),
            narrative_structure=analysis.get("narrative_features", {}),
            entity_graph=analysis.get("graph_pipeline", {}),
            credibility_score=analysis.get("scores", {}).get("truthlens_credibility_score"),
        )

        return {
            "analysis": analysis,
            "report": report,
        }

    def set_temperature_scaler(self, scaler: TemperatureScaler) -> None:
        self.temperature_scaler = scaler

    def set_isotonic_calibrator(self, calibrator: IsotonicCalibrator) -> None:
        self.isotonic_calibrator = calibrator

    def set_ensemble_model(self, ensemble_model: torch.nn.Module) -> None:
        self.ensemble_model = ensemble_model
        self.ensemble_model.to(self.device)
        self.ensemble_model.eval()

    def _apply_calibration(
        self,
        logits: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> torch.Tensor:
        if self.temperature_scaler is not None:
            try:
                calibrated = self.temperature_scaler.predict_proba(
                    logits.to(self.temperature_scaler.device)
                )
                return calibrated.to(probabilities.device)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Temperature scaling skipped in InferenceEngine: %s", exc)

        if self.isotonic_calibrator is not None:
            try:
                probs_np = probabilities.detach().cpu().numpy().astype(np.float64)
                calibrated_np = self.isotonic_calibrator.predict_proba(probs_np)
                return torch.tensor(
                    calibrated_np,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Isotonic calibration skipped in InferenceEngine: %s", exc)

        return probabilities

    def _compute_logits(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        outputs = self.model(**encoded)
        if not hasattr(outputs, "logits"):
            raise RuntimeError("Inference model output missing logits")
        return outputs.logits

    def _apply_ensemble(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.ensemble_model is None:
            return None

        try:
            maybe_outputs = self.ensemble_model(**encoded)
            if isinstance(maybe_outputs, torch.Tensor):
                logits = maybe_outputs
            elif isinstance(maybe_outputs, dict) and "logits" in maybe_outputs:
                logits = maybe_outputs["logits"]
            else:
                logits = None
        except Exception:  # noqa: BLE001
            logits = None

        if logits is None:
            if "input_ids" in encoded and isinstance(encoded["input_ids"], torch.Tensor):
                logits = self.ensemble_model(encoded["input_ids"])
            else:
                logger.warning("Ensemble model skipped: incompatible input signature.")
                return None

        if not isinstance(logits, torch.Tensor):
            logger.warning("Ensemble model output is not a tensor. Skipping ensemble.")
            return None

        return torch.softmax(logits, dim=-1)
