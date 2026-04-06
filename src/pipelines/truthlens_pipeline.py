"""
File Name: truthlens_pipeline.py
Module: TruthLens Pipeline - Unified Analysis Pipeline
Description:
    Implements the central orchestration pipeline for the TruthLens AI system.
    The pipeline coordinates preprocessing, feature extraction, model prediction,
    analytical modules, aggregation logic, and report generation.

    Processing Flow:
        Article
        ↓
        Preprocessing
        ↓
        Feature Extraction
        ↓
        Model Prediction
        ↓
        Analysis Modules
        ↓
        Aggregation
        ↓
        Output Report

    This module serves as the primary entry point for TruthLens article analysis
    and produces a complete structured report containing intermediate outputs,
    predictions, and final TruthLens scores.

Author: TruthLens Engineering Team
Date: 2026-04-03
Dependencies:
    logging
    typing
    dataclasses
    time

Inputs:
    Raw article text

Outputs:
    Structured TruthLens analysis report
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.pipelines.prediction_pipeline import Predictor
from src.pipelines.feature_pipeline import FeaturePipeline
from src.pipelines.preprocessing_pipeline import PreprocessingPipeline
from src.features.base.base_feature import FeatureContext


logger = logging.getLogger(__name__)


class EmotionPipeline:
    """
    Lightweight emotion analysis pipeline used by TruthLensPipeline.
    """

    def __init__(self) -> None:
        self._extractors = []

        try:
            from src.features.emotion.emotion_features import EmotionFeatures
            from src.features.emotion.emotion_intensity_features import (
                EmotionIntensityFeatures,
            )
            from src.features.emotion.emotion_lexicon_features import (
                EmotionLexiconFeatures,
            )
            from src.features.emotion.emotion_target_features import (
                EmotionTargetFeatures,
            )
            from src.features.emotion.emotion_trajectory_features import (
                EmotionTrajectoryFeatures,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Emotion extractors unavailable, using empty emotion output: %s",
                exc,
            )
            return

        self._extractors = [
            EmotionFeatures(),
            EmotionIntensityFeatures(),
            EmotionLexiconFeatures(),
            EmotionTargetFeatures(),
            EmotionTrajectoryFeatures(),
        ]

    def analyze(self, text: str) -> Dict[str, Dict[str, float]]:
        context = FeatureContext(text=text)
        output: Dict[str, Dict[str, float]] = {}

        for extractor in self._extractors:
            output[extractor.name] = extractor.extract(context)

        return output


@dataclass
class PipelineMetadata:
    """
    Metadata describing pipeline execution details.
    """

    processing_time: float
    article_length: int
    token_count: int
    model_version: Optional[str] = None


class TruthLensPipeline:
    """
    Main orchestration pipeline for TruthLens analysis.
    """

    def __init__(
        self,
        preprocessor: Optional[PreprocessingPipeline] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        emotion_pipeline: Optional[EmotionPipeline] = None,
        predictor: Optional[Predictor] = None,
        profile_builder: Optional[BiasProfileBuilder] = None,
        score_calculator: Optional[TruthLensScoreCalculator] = None,
        aggregation_pipeline: Optional[AggregationPipeline] = None,
    ) -> None:
        """
        Initialize pipeline components with dependency injection.
        """

        try:
            self.preprocessor = preprocessor or PreprocessingPipeline()
            self.feature_pipeline = feature_pipeline or FeaturePipeline()
            self.emotion_pipeline = emotion_pipeline or EmotionPipeline()
            self.predictor = predictor or Predictor()
            self.profile_builder = profile_builder or BiasProfileBuilder()
            self.score_calculator = score_calculator or TruthLensScoreCalculator()
            self.aggregation_pipeline = (
                aggregation_pipeline or AggregationPipeline()
            )

        except Exception as exc:
            logger.exception("TruthLensPipeline initialization failed")
            raise RuntimeError("Failed to initialize TruthLensPipeline") from exc

        logger.info("TruthLensPipeline initialized")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Run the complete TruthLens analysis pipeline.

        Parameters
        ----------
        text : str
            Raw article text.

        Returns
        -------
        Dict[str, Any]
            Complete analysis report.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        start_time = time.time()

        try:
            preprocessing_output = self.preprocessor.preprocess(text)

            normalized_text = preprocessing_output.normalized_text
            tokens = preprocessing_output.tokens

            if not normalized_text.strip():
                raise RuntimeError(
                    "Preprocessing output missing valid normalized_text"
                )

            feature_bundle = self.feature_pipeline.extract_features(normalized_text)

            emotion_outputs = self.emotion_pipeline.analyze(normalized_text)

            model_predictions = self._run_model_predictions(normalized_text)

        except Exception as exc:
            logger.exception("TruthLens pipeline execution failed")
            raise RuntimeError("TruthLens pipeline execution failed") from exc

        bias_features = feature_bundle.bias
        narrative_features = feature_bundle.narrative
        discourse_features = feature_bundle.discourse
        linguistic_features = feature_bundle.linguistic
        graph_features = feature_bundle.graph

        emotion_features: Dict[str, Any] = {}

        for section in emotion_outputs.values():
            if isinstance(section, dict):
                emotion_features.update(section)

        profile = self.profile_builder.build_profile(
            bias_features=bias_features,
            emotion_features=emotion_features,
            narrative_features=narrative_features,
            discourse_features=discourse_features,
            ideology_predictions=model_predictions,
        )

        profile["graph"] = graph_features
        profile["linguistic"] = linguistic_features

        aggregation_output: Dict[str, Any] = {}
        try:
            aggregation_output = self.aggregation_pipeline.run(
                profile,
                text=normalized_text,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Aggregation integration skipped: %s", exc)

        scores = aggregation_output.get("raw_scores")
        if not isinstance(scores, dict) or not scores:
            scores = self.score_calculator.compute_scores(profile)

        processing_time = time.time() - start_time

        metadata = PipelineMetadata(
            processing_time=processing_time,
            article_length=len(text),
            token_count=len(tokens),
            model_version=model_predictions.get("model_version"),
        )

        report: Dict[str, Any] = {
            "metadata": metadata.__dict__,
            "preprocessing": preprocessing_output.__dict__,
            "features": {
                "bias": bias_features,
                "narrative": narrative_features,
                "discourse": discourse_features,
                "linguistic": linguistic_features,
                "graph": graph_features,
            },
            "emotion_analysis": emotion_outputs,
            "model_predictions": model_predictions,
            "profile": profile,
            "scores": scores,
            "aggregation": aggregation_output,
        }

        return report

    def _run_model_predictions(self, text: str) -> Dict[str, Any]:
        """
        Run ML model predictions.

        Parameters
        ----------
        text : str

        Returns
        -------
        Dict[str, Any]
        """

        try:
            prediction_output = self.predictor.predict(text)

            if not isinstance(prediction_output, dict):
                raise RuntimeError("Predictor returned invalid output")

            return prediction_output

        except Exception as exc:
            logger.exception("Model prediction stage failed")
            raise RuntimeError("Prediction stage failed") from exc
