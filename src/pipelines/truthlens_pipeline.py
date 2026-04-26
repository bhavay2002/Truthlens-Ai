"""
src/pipelines/truthlens_pipeline.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from src.analysis.preprocessing import PreprocessingPipeline
from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import FeaturePipeline
from src.inference.inference_pipeline import Predictor
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator
from src.explainability.explainability_pipeline import run_explainability_pipeline
from src.graph.graph_pipeline import GraphPipeline
from src.evaluation.evaluation_pipeline import run_evaluation_pipeline

logger = logging.getLogger(__name__)


# =========================================================
# METADATA
# =========================================================

@dataclass
class PipelineMetadata:
    total_time: float
    text_length: int
    token_count: int
    model_version: Optional[str]
    stages: Dict[str, float]


# =========================================================
# PIPELINE
# =========================================================

class TruthLensPipeline:

    def __init__(
        self,
        *,
        preprocessor: Optional[PreprocessingPipeline] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
        predictor: Optional[Predictor] = None,
        profile_builder: Optional[BiasProfileBuilder] = None,
        aggregation_pipeline: Optional[AggregationPipeline] = None,
        score_calculator: Optional[TruthLensScoreCalculator] = None,
        graph_pipeline: Optional[GraphPipeline] = None,
        enable_explainability: bool = False,
        enable_evaluation: bool = False,
    ):

        self.preprocessor = preprocessor or PreprocessingPipeline()
        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.predictor = predictor or Predictor()
        self.profile_builder = profile_builder or BiasProfileBuilder()
        self.aggregation_pipeline = aggregation_pipeline or AggregationPipeline()
        self.score_calculator = score_calculator or TruthLensScoreCalculator()
        self.graph_pipeline = graph_pipeline or GraphPipeline()

        self.enable_explainability = enable_explainability
        self.enable_evaluation = enable_evaluation

        logger.info("TruthLensPipeline initialized (FINAL CLEAN)")

    # =====================================================
    # MAIN
    # =====================================================

    def analyze(self, text: str) -> Dict[str, Any]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")

        start = time.time()
        stage_time: Dict[str, float] = {}

        # -------------------------------------------------
        # 1. PREPROCESSING
        # -------------------------------------------------
        t0 = time.time()
        prep = self.preprocessor.preprocess(text)
        stage_time["preprocessing"] = time.time() - t0

        # -------------------------------------------------
        # 2. FEATURE CONTEXT
        # -------------------------------------------------
        ctx = FeatureContext(text=prep.normalized_text)

        # -------------------------------------------------
        # 3. FEATURE EXTRACTION
        # -------------------------------------------------
        t0 = time.time()
        features = self.feature_pipeline.extract(ctx)
        stage_time["features"] = time.time() - t0

        # -------------------------------------------------
        # 4. GRAPH PIPELINE (REAL USE)
        # -------------------------------------------------
        t0 = time.time()
        graph_output: Dict[str, Any] = {}

        try:
            graph_output = self.graph_pipeline.run(prep.normalized_text)

            # 🔥 Merge graph into features (critical)
            if isinstance(graph_output, dict) and "graph_features" in graph_output:
                features.update(graph_output["graph_features"])

        except Exception:
            logger.warning("Graph pipeline failed", exc_info=True)

        stage_time["graph"] = time.time() - t0

        # -------------------------------------------------
        # 5. PREDICTION
        # -------------------------------------------------
        t0 = time.time()
        predictions = self.predictor.predict(prep.normalized_text)
        stage_time["prediction"] = time.time() - t0

        # -------------------------------------------------
        # 6. PROFILE BUILDING
        # -------------------------------------------------
        t0 = time.time()

        profile = self.profile_builder.build_profile(
            bias_features=self._section(features, "bias"),
            emotion_features=self._section(features, "emotion"),
            narrative_features=self._section(features, "narrative"),
            discourse_features=self._section(features, "discourse"),
            ideology_predictions=predictions,
        )

        # 🔥 Inject graph into profile (important)
        if isinstance(graph_output, dict):
            profile["graph"] = graph_output.get("graph_features", {})
            profile["graph_explanation"] = graph_output.get("graph_explanation")

        stage_time["analysis"] = time.time() - t0

        # -------------------------------------------------
        # 7. AGGREGATION
        # -------------------------------------------------
        t0 = time.time()

        try:
            aggregation = self.aggregation_pipeline.run(
                profile,
                text=prep.normalized_text,
            )

            scores = (
                aggregation.get("scores")
                or aggregation.get("raw_scores")
                or {}
            )

        except Exception:
            logger.exception("Aggregation failed")
            aggregation = {}
            scores = self.score_calculator.compute_scores(profile)

        stage_time["aggregation"] = time.time() - t0

        # -------------------------------------------------
        # 8. EXPLAINABILITY
        # -------------------------------------------------
        explanation = None

        if self.enable_explainability:
            try:
                explanation = run_explainability_pipeline(
                    text=prep.normalized_text,
                    predict_fn=self.predictor.predict,
                ).model_dump()
            except Exception:
                logger.warning("Explainability failed", exc_info=True)

        stage_time["explainability"] = stage_time.get("explainability", 0.0)

        # -------------------------------------------------
        # 9. EVALUATION (OPTIONAL)
        # -------------------------------------------------
        evaluation = None

        if self.enable_evaluation:
            try:
                evaluation = run_evaluation_pipeline(
                    model=getattr(self.predictor, "model", None),
                    tokenizer=getattr(self.predictor, "tokenizer", None),
                    texts=[prep.normalized_text],
                    labels=None,  # supply real labels when available
                )
            except Exception:
                logger.warning("Evaluation skipped", exc_info=True)

        stage_time["evaluation"] = stage_time.get("evaluation", 0.0)

        # -------------------------------------------------
        # METADATA
        # -------------------------------------------------
        metadata = PipelineMetadata(
            total_time=time.time() - start,
            text_length=len(text),
            token_count=len(prep.tokens),
            model_version=predictions.get("model_version"),
            stages=stage_time,
        )

        # -------------------------------------------------
        # FINAL OUTPUT
        # -------------------------------------------------
        return {
            "metadata": asdict(metadata),
            "preprocessing": prep.__dict__,
            "features": features,
            "graph": graph_output,
            "predictions": predictions,
            "profile": profile,
            "scores": scores,
            "aggregation": aggregation,
            "explainability": explanation,
            "evaluation": evaluation,
        }

    # =====================================================
    # HELPERS
    # =====================================================

    def _section(self, features: Dict[str, float], prefix: str) -> Dict[str, float]:
        return {
            k: v
            for k, v in features.items()
            if k.startswith(prefix)
        }