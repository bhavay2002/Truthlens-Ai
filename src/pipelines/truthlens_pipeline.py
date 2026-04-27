from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from src.analysis.preprocessing import PreprocessingPipeline

from src.analysis.analysis_pipeline import AnalysisPipeline
from src.analysis.analysis_registry import build_default_registry
from src.analysis.orchestrator import AnalysisOrchestrator

from src.inference.inference_pipeline import Predictor
from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator
from src.graph.graph_pipeline import GraphPipeline, get_default_pipeline
from src.explainability.explainability_pipeline import run_explainability_pipeline
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
        predictor: Optional[Predictor] = None,
        aggregation_pipeline: Optional[AggregationPipeline] = None,
        score_calculator: Optional[TruthLensScoreCalculator] = None,
        graph_pipeline: Optional[GraphPipeline] = None,
        enable_explainability: bool = False,
        enable_evaluation: bool = False,
    ):

        self.preprocessor = preprocessor or PreprocessingPipeline()
        self.predictor = predictor or Predictor()
        self.aggregation_pipeline = aggregation_pipeline or AggregationPipeline()
        self.score_calculator = score_calculator or TruthLensScoreCalculator()
        # G-R1: fall back to the process-wide singleton instead of
        # building a fresh ``GraphPipeline`` (which spins up 6 builders
        # + 15 analyzers) per ``TruthLensPipeline`` instance.
        self.graph_pipeline = graph_pipeline or get_default_pipeline()

        self.enable_explainability = enable_explainability
        self.enable_evaluation = enable_evaluation

        registry = build_default_registry()
        analysis_pipeline = AnalysisPipeline(registry=registry)

        self.analysis_orchestrator = AnalysisOrchestrator(
            pipeline=analysis_pipeline
        )

        logger.info("TruthLensPipeline initialized (WITH ANALYSIS LAYER)")

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
        # 2. ANALYSIS LAYER (CORE)
        # -------------------------------------------------
        t0 = time.time()

        analysis_output = self.analysis_orchestrator.run(
            prep.normalized_text
        )

        features = analysis_output.get("features", {})
        profile = analysis_output.get("profile", {})
        propaganda = analysis_output.get("propaganda", {})

        stage_time["analysis"] = time.time() - t0

        # -------------------------------------------------
        # 3. GRAPH PIPELINE
        # -------------------------------------------------
        t0 = time.time()
        graph_output: Dict[str, Any] = {}

        try:
            graph_output = self.graph_pipeline.run(prep.normalized_text)

            # merge graph features
            if isinstance(graph_output, dict) and "graph_features" in graph_output:
                features.update(graph_output["graph_features"])

                # also enrich profile if exists
                if isinstance(profile, dict):
                    profile["graph"] = graph_output.get("graph_features", {})
                    profile["graph_explanation"] = graph_output.get("graph_explanation")

        except Exception:
            logger.warning("Graph pipeline failed", exc_info=True)

        stage_time["graph"] = time.time() - t0

        # -------------------------------------------------
        # 4. PREDICTION
        # -------------------------------------------------
        t0 = time.time()

        # ⚠️ NOTE: predictor still uses text (can upgrade later)
        predictions = self.predictor.predict(prep.normalized_text)

        stage_time["prediction"] = time.time() - t0

        # -------------------------------------------------
        # 5. AGGREGATION
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
        # 6. EXPLAINABILITY
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
        # 7. EVALUATION
        # -------------------------------------------------
        evaluation = None

        if self.enable_evaluation:
            try:
                evaluation = run_evaluation_pipeline(
                    model=getattr(self.predictor, "model", None),
                    tokenizer=getattr(self.predictor, "tokenizer", None),
                    texts=[prep.normalized_text],
                    labels=None,
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

            "analysis": analysis_output,

            "features": features,
            "profile": profile,
            "propaganda": propaganda,

            "graph": graph_output,
            "predictions": predictions,
            "scores": scores,
            "aggregation": aggregation,
            "explainability": explanation,
            "evaluation": evaluation,
        }