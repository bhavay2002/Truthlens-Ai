"""
File Name: analyze_article.py
Module: TruthLens Pipeline - Article Analysis
Description:
    Implements the main analysis pipeline for processing a single article in the
    TruthLens AI system. The module orchestrates multiple analytical components
    including bias detection, emotion analysis, narrative extraction, discourse
    analysis, graph construction, and final scoring. It produces a comprehensive
    analysis report describing linguistic signals and credibility indicators.

    Integrates the explainability subsystem via an optional ExplainabilityLayer
    attached to the inference prediction pipeline, enriching the report with
    token-level attribution, attention rollout, aggregated explanations, and
    cross-method consistency metrics.

Author: TruthLens Engineering Team
Date: 2026-04-02
Dependencies:
    logging
    typing
    dataclasses
    src.features.base.base_feature
    src.features.feature_pipeline
    src.graph.entity_graph
    src.graph.graph_analysis
    src.analysis.bias_profile_builder
    src.aggregation.truthlens_score_calculator
    src.inference.prediction_pipeline (ExplainabilityLayer)
    src.explainability.explanation_report_generator

Inputs:
    Raw article text

Outputs:
    Structured analysis report containing extracted features, TruthLens scores,
    and explainability output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Callable, List, Optional

import numpy as np
import torch

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import FeaturePipeline
from src.graph.entity_graph import EntityGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer
from src.graph.narrative_graph_builder import NarrativeGraphBuilder
from src.graph.graph_pipeline import GraphPipeline
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.analysis.integration_runner import AnalysisIntegrationRunner
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator
from src.inference.feature_preparer import FeaturePreparer
from src.inference.prediction_pipeline import (
    PredictionPipeline as InferencePredictionPipeline,
    ExplainabilityLayer,
)
from src.inference.report_generator import ReportGenerator
from src.explainability.explanation_report_generator import ExplanationReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class ArticleAnalyzer:
    """
    Coordinates the full TruthLens analysis pipeline for an article.

    When an ExplainabilityLayer is provided on the inference_prediction_pipeline,
    the analyze() method automatically enriches predictions with token-level
    attribution from LIME, attention rollout, propaganda gradient scores, and
    cross-method consistency metrics. These appear under the
    'explainability' key in the returned report.
    """

    feature_pipeline: FeaturePipeline
    entity_graph_builder: EntityGraphBuilder
    graph_analyzer: GraphAnalyzer
    profile_builder: BiasProfileBuilder
    score_calculator: TruthLensScoreCalculator
    narrative_graph_builder: Optional[NarrativeGraphBuilder] = None
    graph_pipeline: Optional[GraphPipeline] = None
    analysis_runner: AnalysisIntegrationRunner | None = None
    inference_prediction_pipeline: InferencePredictionPipeline | None = None
    inference_feature_preparer: FeaturePreparer | None = None
    report_generator: ReportGenerator | None = None
    explanation_report_generator: ExplanationReportGenerator | None = None
    predict_fn: Optional[Callable[[str], Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        """
        Initialize the feature pipeline and any lazy defaults.
        """
        self.feature_pipeline.initialize()
        if self.narrative_graph_builder is None:
            self.narrative_graph_builder = NarrativeGraphBuilder()
        if self.graph_pipeline is None:
            self.graph_pipeline = GraphPipeline()
        if self.analysis_runner is None:
            self.analysis_runner = AnalysisIntegrationRunner()
        if self.report_generator is None:
            self.report_generator = ReportGenerator()
        if self.explanation_report_generator is None:
            self.explanation_report_generator = ExplanationReportGenerator()
        logger.info("ArticleAnalyzer initialized")

    def _extract_feature_sections(
        self, features: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Organize features by prefix namespace.
        """

        sections: Dict[str, Dict[str, float]] = {
            "bias": {},
            "emotion": {},
            "narrative": {},
            "discourse": {},
        }

        for key, value in features.items():

            if key.startswith("bias_"):
                sections["bias"][key] = value

            elif key.startswith("emotion_") or key.startswith("lexicon_emotion_"):
                sections["emotion"][key] = value

            elif key.startswith("narrative_"):
                sections["narrative"][key] = value

            elif key.startswith("discourse_"):
                sections["discourse"][key] = value

        return sections

    def _run_prediction(
        self,
        text: str,
        fused_features: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Run the inference prediction pipeline on prepared features.

        When the pipeline has an attached ExplainabilityLayer, this calls
        predict_with_explanation() so that token-level attributions,
        attention rollout, and consistency metrics are included in the
        returned dict under the 'explainability' key.
        """

        if (
            self.inference_prediction_pipeline is None
            or self.inference_feature_preparer is None
        ):
            return {}

        try:
            prepared = self.inference_feature_preparer.prepare_single(
                {"text": text, **fused_features}
            )
            if isinstance(prepared, np.ndarray):
                prepared_tensor = torch.tensor(prepared, dtype=torch.float32)
            else:
                prepared_tensor = prepared

            has_explainability = (
                self.inference_prediction_pipeline.explainability_layer is not None
                and self.predict_fn is not None
            )

            if has_explainability:
                return self.inference_prediction_pipeline.predict_with_explanation(
                    features=prepared_tensor,
                    text=text,
                    predict_fn=self.predict_fn,
                )

            return self.inference_prediction_pipeline.predict(prepared_tensor)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Inference prediction integration skipped: %s", exc)
            return {}

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Run the full analysis pipeline on an article.

        Returns a report dict containing feature sections, graph data,
        analysis module outputs, credibility scores, ML predictions, and
        — when an ExplainabilityLayer is configured — a full explainability
        package with token attributions, aggregated importance scores, and
        cross-method consistency metrics.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        context = FeatureContext(text=text)

        try:
            fused_features = self.feature_pipeline.extract(context)
            feature_sections = self._extract_feature_sections(fused_features)

            entity_graph = self.entity_graph_builder.build_graph(text)
            graph_features = self.entity_graph_builder.extract_graph_features(
                entity_graph
            )
            graph_metrics = self.graph_analyzer.analyze(entity_graph)

            narrative_graph = (
                self.narrative_graph_builder.build_graph(text)
                if self.narrative_graph_builder is not None
                else {}
            )
            narrative_graph_features = (
                self.narrative_graph_builder.extract_graph_features(narrative_graph).to_dict()
                if self.narrative_graph_builder is not None
                else {}
            )
            narrative_graph_metrics = (
                self.graph_analyzer.analyze(narrative_graph).to_dict()
                if narrative_graph
                else {}
            )
            graph_pipeline_output = (
                self.graph_pipeline.run(text)
                if self.graph_pipeline is not None
                else {}
            )
            analysis_modules = (
                self.analysis_runner.analyze_text(text)
                if self.analysis_runner is not None
                else {}
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Article analysis pipeline failed")
            raise RuntimeError("Article analysis failed") from exc

        graph_section = {
            **graph_features.to_dict(),
            **graph_metrics.to_dict(),
            **narrative_graph_features,
            **narrative_graph_metrics,
            **(
                graph_pipeline_output.get("graph_features", {})
                if isinstance(graph_pipeline_output, dict)
                else {}
            ),
            **(
                graph_pipeline_output.get("entity_graph_metrics", {})
                if isinstance(graph_pipeline_output, dict)
                else {}
            ),
            **(
                graph_pipeline_output.get("narrative_graph_metrics", {})
                if isinstance(graph_pipeline_output, dict)
                else {}
            ),
        }

        profile = self.profile_builder.build_profile(
            bias_features=feature_sections["bias"],
            emotion_features=feature_sections["emotion"],
            narrative_features=feature_sections["narrative"],
            discourse_features=feature_sections["discourse"],
            ideology_predictions={},
        )

        profile["graph"] = graph_section
        scores = self.score_calculator.compute_scores(profile)

        prediction_output = self._run_prediction(text, fused_features)

        inference_report: Dict[str, Any] = {}
        if self.report_generator is not None:
            try:
                inference_report = self.report_generator.generate_report(
                    article_text=text,
                    bias_analysis={
                        "features": feature_sections["bias"],
                        "prediction": prediction_output.get("bias"),
                    },
                    emotion_analysis={
                        "features": feature_sections["emotion"],
                        "prediction": prediction_output.get("emotion"),
                    },
                    narrative_structure={
                        "features": feature_sections["narrative"],
                        "analysis_modules": analysis_modules.get("narrative_conflict", {}),
                    },
                    entity_graph={
                        "entity_graph": entity_graph,
                        "narrative_graph": narrative_graph,
                        "graph_metrics": graph_section,
                    },
                    credibility_score=prediction_output.get(
                        "credibility_score",
                        scores.get("truthlens_credibility_score"),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Report generation integration skipped: %s", exc)

        explainability_output = prediction_output.pop("explainability", {})

        if explainability_output and self.explanation_report_generator is not None:
            try:
                import hashlib
                article_id = hashlib.sha256(text.encode()).hexdigest()[:16]
                self.explanation_report_generator.generate(
                    article_id=article_id,
                    explanation=explainability_output,
                    save_json=True,
                    save_html=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Explanation report artifact generation skipped: %s", exc)

        report: Dict[str, Any] = {
            "bias_features": feature_sections["bias"],
            "emotion_features": feature_sections["emotion"],
            "narrative_features": feature_sections["narrative"],
            "discourse_features": feature_sections["discourse"],
            "graph_features": graph_section,
            "entity_graph": entity_graph,
            "narrative_graph": narrative_graph,
            "graph_pipeline": graph_pipeline_output,
            "analysis_modules": analysis_modules,
            "profile": profile,
            "scores": scores,
            "predictions": prediction_output,
            "inference_report": inference_report,
            "explainability": explainability_output,
        }

        logger.info("Article analysis completed")
        return report
