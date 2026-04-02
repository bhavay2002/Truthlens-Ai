"""
File Name: analyze_article.py
Module: TruthLens Pipeline - Article Analysis
Description:
    Implements the main analysis pipeline for processing a single article in the
    TruthLens AI system. The module orchestrates multiple analytical components
    including bias detection, emotion analysis, narrative extraction, discourse
    analysis, graph construction, and final scoring. It produces a comprehensive
    analysis report describing linguistic signals and credibility indicators.

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

Inputs:
    Raw article text

Outputs:
    Structured analysis report containing extracted features and TruthLens scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import FeaturePipeline
from src.graph.entity_graph import EntityGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator

logger = logging.getLogger(__name__)


@dataclass
class ArticleAnalyzer:
    """
    Coordinates the full TruthLens analysis pipeline for an article.
    """

    feature_pipeline: FeaturePipeline
    entity_graph_builder: EntityGraphBuilder
    graph_analyzer: GraphAnalyzer
    profile_builder: BiasProfileBuilder
    score_calculator: TruthLensScoreCalculator

    def __post_init__(self) -> None:
        """
        Initialize the feature pipeline.
        """
        self.feature_pipeline.initialize()
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

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Run the full analysis pipeline on an article.
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

        except Exception as exc:  # noqa: BLE001
            logger.exception("Article analysis pipeline failed")
            raise RuntimeError("Article analysis failed") from exc

        graph_section = {**graph_features, **graph_metrics}

        profile = self.profile_builder.build_profile(
            bias_features=feature_sections["bias"],
            emotion_features=feature_sections["emotion"],
            narrative_features=feature_sections["narrative"],
            discourse_features=feature_sections["discourse"],
            ideology_predictions={},
        )

        profile["graph"] = graph_section

        scores = self.score_calculator.compute_scores(profile)

        report: Dict[str, Any] = {
            "bias_features": feature_sections["bias"],
            "emotion_features": feature_sections["emotion"],
            "narrative_features": feature_sections["narrative"],
            "discourse_features": feature_sections["discourse"],
            "graph_features": graph_section,
            "profile": profile,
            "scores": scores,
        }

        logger.info("Article analysis completed")

        return report