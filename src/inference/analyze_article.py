"""
File Name: analyze_article.py
Module: TruthLens Pipeline - Article Analysis
Description:
    Implements the main analysis pipeline for processing a single article in the
    TruthLens AI system. The module orchestrates multiple analytical components
    including bias detection, emotion analysis, narrative extraction, discourse
    analysis, graph construction, and final scoring. It produces a comprehensive
    analysis report describing linguistic signals and credibility indicators.

Dependencies:
    logging
    typing
    bias_features
    emotion_feature_extractor
    narrative_features
    discourse_features
    entity_graph
    graph_analysis
    bias_profile_builder
    truthlens_score_calculator

Inputs:
    Raw article text

Outputs:
    Structured analysis report containing extracted features and TruthLens scores
"""

import logging
from typing import Dict, Any

from src.features.bias.bias_features import BiasFeatureExtractor
from src.features.emotion.emotion_feature_extractor import EmotionFeatureExtractor
from src.features.narrative.narrative_features import NarrativeFeatureExtractor
from src.features.discourse.discourse_features import DiscourseFeatureExtractor
from src.graph.entity_graph import EntityGraphBuilder
from src.graph.graph_analysis import GraphAnalyzer
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.aggregation.truthlens_score_calculator import TruthLensScoreCalculator


logger = logging.getLogger(__name__)


class ArticleAnalyzer:
    """
    Coordinates the full TruthLens analysis pipeline for an article.
    """

    def __init__(self) -> None:
        """Initialize all analysis components."""

        self.bias_extractor = BiasFeatureExtractor()
        self.emotion_extractor = EmotionFeatureExtractor()
        self.narrative_extractor = NarrativeFeatureExtractor()
        self.discourse_extractor = DiscourseFeatureExtractor()
        self.entity_graph_builder = EntityGraphBuilder()
        self.graph_analyzer = GraphAnalyzer()
        self.profile_builder = BiasProfileBuilder()
        self.score_calculator = TruthLensScoreCalculator()

        logger.info("ArticleAnalyzer initialized")

    def analyze(self, text: str) -> Dict[str, Any]:
        """Run the full analysis pipeline on an article."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:

            bias_features = self.bias_extractor.extract_features(text)

            emotion_features = self.emotion_extractor.extract_features(text)

            narrative_features = self.narrative_extractor.extract(text)

            discourse_features = self.discourse_extractor.extract(text)

            entity_graph = self.entity_graph_builder.build_graph(text)

            graph_features = self.entity_graph_builder.extract_graph_features(
                entity_graph
            )

            graph_metrics = self.graph_analyzer.analyze(entity_graph)

        except Exception as exc:
            logger.exception("Feature extraction pipeline failed")
            raise RuntimeError("Article analysis failed") from exc

        graph_section = {**graph_features, **graph_metrics}

        profile = self.profile_builder.build_profile(
            bias_features=bias_features,
            emotion_features=emotion_features,
            narrative_features=narrative_features,
            discourse_features=discourse_features,
            ideology_predictions={},
        )

        profile["graph"] = graph_section

        scores = self.score_calculator.compute_scores(profile)

        report = {
            "bias_features": bias_features,
            "emotion_features": emotion_features,
            "narrative_features": narrative_features,
            "discourse_features": discourse_features,
            "graph_features": graph_section,
            "profile": profile,
            "scores": scores,
        }

        return report
