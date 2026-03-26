 """
File Name: truthlens_pipeline.py
Module: TruthLens Pipeline - Unified Analysis Pipeline
Description:
    Implements the top-level TruthLens analysis pipeline used to process and
    analyze textual content. The pipeline coordinates preprocessing, feature
    extraction, emotion analysis, graph construction, bias profile generation,
    and final TruthLens scoring. It serves as the central orchestration layer
    for the entire TruthLens system and produces a complete analysis report
    for a given input article.

Dependencies:
    logging
    typing
    preprocessing_pipeline
    feature_pipeline
    emotion_pipeline
    bias_profile_builder
    truthlens_score_calculator

Inputs:
    Raw text string

Outputs:
    Complete TruthLens analysis report containing extracted features,
    intermediate analysis outputs, and final scoring metrics
"""

import logging
from typing import Dict, Any

from preprocessing_pipeline import PreprocessingPipeline
from feature_pipeline import FeaturePipeline
from emotion_pipeline import EmotionPipeline
from bias_profile_builder import BiasProfileBuilder
from truthlens_score_calculator import TruthLensScoreCalculator


logger = logging.getLogger(__name__)


class TruthLensPipeline:
    """
    Main orchestration pipeline for TruthLens analysis.
    """

    def __init__(self) -> None:
        """Initialize pipeline components."""

        self.preprocessor = PreprocessingPipeline()
        self.feature_pipeline = FeaturePipeline()
        self.emotion_pipeline = EmotionPipeline()
        self.profile_builder = BiasProfileBuilder()
        self.score_calculator = TruthLensScoreCalculator()

        logger.info("TruthLensPipeline initialized")

    def analyze(self, text: str) -> Dict[str, Any]:
        """Run the complete TruthLens analysis pipeline."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            preprocessing_output = self.preprocessor.preprocess(text)

            normalized_text = preprocessing_output["normalized_text"]

            feature_outputs = self.feature_pipeline.extract_features(normalized_text)

            emotion_outputs = self.emotion_pipeline.analyze(normalized_text)

        except Exception as exc:
            logger.exception("TruthLens pipeline preprocessing or feature stage failed")
            raise RuntimeError("TruthLens pipeline failed during feature extraction") from exc

        bias_features = feature_outputs.get("bias", {})
        narrative_features = feature_outputs.get("narrative", {})
        discourse_features = feature_outputs.get("discourse", {})
        graph_features = feature_outputs.get("graph", {})

        emotion_features = {}

        for section in emotion_outputs.values():
            if isinstance(section, dict):
                emotion_features.update(section)

        profile = self.profile_builder.build_profile(
            bias_features=bias_features,
            emotion_features=emotion_features,
            narrative_features=narrative_features,
            discourse_features=discourse_features,
            ideology_predictions={}
        )

        profile["graph"] = graph_features

        scores = self.score_calculator.compute_scores(profile)

        report = {
            "preprocessing": preprocessing_output,
            "features": feature_outputs,
            "emotion_analysis": emotion_outputs,
            "profile": profile,
            "scores": scores
        }

        return report