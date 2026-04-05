"""
File Name: feature_pipeline.py
Module: Feature Engineering - Feature Pipeline
Description:
    Implements the orchestrated feature extraction pipeline used across the
    TruthLens system. The pipeline coordinates feature discovery, execution,
    fusion, optional scaling, and optional feature selection.

    The pipeline integrates with:
        • BaseFeature abstractions
        • FeatureRegistry
        • FeatureFusion
        • FeatureScalingPipeline
        • FeatureSelectionPipeline

    Explicit integration of bias, framing, and ideological feature extractors:

        BiasFeatures (src.features.bias.bias_features)
            Output keys (prefix: bias_):
                bias_loaded_language_ratio, bias_subjective_ratio,
                bias_uncertainty_ratio, bias_polarization_ratio,
                bias_evaluative_ratio, bias_phrase_count,
                bias_exclamation_density, bias_caps_ratio,
                bias_intensity, bias_diversity

        FramingFeatures (src.features.bias.framing_features)
            Output keys (prefix: frame_):
                frame_economic_ratio, frame_moral_ratio,
                frame_security_ratio, frame_human_interest_ratio,
                frame_conflict_ratio, frame_phrase_count,
                frame_quote_density, frame_diversity,
                frame_dominance, frame_entropy

        IdeologicalFeatures (src.features.bias.ideological_features)
            Output keys (prefix: ideology_):
                ideology_left_ratio, ideology_right_ratio,
                ideology_balance, ideology_entropy,
                ideology_polarization_ratio, ideology_group_reference_ratio,
                ideology_phrase_count, ideology_signal_strength

    All three are registered via @register_feature and discovered automatically
    through bootstrap_feature_registry(). Explicit imports here guarantee
    registration even when bootstrap is not called, and expose their output
    key constants for downstream schema building and section routing.

    This module is responsible for producing deterministic, reproducible
    feature vectors from raw text inputs.

Dependencies:
    dataclasses
    typing
    logging

Inputs:
    FeatureContext

Outputs:
    Dict[str, float] feature vector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import FeatureRegistry
from src.features.feature_bootstrap import bootstrap_feature_registry
from src.features.fusion.feature_fusion import FeatureFusion
from src.features.fusion.feature_scaling import FeatureScalingPipeline
from src.features.fusion.feature_selection import FeatureSelectionPipeline
from src.graph.graph_pipeline import GraphPipeline

from src.features.bias.bias_features import BiasFeatures
from src.features.bias.framing_features import FramingFeatures
from src.features.bias.ideological_features import IdeologicalFeatures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output feature name constants
# ---------------------------------------------------------------------------

BIAS_FEATURE_NAMES: List[str] = [
    "bias_loaded_language_ratio",
    "bias_subjective_ratio",
    "bias_uncertainty_ratio",
    "bias_polarization_ratio",
    "bias_evaluative_ratio",
    "bias_phrase_count",
    "bias_exclamation_density",
    "bias_caps_ratio",
    "bias_intensity",
    "bias_diversity",
]

FRAMING_FEATURE_NAMES: List[str] = [
    "frame_economic_ratio",
    "frame_moral_ratio",
    "frame_security_ratio",
    "frame_human_interest_ratio",
    "frame_conflict_ratio",
    "frame_phrase_count",
    "frame_quote_density",
    "frame_diversity",
    "frame_dominance",
    "frame_entropy",
]

IDEOLOGICAL_FEATURE_NAMES: List[str] = [
    "ideology_left_ratio",
    "ideology_right_ratio",
    "ideology_balance",
    "ideology_entropy",
    "ideology_polarization_ratio",
    "ideology_group_reference_ratio",
    "ideology_phrase_count",
    "ideology_signal_strength",
]

ALL_BIAS_MODULE_FEATURE_NAMES: List[str] = sorted(
    BIAS_FEATURE_NAMES + FRAMING_FEATURE_NAMES + IDEOLOGICAL_FEATURE_NAMES
)


# ---------------------------------------------------------------------------
# Section partitioning helper
# ---------------------------------------------------------------------------

def partition_feature_sections(
    features: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Partition a flat feature dict from the pipeline into named sections.

    Routes features to one of five sections:
        "bias"      — keys starting with ``bias_``
        "framing"   — keys starting with ``frame_``
        "ideology"  — keys starting with ``ideology_``
        "emotion"   — keys starting with ``emotion_`` or ``lexicon_emotion_``
        "narrative" — keys starting with ``narrative_``
        "discourse" — keys starting with ``discourse_``
        "graph"     — keys starting with ``graph_``
        "other"     — everything else

    Parameters
    ----------
    features : Dict[str, float]
        Flat feature dict produced by FeaturePipeline.extract().

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict keyed by section name.
    """

    sections: Dict[str, Dict[str, float]] = {
        "bias": {},
        "framing": {},
        "ideology": {},
        "emotion": {},
        "narrative": {},
        "discourse": {},
        "graph": {},
        "other": {},
    }

    for key, value in features.items():
        if key.startswith("bias_"):
            sections["bias"][key] = value
        elif key.startswith("frame_"):
            sections["framing"][key] = value
        elif key.startswith("ideology_"):
            sections["ideology"][key] = value
        elif key.startswith("emotion_") or key.startswith("lexicon_emotion_"):
            sections["emotion"][key] = value
        elif key.startswith("narrative_"):
            sections["narrative"][key] = value
        elif key.startswith("discourse_"):
            sections["discourse"][key] = value
        elif key.startswith("graph_") or key.startswith("graph_pipeline_"):
            sections["graph"][key] = value
        else:
            sections["other"][key] = value

    return sections


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class FeaturePipeline:
    """
    Main feature extraction pipeline.

    Responsibilities:
        • initialize feature extractors via FeatureRegistry
        • execute feature extraction (including BiasFeatures,
          FramingFeatures, and IdeologicalFeatures)
        • fuse outputs
        • optionally scale features
        • optionally apply feature selection

    The three bias-module extractors (BiasFeatures, FramingFeatures,
    IdeologicalFeatures) contribute 28 features in total. Their output
    keys are available via the module-level constants:
        BIAS_FEATURE_NAMES, FRAMING_FEATURE_NAMES, IDEOLOGICAL_FEATURE_NAMES

    To partition extracted features by module section, use:
        partition_feature_sections(features)
    """

    feature_names: Optional[List[str]] = None
    scaler: Optional[FeatureScalingPipeline] = None
    selector: Optional[FeatureSelectionPipeline] = None

    features: List[BaseFeature] = field(default_factory=list)
    fusion: Optional[FeatureFusion] = None
    graph_pipeline: GraphPipeline | None = field(default=None, init=False, repr=False)

    def initialize(self) -> None:
        """
        Initialize feature extractors using FeatureRegistry.

        Calls bootstrap_feature_registry() which imports all registered
        feature modules, including:
            • BiasFeatures      (bias_*)
            • FramingFeatures   (frame_*)
            • IdeologicalFeatures (ideology_*)
        """
        bootstrap_feature_registry()

        if self.feature_names is None:
            self.feature_names = FeatureRegistry.list_features()

        self.features = [
            FeatureRegistry.create_feature(name) for name in self.feature_names
        ]

        self.fusion = FeatureFusion(self.features)
        try:
            self.graph_pipeline = GraphPipeline()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GraphPipeline unavailable in FeaturePipeline: %s", exc)
            self.graph_pipeline = None

        bias_active = any(
            isinstance(f, (BiasFeatures, FramingFeatures, IdeologicalFeatures))
            for f in self.features
        )
        logger.info(
            "FeaturePipeline initialized | feature_count=%d bias_modules_active=%s",
            len(self.features),
            bias_active,
        )

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract features from a single FeatureContext.

        Output includes contributions from BiasFeatures (bias_*),
        FramingFeatures (frame_*), and IdeologicalFeatures (ideology_*).
        Use partition_feature_sections() on the result to separate sections.
        """

        if self.fusion is None:
            raise RuntimeError("FeaturePipeline must be initialized before extraction")

        features = self.fusion.extract(context)

        if self.graph_pipeline is not None:
            try:
                graph_output = self.graph_pipeline.run(context.text)
                context.cache["graph_pipeline_output"] = graph_output

                graph_features = graph_output.get("graph_features", {})
                if isinstance(graph_features, dict):
                    for key, value in graph_features.items():
                        if isinstance(value, (int, float)):
                            features.setdefault(key, float(value))

                for section_name in ("entity_graph_metrics", "narrative_graph_metrics"):
                    section = graph_output.get(section_name, {})
                    if isinstance(section, dict):
                        for key, value in section.items():
                            if isinstance(value, (int, float)):
                                features.setdefault(f"graph_pipeline_{key}", float(value))
            except Exception as exc:  # noqa: BLE001
                logger.warning("GraphPipeline feature merge skipped: %s", exc)

        logger.debug("Feature extraction completed | feature_count=%d", len(features))

        return features

    def extract_with_sections(
        self, context: FeatureContext
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract features and return them partitioned by module section.

        Returns a dict of section -> feature dict via partition_feature_sections().
        Sections include: bias, framing, ideology, emotion, narrative,
        discourse, graph, other.
        """
        features = self.extract(context)
        return partition_feature_sections(features)

    def batch_extract(self, contexts: List[FeatureContext]) -> List[Dict[str, float]]:
        """
        Extract features for multiple contexts.
        """

        if not contexts:
            raise ValueError("Context list cannot be empty")

        results = []

        for ctx in contexts:
            results.append(self.extract(ctx))

        logger.info(
            "Batch feature extraction completed | samples=%d",
            len(results),
        )

        return results

    def fit_scaler(self, features: List[Dict[str, float]]) -> None:
        """
        Fit scaling pipeline.
        """

        if self.scaler is None:
            raise RuntimeError("No scaler configured")

        self.scaler.fit(features)

        logger.info("Feature scaler fitted")

    def transform_scaler(
        self, features: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Apply scaling transformation.
        """

        if self.scaler is None:
            return features

        return self.scaler.transform(features)

    def fit_selector(
        self,
        features: List[Dict[str, float]],
        labels: Optional[List[int]] = None,
    ) -> None:
        """
        Fit feature selector.
        """

        if self.selector is None:
            raise RuntimeError("No feature selector configured")

        self.selector.fit(features, labels)

        logger.info("Feature selector fitted")

    def transform_selector(
        self, features: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Apply feature selection.
        """

        if self.selector is None:
            return features

        return self.selector.transform(features)

    def process(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ) -> List[Dict[str, float]]:
        """
        Full pipeline execution.

        Steps:
            1. Feature extraction (bias_*, frame_*, ideology_*, ...)
            2. Optional scaling
            3. Optional feature selection
        """

        features = self.batch_extract(contexts)

        if self.scaler:

            if fit:
                self.fit_scaler(features)

            features = self.transform_scaler(features)

        if self.selector:

            if fit:
                self.fit_selector(features, labels)

            features = self.transform_selector(features)

        logger.info(
            "FeaturePipeline processing complete | samples=%d features=%d",
            len(features),
            len(features[0]) if features else 0,
        )

        return features


def apply_feature_engineering(
    df: pd.DataFrame,
    *,
    text_column: str = "text",
    tfidf_max_features: int = 5000,
    top_terms_per_doc: int = 5,
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Build engineered text from top TF-IDF terms per document.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    if text_column not in df.columns:
        raise ValueError(f"Missing text column: {text_column}")

    if top_terms_per_doc < 1:
        raise ValueError("top_terms_per_doc must be >= 1")

    texts = df[text_column].fillna("").astype(str).tolist()

    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=tfidf_max_features)
        matrix = vectorizer.fit_transform(texts)
    else:
        matrix = vectorizer.transform(texts)

    feature_names = vectorizer.get_feature_names_out()

    engineered: list[str] = []

    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        if row.nnz == 0:
            engineered.append("")
            continue

        order = row.data.argsort()[::-1][:top_terms_per_doc]
        top_indices = row.indices[order]
        terms = [str(feature_names[idx]) for idx in top_indices]
        engineered.append(" ".join(terms))

    output_df = df.copy()
    output_df["engineered_text"] = engineered

    return output_df, vectorizer


def transform_feature_pipeline(
    df: pd.DataFrame,
    *,
    vectorizer: TfidfVectorizer,
    text_column: str = "text",
    top_terms_per_doc: int = 5,
) -> pd.DataFrame:
    """
    Transform text data using an existing TF-IDF vectorizer.
    """

    transformed_df, _ = apply_feature_engineering(
        df,
        text_column=text_column,
        top_terms_per_doc=top_terms_per_doc,
        vectorizer=vectorizer,
    )
    return transformed_df
