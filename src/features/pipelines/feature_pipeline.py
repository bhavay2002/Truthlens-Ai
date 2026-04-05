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

    Explicit integration of all feature extractor modules:

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

        ArgumentStructureFeatures (src.features.discourse.argument_structure_features)
            Output keys (prefix: argument_):
                argument_claim_ratio, argument_premise_ratio,
                argument_evidence_ratio, argument_counterargument_ratio,
                argument_rhetorical_question_ratio,
                argument_structure_density, argument_structure_diversity

        DiscourseFeatures (src.features.discourse.discourse_features)
            Output keys (prefix: discourse_):
                discourse_causal_ratio, discourse_contrast_ratio,
                discourse_additive_ratio, discourse_sequential_ratio,
                discourse_evidential_ratio, discourse_marker_density,
                discourse_diversity

        EntityGraphFeatures (src.features.graph.entity_graph_features)
            Output keys (prefix: entity_):
                entity_count, entity_edge_count, entity_avg_degree,
                entity_density, entity_centralization

        InteractionGraphFeatures (src.features.graph.interaction_graph_features)
            Output keys (prefix: interaction_):
                interaction_node_count, interaction_edge_count,
                interaction_avg_degree, interaction_density,
                interaction_clustering, interaction_component_count

        ConflictFeatures (src.features.narrative.conflict_features)
            Output keys (prefix: conflict_):
                conflict_confrontation_ratio, conflict_dispute_ratio,
                conflict_accusation_ratio, conflict_aggression_ratio,
                conflict_polarization_ratio, conflict_escalation_ratio,
                conflict_intensity, conflict_diversity, conflict_rhetoric_score

        NarrativeFeatures (src.features.narrative.narrative_features)
            Output keys (prefix: narrative_):
                narrative_hero_ratio, narrative_villain_ratio,
                narrative_victim_ratio, narrative_conflict_ratio,
                narrative_resolution_ratio, narrative_crisis_ratio,
                narrative_polarization_ratio, narrative_role_diversity,
                narrative_conflict_intensity, narrative_progression_score,
                narrative_rhetoric_score

        NarrativeFrameFeatures (src.features.narrative.narrative_frame_features)
            Output keys (prefix: narrative_frame_):
                narrative_frame_conflict_ratio, narrative_frame_economic_ratio,
                narrative_frame_human_interest_ratio, narrative_frame_moral_ratio,
                narrative_frame_responsibility_ratio, narrative_frame_diversity,
                narrative_frame_dominance, narrative_frame_balance,
                narrative_frame_rhetoric_score

        NarrativeRoleFeatures (src.features.narrative.narrative_role_features)
            Output keys (prefix: narrative_role_):
                narrative_role_hero_ratio, narrative_role_villain_ratio,
                narrative_role_victim_ratio, narrative_role_polarization_ratio,
                narrative_role_balance, narrative_role_diversity,
                narrative_entity_density

        ManipulationPatterns (src.features.propaganda.manipulation_patterns)
            Output keys (prefix: manipulation_):
                manipulation_urgency_ratio, manipulation_fear_ratio,
                manipulation_blame_ratio, manipulation_scapegoat_ratio,
                manipulation_absolute_ratio, manipulation_conspiracy_ratio,
                manipulation_false_dilemma_ratio, manipulation_exaggeration_ratio,
                manipulation_intensifier_ratio, manipulation_exclamation_density,
                manipulation_caps_emphasis, manipulation_intensity,
                manipulation_diversity

        PropagandaFeatures (src.features.propaganda.propaganda_features)
            Output keys (prefix: propaganda_):
                propaganda_name_calling_ratio, propaganda_fear_ratio,
                propaganda_exaggeration_ratio, propaganda_glitter_ratio,
                propaganda_us_vs_them_ratio, propaganda_authority_ratio,
                propaganda_intensifier_ratio, propaganda_exclamation_density,
                propaganda_caps_ratio, propaganda_intensity, propaganda_diversity

        PropagandaLexiconFeatures (src.features.propaganda.propaganda_lexicon_features)
            Output keys (prefix: propaganda_):
                propaganda_name_calling_ratio, propaganda_fear_ratio,
                propaganda_exaggeration_ratio, propaganda_bandwagon_ratio,
                propaganda_slogan_ratio, propaganda_phrase_bandwagon,
                propaganda_phrase_slogan, propaganda_exclamation_density,
                propaganda_caps_ratio, propaganda_lexicon_density,
                propaganda_lexicon_diversity

        LexicalFeatures (src.features.text.lexical_features)
            Output keys: vocabulary_size, hapax_legomena_ratio,
                hapax_dislegomena_ratio, lexical_density, average_word_length

        SemanticFeatures (src.features.text.semantic_features)
            Output keys (prefix: embedding_):
                embedding_norm, embedding_mean, embedding_std,
                embedding_max, embedding_min

        SyntacticFeatures (src.features.text.syntactic_features)
            Output keys: sentence_count, avg_sentence_length,
                noun_ratio, verb_ratio, adjective_ratio,
                adverb_ratio, punctuation_ratio

        TokenFeatures (src.features.text.token_features)
            Output keys (prefix: token_):
                token_count, unique_token_count, type_token_ratio,
                avg_token_length, max_token_length, repetition_ratio

    All extractors are registered via @register_feature and discovered
    automatically through bootstrap_feature_registry(). Explicit imports
    here guarantee registration even when bootstrap is not called, and
    expose their output key constants for downstream schema building and
    section routing.

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
from src.features.feature_schema_validator import FeatureSchemaValidator
from src.features.feature_statistics import FeatureStatistics
from src.features.fusion.feature_fusion import FeatureFusion
from src.features.fusion.feature_scaling import FeatureScalingPipeline
from src.features.fusion.feature_selection import FeatureSelectionPipeline
from src.graph.graph_pipeline import GraphPipeline

from src.features.bias.bias_features import BiasFeatures
from src.features.bias.framing_features import FramingFeatures
from src.features.bias.ideological_features import IdeologicalFeatures

from src.features.discourse.argument_structure_features import ArgumentStructureFeatures
from src.features.discourse.discourse_features import DiscourseFeatures

from src.features.graph.entity_graph_features import EntityGraphFeatures
from src.features.graph.interaction_graph_features import InteractionGraphFeatures

from src.features.narrative.conflict_features import ConflictFeatures
from src.features.narrative.narrative_features import NarrativeFeatures
from src.features.narrative.narrative_frame_features import NarrativeFrameFeatures
from src.features.narrative.narrative_role_features import NarrativeRoleFeatures

from src.features.propaganda.manipulation_patterns import ManipulationPatterns
from src.features.propaganda.propaganda_features import PropagandaFeatures
from src.features.propaganda.propaganda_lexicon_features import PropagandaLexiconFeatures

from src.features.text.lexical_features import LexicalFeatures
from src.features.text.semantic_features import SemanticFeatures
from src.features.text.syntactic_features import SyntacticFeatures
from src.features.text.token_features import TokenFeatures

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

ARGUMENT_STRUCTURE_FEATURE_NAMES: List[str] = [
    "argument_claim_ratio",
    "argument_premise_ratio",
    "argument_evidence_ratio",
    "argument_counterargument_ratio",
    "argument_rhetorical_question_ratio",
    "argument_structure_density",
    "argument_structure_diversity",
]

DISCOURSE_FEATURE_NAMES: List[str] = [
    "discourse_causal_ratio",
    "discourse_contrast_ratio",
    "discourse_additive_ratio",
    "discourse_sequential_ratio",
    "discourse_evidential_ratio",
    "discourse_marker_density",
    "discourse_diversity",
]

ENTITY_GRAPH_FEATURE_NAMES: List[str] = [
    "entity_count",
    "entity_edge_count",
    "entity_avg_degree",
    "entity_density",
    "entity_centralization",
]

INTERACTION_GRAPH_FEATURE_NAMES: List[str] = [
    "interaction_node_count",
    "interaction_edge_count",
    "interaction_avg_degree",
    "interaction_density",
    "interaction_clustering",
    "interaction_component_count",
]

CONFLICT_FEATURE_NAMES: List[str] = [
    "conflict_confrontation_ratio",
    "conflict_dispute_ratio",
    "conflict_accusation_ratio",
    "conflict_aggression_ratio",
    "conflict_polarization_ratio",
    "conflict_escalation_ratio",
    "conflict_intensity",
    "conflict_diversity",
    "conflict_rhetoric_score",
]

NARRATIVE_FEATURE_NAMES: List[str] = [
    "narrative_hero_ratio",
    "narrative_villain_ratio",
    "narrative_victim_ratio",
    "narrative_conflict_ratio",
    "narrative_resolution_ratio",
    "narrative_crisis_ratio",
    "narrative_polarization_ratio",
    "narrative_role_diversity",
    "narrative_conflict_intensity",
    "narrative_progression_score",
    "narrative_rhetoric_score",
]

NARRATIVE_FRAME_FEATURE_NAMES: List[str] = [
    "narrative_frame_conflict_ratio",
    "narrative_frame_economic_ratio",
    "narrative_frame_human_interest_ratio",
    "narrative_frame_moral_ratio",
    "narrative_frame_responsibility_ratio",
    "narrative_frame_diversity",
    "narrative_frame_dominance",
    "narrative_frame_balance",
    "narrative_frame_rhetoric_score",
]

NARRATIVE_ROLE_FEATURE_NAMES: List[str] = [
    "narrative_role_hero_ratio",
    "narrative_role_villain_ratio",
    "narrative_role_victim_ratio",
    "narrative_role_polarization_ratio",
    "narrative_role_balance",
    "narrative_role_diversity",
    "narrative_entity_density",
]

MANIPULATION_FEATURE_NAMES: List[str] = [
    "manipulation_urgency_ratio",
    "manipulation_fear_ratio",
    "manipulation_blame_ratio",
    "manipulation_scapegoat_ratio",
    "manipulation_absolute_ratio",
    "manipulation_conspiracy_ratio",
    "manipulation_false_dilemma_ratio",
    "manipulation_exaggeration_ratio",
    "manipulation_intensifier_ratio",
    "manipulation_exclamation_density",
    "manipulation_caps_emphasis",
    "manipulation_intensity",
    "manipulation_diversity",
]

PROPAGANDA_FEATURE_NAMES: List[str] = [
    "propaganda_name_calling_ratio",
    "propaganda_fear_ratio",
    "propaganda_exaggeration_ratio",
    "propaganda_glitter_ratio",
    "propaganda_us_vs_them_ratio",
    "propaganda_authority_ratio",
    "propaganda_intensifier_ratio",
    "propaganda_exclamation_density",
    "propaganda_caps_ratio",
    "propaganda_intensity",
    "propaganda_diversity",
]

PROPAGANDA_LEXICON_FEATURE_NAMES: List[str] = [
    "propaganda_name_calling_ratio",
    "propaganda_fear_ratio",
    "propaganda_exaggeration_ratio",
    "propaganda_bandwagon_ratio",
    "propaganda_slogan_ratio",
    "propaganda_phrase_bandwagon",
    "propaganda_phrase_slogan",
    "propaganda_exclamation_density",
    "propaganda_caps_ratio",
    "propaganda_lexicon_density",
    "propaganda_lexicon_diversity",
]

LEXICAL_FEATURE_NAMES: List[str] = [
    "vocabulary_size",
    "hapax_legomena_ratio",
    "hapax_dislegomena_ratio",
    "lexical_density",
    "average_word_length",
]

SEMANTIC_FEATURE_NAMES: List[str] = [
    "embedding_norm",
    "embedding_mean",
    "embedding_std",
    "embedding_max",
    "embedding_min",
]

SYNTACTIC_FEATURE_NAMES: List[str] = [
    "sentence_count",
    "avg_sentence_length",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "punctuation_ratio",
]

TOKEN_FEATURE_NAMES: List[str] = [
    "token_count",
    "unique_token_count",
    "type_token_ratio",
    "avg_token_length",
    "max_token_length",
    "repetition_ratio",
]

ALL_BIAS_MODULE_FEATURE_NAMES: List[str] = sorted(
    BIAS_FEATURE_NAMES + FRAMING_FEATURE_NAMES + IDEOLOGICAL_FEATURE_NAMES
)

ALL_DISCOURSE_FEATURE_NAMES: List[str] = sorted(
    ARGUMENT_STRUCTURE_FEATURE_NAMES + DISCOURSE_FEATURE_NAMES
)

ALL_GRAPH_FEATURE_NAMES: List[str] = sorted(
    ENTITY_GRAPH_FEATURE_NAMES + INTERACTION_GRAPH_FEATURE_NAMES
)

ALL_NARRATIVE_FEATURE_NAMES: List[str] = sorted(
    CONFLICT_FEATURE_NAMES
    + NARRATIVE_FEATURE_NAMES
    + NARRATIVE_FRAME_FEATURE_NAMES
    + NARRATIVE_ROLE_FEATURE_NAMES
)

ALL_PROPAGANDA_FEATURE_NAMES: List[str] = sorted(
    MANIPULATION_FEATURE_NAMES
    + PROPAGANDA_FEATURE_NAMES
    + PROPAGANDA_LEXICON_FEATURE_NAMES
)

ALL_TEXT_FEATURE_NAMES: List[str] = sorted(
    LEXICAL_FEATURE_NAMES
    + SEMANTIC_FEATURE_NAMES
    + SYNTACTIC_FEATURE_NAMES
    + TOKEN_FEATURE_NAMES
)

# ---------------------------------------------------------------------------
# Text feature key prefixes used by partition_feature_sections
# ---------------------------------------------------------------------------

_TEXT_KEY_PREFIXES = (
    "embedding_",
    "vocabulary_",
    "hapax_",
    "token_",
    "unique_token_",
    "type_token_",
    "avg_token_",
    "max_token_",
    "repetition_",
    "sentence_",
    "avg_sentence_",
    "noun_",
    "verb_",
    "adjective_",
    "adverb_",
    "punctuation_",
    "lexical_",
    "average_word_",
)


# ---------------------------------------------------------------------------
# Section partitioning helper
# ---------------------------------------------------------------------------

def partition_feature_sections(
    features: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    """
    Partition a flat feature dict from the pipeline into named sections.

    Routes features to one of the following sections:
        "bias"        — keys starting with ``bias_``
        "framing"     — keys starting with ``frame_``
        "ideology"    — keys starting with ``ideology_``
        "emotion"     — keys starting with ``emotion_`` or ``lexicon_emotion_``
        "discourse"   — keys starting with ``discourse_`` or ``argument_``
        "graph"       — keys starting with ``graph_``, ``graph_pipeline_``,
                        ``entity_``, or ``interaction_``
        "narrative"   — keys starting with ``narrative_`` or ``conflict_``
        "propaganda"  — keys starting with ``propaganda_`` or ``manipulation_``
        "text"        — token, lexical, semantic, and syntactic feature keys
        "other"       — everything else

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
        "discourse": {},
        "graph": {},
        "narrative": {},
        "propaganda": {},
        "text": {},
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
        elif key.startswith("discourse_") or key.startswith("argument_"):
            sections["discourse"][key] = value
        elif (
            key.startswith("graph_")
            or key.startswith("graph_pipeline_")
            or key.startswith("entity_")
            or key.startswith("interaction_")
        ):
            sections["graph"][key] = value
        elif key.startswith("narrative_") or key.startswith("conflict_"):
            sections["narrative"][key] = value
        elif key.startswith("propaganda_") or key.startswith("manipulation_"):
            sections["propaganda"][key] = value
        elif key.startswith(_TEXT_KEY_PREFIXES):
            sections["text"][key] = value
        else:
            sections["other"][key] = value

    return sections


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_ALL_FEATURE_MODULE_TYPES = (
    BiasFeatures,
    FramingFeatures,
    IdeologicalFeatures,
    ArgumentStructureFeatures,
    DiscourseFeatures,
    EntityGraphFeatures,
    InteractionGraphFeatures,
    ConflictFeatures,
    NarrativeFeatures,
    NarrativeFrameFeatures,
    NarrativeRoleFeatures,
    ManipulationPatterns,
    PropagandaFeatures,
    PropagandaLexiconFeatures,
    LexicalFeatures,
    SemanticFeatures,
    SyntacticFeatures,
    TokenFeatures,
)


@dataclass
class FeaturePipeline:
    """
    Main feature extraction pipeline.

    Responsibilities:
        • initialize feature extractors via FeatureRegistry
        • execute feature extraction across all registered modules
        • fuse outputs
        • optionally scale features
        • optionally apply feature selection

    Integrated feature modules and their output key counts:
        BiasFeatures             (bias_*)            10 features
        FramingFeatures          (frame_*)            10 features
        IdeologicalFeatures      (ideology_*)          8 features
        ArgumentStructureFeatures (argument_*)         7 features
        DiscourseFeatures        (discourse_*)         7 features
        EntityGraphFeatures      (entity_*)            5 features
        InteractionGraphFeatures (interaction_*)       6 features
        ConflictFeatures         (conflict_*)          9 features
        NarrativeFeatures        (narrative_*)        11 features
        NarrativeFrameFeatures   (narrative_frame_*)   9 features
        NarrativeRoleFeatures    (narrative_role_*)    7 features
        ManipulationPatterns     (manipulation_*)     13 features
        PropagandaFeatures       (propaganda_*)       11 features
        PropagandaLexiconFeatures (propaganda_*)      11 features
        LexicalFeatures          (vocabulary_/hapax_)  5 features
        SemanticFeatures         (embedding_*)         5 features
        SyntacticFeatures        (sentence_/pos_*)     7 features
        TokenFeatures            (token_*)             6 features

    Output key constants are available as module-level lists:
        BIAS_FEATURE_NAMES, FRAMING_FEATURE_NAMES, IDEOLOGICAL_FEATURE_NAMES,
        ARGUMENT_STRUCTURE_FEATURE_NAMES, DISCOURSE_FEATURE_NAMES,
        ENTITY_GRAPH_FEATURE_NAMES, INTERACTION_GRAPH_FEATURE_NAMES,
        CONFLICT_FEATURE_NAMES, NARRATIVE_FEATURE_NAMES,
        NARRATIVE_FRAME_FEATURE_NAMES, NARRATIVE_ROLE_FEATURE_NAMES,
        MANIPULATION_FEATURE_NAMES, PROPAGANDA_FEATURE_NAMES,
        PROPAGANDA_LEXICON_FEATURE_NAMES, LEXICAL_FEATURE_NAMES,
        SEMANTIC_FEATURE_NAMES, SYNTACTIC_FEATURE_NAMES, TOKEN_FEATURE_NAMES

    To partition extracted features by module section, use:
        partition_feature_sections(features)
    """

    feature_names: Optional[List[str]] = None
    scaler: Optional[FeatureScalingPipeline] = None
    selector: Optional[FeatureSelectionPipeline] = None
    validator: Optional[FeatureSchemaValidator] = None
    stats_enabled: bool = False

    features: List[BaseFeature] = field(default_factory=list)
    fusion: Optional[FeatureFusion] = None
    graph_pipeline: GraphPipeline | None = field(default=None, init=False, repr=False)

    def initialize(self) -> None:
        """
        Initialize feature extractors using FeatureRegistry.

        Calls bootstrap_feature_registry() which imports all registered
        feature modules, including all discourse, graph, narrative,
        propaganda, and text feature extractors.
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

        active_modules = [
            type(f).__name__
            for f in self.features
            if isinstance(f, _ALL_FEATURE_MODULE_TYPES)
        ]
        logger.info(
            "FeaturePipeline initialized | feature_count=%d active_modules=%s",
            len(self.features),
            active_modules,
        )

    def extract(self, context: FeatureContext) -> Dict[str, float]:
        """
        Extract features from a single FeatureContext.

        Output includes contributions from all registered feature modules.
        Use partition_feature_sections() on the result to separate into
        named sections: bias, framing, ideology, emotion, discourse, graph,
        narrative, propaganda, text, other.
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
        Sections: bias, framing, ideology, emotion, discourse, graph,
        narrative, propaganda, text, other.
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
            1. Feature extraction (all registered modules)
            2. Optional scaling
            3. Optional feature selection
        """

        features = self.batch_extract(contexts)

        if self.validator is not None:
            try:
                features = self.validator.validate_batch(features)
                logger.debug(
                    "Feature schema validation passed | samples=%d",
                    len(features),
                )
            except Exception as _val_exc:
                logger.warning("Feature schema validation failed: %s", _val_exc)

        if self.stats_enabled and features:
            try:
                _stats = FeatureStatistics()
                _summary = _stats.dataset_summary(features)
                logger.info(
                    "Feature statistics | samples=%d features=%d mean_variance=%.6f",
                    int(_summary["num_samples"]),
                    int(_summary["num_features"]),
                    _summary["mean_variance"],
                )
                _constant = _stats.detect_constant_features(features)
                if _constant:
                    logger.warning(
                        "Constant (zero-variance) features detected: %s",
                        _constant[:10],
                    )
            except Exception as _stats_exc:
                logger.warning("Feature statistics computation failed: %s", _stats_exc)

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
