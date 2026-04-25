from __future__ import annotations

"""
Central Feature Schema for TruthLens

Defines:
- Canonical feature names
- Section-wise grouping
- Global schema used by validator / model / training

This is the SINGLE SOURCE OF TRUTH for feature structure.
"""

from typing import Dict, List


# =========================================================
# BIAS
# =========================================================

BIAS_FEATURES = [
    "bias_loaded",
    "bias_subjective",
    "bias_uncertainty",
    "bias_polarization",
    "bias_evaluative",
    "bias_intensity",
    "bias_variance",
    "bias_caps_ratio",
    "bias_exclamation_density",
]


# =========================================================
# DISCOURSE
# =========================================================

DISCOURSE_FEATURES = [
    "discourse_causal_ratio",
    "discourse_contrast_ratio",
    "discourse_additive_ratio",
    "discourse_sequential_ratio",
    "discourse_evidential_ratio",
    "discourse_marker_density",
    "discourse_diversity",
]


# =========================================================
# ARGUMENT STRUCTURE
# =========================================================

ARGUMENT_FEATURES = [
    "argument_claim_ratio",
    "argument_premise_ratio",
    "argument_evidence_ratio",
    "argument_counterargument_ratio",
    "argument_rhetorical_question_ratio",
    "argument_structure_density",
    "argument_structure_diversity",
]


# =========================================================
# EMOTION
# =========================================================

# NOTE: dynamic generation based on schema
EMOTION_LABELS = [
    "neutral","admiration","approval","gratitude","annoyance","amusement",
    "curiosity","disapproval","love","optimism","anger","joy","confusion",
    "sadness","disappointment","realization","caring","surprise",
    "excitement","disgust"
]

EMOTION_FEATURES = (
    [f"emotion_{e}" for e in EMOTION_LABELS] +
    ["emotion_intensity"] +
    [f"emotion_dominant_{e}" for e in EMOTION_LABELS]
)


# =========================================================
# NARRATIVE
# =========================================================

NARRATIVE_FEATURES = [
    "narrative_role_hero_ratio",
    "narrative_role_villain_ratio",
    "narrative_role_victim_ratio",
    "narrative_role_polarization_ratio",
    "narrative_role_balance",
    "narrative_role_diversity",
    "narrative_entity_density",
]


# =========================================================
# GRAPH
# =========================================================

GRAPH_FEATURES = [
    # base
    "entity_count",
    "entity_edge_count",
    "entity_avg_degree",
    "entity_density",
    "entity_centralization",

    "interaction_node_count",
    "interaction_edge_count",
    "interaction_avg_degree",
    "interaction_density",
    "interaction_clustering",
    "interaction_component_count",
]

# GraphPipeline outputs (dynamic prefix)
GRAPH_PIPELINE_FEATURES = [
    "graph_pipeline_entity_density",
    "graph_pipeline_entity_centralization",
    "graph_pipeline_narrative_flow",
    "graph_pipeline_narrative_coherence",
]


# =========================================================
# TEXT FEATURES
# =========================================================

LEXICAL_FEATURES = [
    "vocabulary_size",
    "hapax_legomena_ratio",
    "hapax_dislegomena_ratio",
    "lexical_density",
    "average_word_length",
]

SYNTACTIC_FEATURES = [
    "sentence_count",
    "avg_sentence_length",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "punctuation_ratio",
]

TOKEN_FEATURES = [
    "token_count",
    "unique_token_count",
    "type_token_ratio",
    "avg_token_length",
    "max_token_length",
    "repetition_ratio",
]


# =========================================================
# ALL FEATURES (MASTER LIST)
# =========================================================

ALL_FEATURES: List[str] = sorted(
    BIAS_FEATURES
    + DISCOURSE_FEATURES
    + ARGUMENT_FEATURES
    + EMOTION_FEATURES
    + NARRATIVE_FEATURES
    + GRAPH_FEATURES
    + GRAPH_PIPELINE_FEATURES
    + LEXICAL_FEATURES
    + SYNTACTIC_FEATURES
    + TOKEN_FEATURES
)


# =========================================================
# SECTION MAP
# =========================================================

FEATURE_SECTIONS: Dict[str, List[str]] = {
    "bias": BIAS_FEATURES,
    "discourse": DISCOURSE_FEATURES + ARGUMENT_FEATURES,
    "emotion": EMOTION_FEATURES,
    "narrative": NARRATIVE_FEATURES,
    "graph": GRAPH_FEATURES + GRAPH_PIPELINE_FEATURES,
    "text": LEXICAL_FEATURES + SYNTACTIC_FEATURES + TOKEN_FEATURES,
}


# =========================================================
# HELPERS
# =========================================================

def get_all_features() -> List[str]:
    """Return full feature schema."""
    return ALL_FEATURES


def get_feature_sections() -> Dict[str, List[str]]:
    """Return section-wise feature mapping."""
    return FEATURE_SECTIONS


def validate_feature_names(features: List[str]) -> None:
    """Sanity check for schema integrity."""
    duplicates = {f for f in features if features.count(f) > 1}
    if duplicates:
        raise ValueError(f"Duplicate features in schema: {duplicates}")