# src/features/feature_schema.py

from __future__ import annotations
from typing import List

# =========================================================
# BIAS
# =========================================================

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

FRAMING_FEATURE_NAMES = [
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

IDEOLOGICAL_FEATURE_NAMES = [
    "ideology_left_ratio",
    "ideology_right_ratio",
    "ideology_balance",
    "ideology_entropy",
    "ideology_polarization_ratio",
    "ideology_group_reference_ratio",
    "ideology_phrase_count",
    "ideology_signal_strength",
]

# =========================================================
# DISCOURSE
# =========================================================

ARGUMENT_STRUCTURE_FEATURE_NAMES = [
    "argument_claim_ratio",
    "argument_premise_ratio",
    "argument_evidence_ratio",
    "argument_counterargument_ratio",
    "argument_rhetorical_question_ratio",
    "argument_structure_density",
    "argument_structure_diversity",
]

DISCOURSE_FEATURE_NAMES = [
    "discourse_causal_ratio",
    "discourse_contrast_ratio",
    "discourse_additive_ratio",
    "discourse_sequential_ratio",
    "discourse_evidential_ratio",
    "discourse_marker_density",
    "discourse_diversity",
]

# =========================================================
# GRAPH
# =========================================================

ENTITY_GRAPH_FEATURE_NAMES = [
    "entity_count",
    "entity_edge_count",
    "entity_avg_degree",
    "entity_density",
    "entity_centralization",
]

INTERACTION_GRAPH_FEATURE_NAMES = [
    "interaction_node_count",
    "interaction_edge_count",
    "interaction_avg_degree",
    "interaction_density",
    "interaction_clustering",
    "interaction_component_count",
]

# =========================================================
# NARRATIVE
# =========================================================

CONFLICT_FEATURE_NAMES = [
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

NARRATIVE_FEATURE_NAMES = [
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

# =========================================================
# PROPAGANDA
# =========================================================

PROPAGANDA_FEATURE_NAMES = [
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

# =========================================================
# TEXT
# =========================================================

LEXICAL_FEATURE_NAMES = [
    "vocabulary_size",
    "hapax_legomena_ratio",
    "hapax_dislegomena_ratio",
    "lexical_density",
    "average_word_length",
]

SEMANTIC_FEATURE_NAMES = [
    "embedding_norm",
    "embedding_mean",
    "embedding_std",
    "embedding_max",
    "embedding_min",
]

SYNTACTIC_FEATURE_NAMES = [
    "sentence_count",
    "avg_sentence_length",
    "noun_ratio",
    "verb_ratio",
    "adjective_ratio",
    "adverb_ratio",
    "punctuation_ratio",
]

TOKEN_FEATURE_NAMES = [
    "token_count",
    "unique_token_count",
    "type_token_ratio",
    "avg_token_length",
    "max_token_length",
    "repetition_ratio",
]

# =========================================================
# GLOBAL SCHEMA
# =========================================================

ALL_FEATURES: List[str] = sorted(
    BIAS_FEATURE_NAMES
    + FRAMING_FEATURE_NAMES
    + IDEOLOGICAL_FEATURE_NAMES
    + ARGUMENT_STRUCTURE_FEATURE_NAMES
    + DISCOURSE_FEATURE_NAMES
    + ENTITY_GRAPH_FEATURE_NAMES
    + INTERACTION_GRAPH_FEATURE_NAMES
    + CONFLICT_FEATURE_NAMES
    + NARRATIVE_FEATURE_NAMES
    + PROPAGANDA_FEATURE_NAMES
    + LEXICAL_FEATURE_NAMES
    + SEMANTIC_FEATURE_NAMES
    + SYNTACTIC_FEATURE_NAMES
    + TOKEN_FEATURE_NAMES
)