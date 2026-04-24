from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema Registry (IMMUTABLE + SAFE)
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: Dict[str, Tuple[str, ...]] = {}


def register_schema(name: str, keys: List[str]) -> None:
    if name in SCHEMA_REGISTRY:
        logger.warning("Overwriting schema: %s", name)

    # 🔥 enforce immutability
    SCHEMA_REGISTRY[name] = tuple(keys)


def get_schema(name: str) -> Tuple[str, ...]:
    if name not in SCHEMA_REGISTRY:
        raise ValueError(f"Schema not found: {name}")
    return SCHEMA_REGISTRY[name]


# ---------------------------------------------------------------------------
# FULL SCHEMA DEFINITIONS (🔥 COMPLETE)
# ---------------------------------------------------------------------------

ARGUMENT_MINING_KEYS = [
    "argument_claim_ratio",
    "argument_premise_ratio",
    "argument_support_ratio",
    "argument_contrast_ratio",
    "argument_rebuttal_ratio",
    "argument_verb_density",
    "argument_clause_density",
    "argument_complexity",
]

CONTEXT_OMISSION_KEYS = [
    "context_vague_reference_ratio",
    "context_attribution_ratio",
    "context_evidence_ratio",
    "context_uncertainty_ratio",
    "context_quote_ratio",
    "context_entity_ratio",
    "context_entity_type_diversity",
    "context_grounding_score",
]

DISCOURSE_COHERENCE_KEYS = [
    "sentence_coherence",
    "topic_drift",
    "narrative_continuity",
    "discourse_transition_ratio",
]

EMOTION_TARGET_KEYS = [
    "emotion_target_diversity",
    "emotion_target_focus",
    "emotion_expression_ratio",
    "emotion_type_diversity",
    "dominant_emotion_strength",
]

RHETORICAL_DEVICE_KEYS = [
    "rhetoric_exaggeration_score",
    "rhetoric_loaded_language_score",
    "rhetoric_emotional_appeal_score",
    "rhetoric_fear_appeal_score",
    "rhetoric_intensifier_ratio",
    "rhetoric_scapegoating_score",
    "rhetoric_false_dilemma_score",
    "rhetoric_punctuation_score",
]

FRAMING_KEYS = [
    "frame_conflict_score",
    "frame_economic_score",
    "frame_moral_score",
    "frame_human_interest_score",
    "frame_security_score",
    "frame_dominance_score",
    "frame_diversity_score",
]

INFORMATION_DENSITY_KEYS = [
    "factual_density",
    "opinion_density",
    "claim_density",
    "rhetorical_density",
    "emotion_density",
    "modal_density",
    "rhetorical_punctuation_density",
    "information_emotion_ratio",
    "information_emotion_ratio_normalized",
]

IDEOLOGICAL_LANGUAGE_KEYS = [
    "liberty_language_ratio",
    "equality_language_ratio",
    "tradition_language_ratio",
    "anti_elite_language_ratio",
    "liberty_vs_equality_balance",
    "ideology_phrase_density",
]

SOURCE_ATTRIBUTION_KEYS = [
    "expert_attribution_ratio",
    "anonymous_source_ratio",
    "credibility_indicator_ratio",
    "attribution_verb_ratio",
    "quotation_ratio",
    "named_source_ratio",
    "source_credibility_balance",
]

PROPAGANDA_PATTERN_KEYS = [
    "fear_propaganda_score",
    "scapegoating_score",
    "polarization_score",
    "emotional_amplification_score",
    "narrative_imbalance_score",
]


# 🔥 REGISTER ALL
register_schema("argument_mining", ARGUMENT_MINING_KEYS)
register_schema("context_omission", CONTEXT_OMISSION_KEYS)
register_schema("discourse_coherence", DISCOURSE_COHERENCE_KEYS)
register_schema("emotion_target", EMOTION_TARGET_KEYS)
register_schema("rhetorical", RHETORICAL_DEVICE_KEYS)
register_schema("framing", FRAMING_KEYS)
register_schema("information_density", INFORMATION_DENSITY_KEYS)
register_schema("ideology", IDEOLOGICAL_LANGUAGE_KEYS)
register_schema("source", SOURCE_ATTRIBUTION_KEYS)
register_schema("propaganda", PROPAGANDA_PATTERN_KEYS)


# ---------------------------------------------------------------------------
# Vectorization (STRONGER)
# ---------------------------------------------------------------------------

def make_vector(
    features: Dict[str, float],
    keys: Tuple[str, ...],
    *,
    strict: bool = False,
    safe: bool = True,
    clip: Tuple[float, float] | None = (0.0, 1.0),
) -> np.ndarray:

    if features is None:
        raise ValueError("features cannot be None")

    values = []

    for k in keys:
        v = features.get(k, 0.0)

        if safe:
            if not isinstance(v, (int, float)):
                v = 0.0
            elif np.isnan(v) or np.isinf(v):
                v = 0.0

        v = float(v)

        # 🔥 optional clipping
        if clip is not None:
            v = float(np.clip(v, clip[0], clip[1]))

        values.append(v)

    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Schema-based vectorization
# ---------------------------------------------------------------------------

def make_vector_from_schema(
    features: Dict[str, float],
    schema_name: str,
    *,
    strict: bool = False,
) -> np.ndarray:

    keys = get_schema(schema_name)
    return make_vector(features, keys, strict=strict)


# ---------------------------------------------------------------------------
# Validation (STRICT MODE ADDED)
# ---------------------------------------------------------------------------

def validate_features(
    features: Dict[str, float],
    schema_keys: Tuple[str, ...],
    *,
    strict: bool = False,
) -> bool:

    ok = True

    for k in schema_keys:
        if k not in features:
            if strict:
                logger.error("Missing key: %s", k)
                return False
            continue

        v = features[k]

        if not isinstance(v, (int, float)):
            logger.warning("Non-numeric value for %s", k)
            ok = False

        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            logger.warning("Invalid value for %s", k)
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "2.0.0"


def get_schema_metadata() -> Dict[str, str]:
    return {
        "version": SCHEMA_VERSION,
        "num_schemas": str(len(SCHEMA_REGISTRY)),
    }