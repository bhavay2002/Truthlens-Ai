"""
File Name: feature_schema.py
Module: Analysis - Central Feature Schema Registry

Description:
    Defines the canonical ordered key lists for every analysis module and
    provides a single :func:`make_vector` helper that converts feature dicts
    into deterministic, schema-ordered ``np.float32`` arrays.

    Motivation
    ----------
    * **Deterministic ordering** – vectors are always produced in the same key
      order regardless of Python dict insertion order or future refactors.
    * **Stable dtype** – all vectors use ``np.float32``.
    * **Missing-key policy** – absent keys default to ``0.0`` in normal mode;
      ``strict=True`` raises ``ValueError`` instead (useful for tests).
    * **Backward compatibility** – the module-level ``*_KEYS`` constants can be
      imported by the original ``*_vector()`` functions so those functions remain
      callable with the same signature while delegating to ``make_vector``.

Usage
-----
::

    from src.analysis.feature_schema import ARGUMENT_MINING_KEYS, make_vector

    vec = make_vector(features, ARGUMENT_MINING_KEYS)
    # strict mode (e.g. in tests):
    vec = make_vector(features, ARGUMENT_MINING_KEYS, strict=True)
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-analyzer canonical key registries
# ---------------------------------------------------------------------------

ARGUMENT_MINING_KEYS: List[str] = [
    "argument_claim_ratio",
    "argument_premise_ratio",
    "argument_support_ratio",
    "argument_contrast_ratio",
    "argument_rebuttal_ratio",
    "argument_verb_density",
    "argument_clause_density",
    "argument_complexity",
]

CONTEXT_OMISSION_KEYS: List[str] = [
    "context_vague_reference_ratio",
    "context_attribution_ratio",
    "context_evidence_ratio",
    "context_uncertainty_ratio",
    "context_quote_ratio",
    "context_entity_ratio",
    "context_entity_type_diversity",
    "context_grounding_score",
]

DISCOURSE_COHERENCE_KEYS: List[str] = [
    "sentence_coherence",
    "topic_drift",
    "narrative_continuity",
    "discourse_transition_ratio",
]

EMOTION_TARGET_KEYS: List[str] = [
    "emotion_target_diversity",
    "emotion_target_focus",
    "emotion_expression_ratio",
    "emotion_type_diversity",
    "dominant_emotion_strength",
]

FRAMING_KEYS: List[str] = [
    "frame_conflict_score",
    "frame_economic_score",
    "frame_moral_score",
    "frame_human_interest_score",
    "frame_security_score",
]

IDEOLOGICAL_LANGUAGE_KEYS: List[str] = [
    "liberty_language_ratio",
    "equality_language_ratio",
    "tradition_language_ratio",
    "anti_elite_language_ratio",
    "liberty_vs_equality_balance",
    "ideology_phrase_density",
]

INFORMATION_DENSITY_KEYS: List[str] = [
    "factual_density",
    "opinion_density",
    "claim_density",
    "rhetorical_density",
    "emotion_density",
    "modal_density",
    "rhetorical_punctuation_density",
    "information_emotion_ratio",
]

INFORMATION_OMISSION_KEYS: List[str] = [
    "missing_counterargument_score",
    "one_sided_framing_score",
    "incomplete_evidence_score",
    "claim_evidence_imbalance",
]

NARRATIVE_CONFLICT_KEYS: List[str] = [
    "conflict_verb_ratio",
    "opposition_marker_ratio",
    "polarization_ratio",
    "hero_mentions",
    "villain_mentions",
    "victim_mentions",
    "hero_villain_conflict_score",
    "villain_victim_harm_score",
    "hero_victim_protection_score",
]

NARRATIVE_PROPAGATION_KEYS: List[str] = [
    "violent_conflict_ratio",
    "political_conflict_ratio",
    "discursive_conflict_ratio",
    "institutional_conflict_ratio",
    "coercion_conflict_ratio",
    "opposition_marker_ratio",
    "polarization_ratio",
    "conflict_phrase_count",
    "hero_mentions",
    "villain_mentions",
    "victim_mentions",
    "hero_villain_conflict_score",
    "villain_victim_harm_score",
    "hero_victim_protection_score",
]

NARRATIVE_TEMPORAL_KEYS: List[str] = [
    "past_framing_ratio",
    "crisis_escalation_ratio",
    "urgency_language_ratio",
    "past_tense_ratio",
    "present_tense_ratio",
    "future_tense_ratio",
    "temporal_contrast_score",
]

PROPAGANDA_PATTERN_KEYS: List[str] = [
    "fear_propaganda_score",
    "scapegoating_score",
    "polarization_score",
    "emotional_amplification_score",
    "narrative_imbalance_score",
]

RHETORICAL_DEVICE_KEYS: List[str] = [
    "rhetoric_exaggeration_score",
    "rhetoric_loaded_language_score",
    "rhetoric_emotional_appeal_score",
    "rhetoric_fear_appeal_score",
    "rhetoric_intensifier_ratio",
    "rhetoric_scapegoating_score",
    "rhetoric_false_dilemma_score",
    "rhetoric_punctuation_score",
]

SOURCE_ATTRIBUTION_KEYS: List[str] = [
    "expert_attribution_ratio",
    "anonymous_source_ratio",
    "credibility_indicator_ratio",
    "attribution_verb_ratio",
    "quotation_ratio",
    "named_source_ratio",
    "source_credibility_balance",
]


# ---------------------------------------------------------------------------
# Vectorisation helper
# ---------------------------------------------------------------------------


def make_vector(
    features: Dict[str, float],
    schema_keys: List[str],
    *,
    strict: bool = False,
) -> np.ndarray:
    """Convert a feature dict to a deterministic ``np.float32`` vector.

    The vector length and element order are fully determined by *schema_keys*,
    independent of the order in which keys were inserted into *features*.

    Args:
        features:    Mapping of feature name → numeric value.
        schema_keys: Ordered list of canonical key names for this vector type.
        strict:      When ``True``, raises :class:`ValueError` if any key in
                     *schema_keys* is absent from *features*, or if *features*
                     contains keys not present in *schema_keys*.  Useful in
                     tests and debug builds.

    Returns:
        ``np.ndarray`` of shape ``(len(schema_keys),)`` and dtype ``float32``.
        Missing keys are filled with ``0.0`` (unless *strict* is ``True``).

    Raises:
        ValueError: In strict mode when keys are missing or unknown.
    """
    if strict:
        missing = [k for k in schema_keys if k not in features]
        if missing:
            raise ValueError(f"Missing required feature keys: {missing}")
        unknown = [k for k in features if k not in schema_keys]
        if unknown:
            raise ValueError(f"Unknown feature keys not in schema: {unknown}")

    return np.array(
        [float(features.get(k, 0.0)) for k in schema_keys],
        dtype=np.float32,
    )


def validate_features(
    features: Dict[str, float],
    schema_keys: List[str],
) -> bool:
    """Validate that *features* contains numeric values for all *schema_keys*.

    Logs a warning for every non-numeric value but does not raise.

    Args:
        features:    Feature dictionary to validate.
        schema_keys: Expected key list from the schema registry.

    Returns:
        ``True`` if all values present in *features* are numeric, else
        ``False``.
    """
    ok = True
    for k in schema_keys:
        v = features.get(k)
        if v is not None and not isinstance(v, (int, float)):
            logger.warning("Non-numeric feature value for key '%s': %r", k, v)
            ok = False
    return ok
