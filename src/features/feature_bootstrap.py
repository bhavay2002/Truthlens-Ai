from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

_BOOTSTRAPPED = False

FEATURE_MODULES = [
    "src.features.text.lexical_features",
    "src.features.text.semantic_features",
    "src.features.text.syntactic_features",
    "src.features.text.token_features",
    "src.features.bias.bias_features",
    "src.features.bias.bias_lexicon_features",
    "src.features.bias.framing_features",
    "src.features.bias.ideological_features",
    "src.features.discourse.argument_structure_features",
    "src.features.discourse.discourse_features",
    "src.features.emotion.emotion_features",
    "src.features.emotion.emotion_intensity_features",
    "src.features.emotion.emotion_lexicon_features",
    "src.features.emotion.emotion_target_features",
    "src.features.emotion.emotion_trajectory_features",
    "src.features.graph.entity_graph_features",
    "src.features.graph.interaction_graph_features",
    "src.features.narrative.conflict_features",
    "src.features.narrative.narrative_features",
    "src.features.narrative.narrative_frame_features",
    "src.features.narrative.narrative_role_features",
    "src.features.propaganda.manipulation_patterns",
    "src.features.propaganda.propaganda_features",
    "src.features.propaganda.propaganda_lexicon_features",
    "src.features.analysis.analysis_adapter_features",
]


def bootstrap_feature_registry() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    for module_path in FEATURE_MODULES:
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Feature module import failed: %s (%s)", module_path, exc)

    _BOOTSTRAPPED = True

