from __future__ import annotations

import importlib
import logging
from typing import List, Set

from src.features.base.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

_BOOTSTRAPPED = False


# =========================================================
# FEATURE MODULES (DEDUP SAFE)
# =========================================================

FEATURE_MODULES: List[str] = list(dict.fromkeys([
    # -------------------------
    # TEXT
    # -------------------------
    "src.features.text.lexical_features",
    "src.features.text.semantic_features",
    "src.features.text.syntactic_features",
    "src.features.text.token_features",

    # -------------------------
    # BIAS
    # -------------------------
    "src.features.bias.bias_features",
    "src.features.bias.bias_lexicon_features",
    "src.features.bias.framing_features",
    "src.features.bias.ideological_features",

    # -------------------------
    # DISCOURSE
    # -------------------------
    "src.features.discourse.argument_structure_features",
    "src.features.discourse.discourse_features",

    # -------------------------
    # EMOTION
    # -------------------------
    "src.features.emotion.emotion_features",  # ✅ confirmed present
    "src.features.emotion.emotion_intensity_features",
    "src.features.emotion.emotion_lexicon_features",
    "src.features.emotion.emotion_target_features",
    "src.features.emotion.emotion_trajectory_features",

    # -------------------------
    # GRAPH
    # -------------------------
    "src.features.graph.entity_graph_features",
    "src.features.graph.interaction_graph_features",

    # -------------------------
    # NARRATIVE
    # -------------------------
    "src.features.narrative.conflict_features",
    "src.features.narrative.narrative_features",
    "src.features.narrative.narrative_frame_features",
    "src.features.narrative.narrative_role_features",  # ✅ confirmed present

    # -------------------------
    # PROPAGANDA
    # -------------------------
    "src.features.propaganda.manipulation_patterns",
    "src.features.propaganda.propaganda_features",
    "src.features.propaganda.propaganda_lexicon_features",

    # -------------------------
    # ANALYSIS ADAPTER
    # -------------------------
    "src.features.analysis.analysis_adapter_features",
]))


# =========================================================
# BOOTSTRAP
# =========================================================

def bootstrap_feature_registry(
    *,
    strict: bool = False,
    reload: bool = False,
    auto_discover: bool = False,
    auto_package: str = "src.features",
) -> None:
    """
    Initialize and register all feature modules.
    """

    global _BOOTSTRAPPED

    if _BOOTSTRAPPED and not reload:
        logger.debug("Feature registry already bootstrapped")
        return

    success = 0
    failed = []
    loaded_modules: Set[str] = set()

    # -----------------------------------------------------
    # Manual module loading
    # -----------------------------------------------------

    for module_path in FEATURE_MODULES:

        if module_path in loaded_modules:
            continue

        try:
            if reload:
                importlib.invalidate_caches()

            importlib.import_module(module_path)

            loaded_modules.add(module_path)
            success += 1

        except Exception as exc:
            failed.append((module_path, exc))
            logger.warning(
                "Feature module import failed: %s (%s)",
                module_path,
                exc,
            )

    # -----------------------------------------------------
    # Auto discovery (optional)
    # -----------------------------------------------------

    if auto_discover:
        try:
            FeatureRegistry.auto_discover(auto_package)
            logger.info("Auto-discovery completed: %s", auto_package)
        except Exception as exc:
            logger.warning("Auto-discovery failed: %s", exc)
            if strict:
                raise

    # -----------------------------------------------------
    # Validation check (NEW 🔥)
    # -----------------------------------------------------

    try:
        registered = FeatureRegistry.list_features()

        if not registered:
            raise RuntimeError("No features registered after bootstrap")

        logger.info("Registered features: %d", len(registered))

        # Audit fix §9 — startup diff log. Surface (a) modules that
        # were declared in FEATURE_MODULES but failed to register any
        # extractor and (b) extractors that registered themselves
        # without being in the manual list. Both are silent
        # configuration drift in the previous code path.
        expected_modules = set(loaded_modules)
        registered_modules = set()
        for name in registered:
            try:
                meta = FeatureRegistry.get_metadata(name)
                mod = meta.get("module")
                if mod:
                    registered_modules.add(mod)
            except Exception:
                continue

        missing_from_registry = sorted(
            m for m in expected_modules if m not in registered_modules
        )
        if missing_from_registry:
            logger.warning(
                "Feature bootstrap diff | imported but no extractor registered: %s",
                missing_from_registry,
            )

        unexpected_in_registry = sorted(
            m for m in registered_modules if m not in expected_modules
        )
        if unexpected_in_registry:
            logger.warning(
                "Feature bootstrap diff | extractor registered outside FEATURE_MODULES: %s",
                unexpected_in_registry,
            )

    except Exception as exc:
        logger.error("Feature registry validation failed: %s", exc)
        if strict:
            raise

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    total = len(FEATURE_MODULES)

    logger.info(
        "Feature bootstrap complete | loaded=%d failed=%d total=%d",
        success,
        len(failed),
        total,
    )

    if failed:
        for mod, err in failed:
            logger.debug("FAILED → %s | %s", mod, err)

        if strict:
            raise RuntimeError(
                f"{len(failed)} feature modules failed to load"
            )

    # -----------------------------------------------------
    # Freeze registry (production safety)
    # -----------------------------------------------------

    try:
        FeatureRegistry.freeze()
        logger.debug("FeatureRegistry frozen")
    except Exception:
        logger.debug("FeatureRegistry freeze skipped")

    _BOOTSTRAPPED = True


# =========================================================
# DEBUG UTIL
# =========================================================

def list_loaded_features() -> List[str]:
    """
    Return all registered feature names.
    """
    return FeatureRegistry.list_features()