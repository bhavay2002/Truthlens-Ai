"""
File Name: spacy_config.py
Module: Analysis - spaCy Loader Configuration

Description:
    Centralized configuration for the shared spaCy loader (:mod:`spacy_loader`).
    Defines the default model, GPU preference, batch / process counts, and the
    per-task pipeline-disable map. Lives in its own module to keep the spaCy
    settings decoupled from the broader :class:`AnalysisConfig` and to avoid
    import cycles between ``analysis_config`` and ``spacy_loader``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


# =========================================================
# DEFAULTS
# =========================================================

DEFAULT_SPACY_MODEL = "en_core_web_sm"
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_PROCESS = 1


# =========================================================
# TASK → PIPELINE DISABLE MAP
# =========================================================
# Maps a logical "task" name (used by analyzers) to the spaCy pipeline
# components that should be DISABLED for that task. Keeping pipelines lean
# avoids unnecessary work when an analyzer only needs lemmas or entities.
#
# Tasks used in the codebase: "syntax" (default, needs tagger/parser/lemmatizer
# for POS / DEP / lemma access) and "ner" (needs NER pipe + lemmatizer).

DEFAULT_TASK_DISABLE_MAP: Dict[str, Tuple[str, ...]] = {
    "syntax": (),
    "ner": (),
    "fast": ("ner", "tagger", "parser", "attribute_ruler", "lemmatizer"),
}


# =========================================================
# CONFIG
# =========================================================

@dataclass(slots=True)
class SpacyConfig:
    """spaCy loader configuration."""

    model: str = DEFAULT_SPACY_MODEL
    use_gpu: bool = False
    batch_size: int = DEFAULT_BATCH_SIZE
    n_process: int = DEFAULT_N_PROCESS

    task_disable_map: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: dict(DEFAULT_TASK_DISABLE_MAP)
    )


__all__ = [
    "SpacyConfig",
    "DEFAULT_SPACY_MODEL",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_N_PROCESS",
    "DEFAULT_TASK_DISABLE_MAP",
]
