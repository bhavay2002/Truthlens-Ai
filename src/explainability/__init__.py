"""
src.explainability
==================
Explainability sub-package for TruthLens AI.

Public API
----------
ExplainabilityOrchestrator
    Single owner of the full explainability lifecycle (SHAP, LIME, bias,
    emotion, attention rollout, propaganda, aggregation, consistency).
ExplainabilityConfig
    Dataclass that controls which components the orchestrator activates.
"""

from src.explainability.orchestrator import (
    ExplainabilityConfig,
    ExplainabilityOrchestrator,
)

__all__ = [
    "ExplainabilityConfig",
    "ExplainabilityOrchestrator",
]
