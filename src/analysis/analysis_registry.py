# src/analysis/analysis_registry.py

from __future__ import annotations

import logging
from typing import Dict, List, Callable, Optional, Any

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext

logger = logging.getLogger(__name__)


# =========================================================
# REGISTRY ENTRY
# =========================================================

class AnalyzerSpec:
    """
    Metadata wrapper for analyzers.
    Enables dependency management + configurability.
    """

    def __init__(
        self,
        name: str,
        analyzer: BaseAnalyzer,
        *,
        enabled: bool = True,
        requires: Optional[List[str]] = None,
        provides: Optional[List[str]] = None,
        order: int = 0,
    ):
        self.name = name
        self.analyzer = analyzer
        self.enabled = enabled
        self.requires = requires or []
        self.provides = provides or []
        self.order = order


# =========================================================
# REGISTRY
# =========================================================

class AnalyzerRegistry:

    def __init__(self):
        self._registry: Dict[str, AnalyzerSpec] = {}

    # -----------------------------------------------------

    def register(
        self,
        name: str,
        analyzer: BaseAnalyzer,
        *,
        enabled: bool = True,
        requires: Optional[List[str]] = None,
        provides: Optional[List[str]] = None,
        order: int = 0,
    ) -> None:

        if name in self._registry:
            raise ValueError(f"Analyzer '{name}' already registered")

        self._registry[name] = AnalyzerSpec(
            name=name,
            analyzer=analyzer,
            enabled=enabled,
            requires=requires,
            provides=provides,
            order=order,
        )

        logger.debug("Registered analyzer: %s", name)

    # -----------------------------------------------------

    def enable(self, name: str):
        self._get(name).enabled = True

    def disable(self, name: str):
        self._get(name).enabled = False

    # -----------------------------------------------------

    def get_active(self) -> List[AnalyzerSpec]:
        return [
            spec for spec in self._registry.values()
            if spec.enabled
        ]

    # -----------------------------------------------------

    def get_ordered(self) -> List[AnalyzerSpec]:
        """
        Returns analyzers sorted by execution order.
        """
        return sorted(self.get_active(), key=lambda x: x.order)

    # -----------------------------------------------------

    def run_all(
        self,
        ctx: FeatureContext,
        *,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, float]]:

        results: Dict[str, Dict[str, float]] = {}
        extra_inputs = extra_inputs or {}

        for spec in self.get_ordered():

            try:
                # dependency check
                for dep in spec.requires:
                    if dep not in results:
                        raise RuntimeError(
                            f"Analyzer '{spec.name}' requires '{dep}'"
                        )

                # run analyzer
                output = self._safe_run(
                    spec,
                    ctx,
                    results,
                    extra_inputs,
                )

                results[spec.name] = output

            except Exception:
                logger.exception("Analyzer failed: %s", spec.name)
                results[spec.name] = {}

        return results

    # -----------------------------------------------------

    def _safe_run(
        self,
        spec: AnalyzerSpec,
        ctx: FeatureContext,
        results: Dict[str, Dict[str, float]],
        extra_inputs: Dict[str, Any],
    ) -> Dict[str, float]:

        analyzer = spec.analyzer

        # pass dependencies if needed
        kwargs = {}

        if spec.requires:
            kwargs.update({
                dep: results.get(dep, {})
                for dep in spec.requires
            })

        kwargs.update(extra_inputs)

        output = analyzer.analyze(ctx, **kwargs)

        if not isinstance(output, dict):
            raise TypeError(
                f"Analyzer '{spec.name}' returned non-dict output"
            )

        return output

    # -----------------------------------------------------

    def list(self) -> List[str]:
        return list(self._registry.keys())

    # -----------------------------------------------------

    def _get(self, name: str) -> AnalyzerSpec:
        if name not in self._registry:
            raise KeyError(f"Analyzer '{name}' not found")
        return self._registry[name]


# =========================================================
# DEFAULT REGISTRY BUILDER
# =========================================================

def build_default_registry() -> AnalyzerRegistry:
    """
    Central place to wire all analyzers.
    """

    from src.analysis.rhetorical_device_detector import RhetoricalDeviceDetector
    from src.analysis.argument_analyzer import ArgumentAnalyzer
    from src.analysis.context_omission_analyzer import ContextOmissionAnalyzer
    from src.analysis.discourse_coherence_analyzer import DiscourseCoherenceAnalyzer
    from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
    from src.analysis.framing_analysis import FramingAnalyzer
    from src.analysis.information_density import InformationDensityAnalyzer
    from src.analysis.information_omission_detector import InformationOmissionDetector
    from src.analysis.ideological_language_detector import IdeologicalLanguageDetector
    from src.analysis.narrative_conflict import NarrativeConflictAnalyzer
    from src.analysis.narrative_propagation import NarrativePropagationAnalyzer
    from src.analysis.narrative_temporal_analyzer import NarrativeTemporalAnalyzer
    from src.analysis.source_attribution_analyzer import SourceAttributionAnalyzer

    registry = AnalyzerRegistry()

    # -----------------------------------------------------
    # Register analyzers with execution order
    # -----------------------------------------------------

    registry.register("rhetorical", RhetoricalDeviceDetector(), order=1)
    registry.register("argument", ArgumentAnalyzer(), order=2)
    registry.register("context", ContextOmissionAnalyzer(), order=3)
    registry.register("discourse", DiscourseCoherenceAnalyzer(), order=4)
    registry.register("emotion", EmotionTargetAnalyzer(), order=5)
    registry.register("framing", FramingAnalyzer(), order=6)
    registry.register("information", InformationDensityAnalyzer(), order=7)
    registry.register("omission", InformationOmissionDetector(), order=8)
    registry.register("ideology", IdeologicalLanguageDetector(), order=9)

    # narrative depends on others
    registry.register(
        "conflict",
        NarrativeConflictAnalyzer(),
        requires=["framing"],
        order=10,
    )

    registry.register(
        "propagation",
        NarrativePropagationAnalyzer(),
        requires=["conflict"],
        order=11,
    )

    registry.register("temporal", NarrativeTemporalAnalyzer(), order=12)
    registry.register("source", SourceAttributionAnalyzer(), order=13)

    return registry