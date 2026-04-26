from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext

logger = logging.getLogger(__name__)


# =========================================================
# SPEC
# =========================================================

@dataclass
class AnalyzerSpec:
    name: str
    analyzer: BaseAnalyzer

    enabled: bool = True
    requires: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)

    order: int = 0
    critical: bool = False  # 🔥 new (fail pipeline if breaks)


# =========================================================
# EXECUTION RESULT (NEW)
# =========================================================

@dataclass
class AnalyzerExecution:
    output: Dict[str, float]
    latency: float
    success: bool
    error: Optional[str] = None


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
        critical: bool = False,
    ) -> None:

        if name in self._registry:
            raise ValueError(f"Analyzer '{name}' already registered")

        self._registry[name] = AnalyzerSpec(
            name=name,
            analyzer=analyzer,
            enabled=enabled,
            requires=requires or [],
            provides=provides or [],
            order=order,
            critical=critical,
        )

    # -----------------------------------------------------

    def get_active(self) -> List[AnalyzerSpec]:
        return [s for s in self._registry.values() if s.enabled]

    # -----------------------------------------------------

    def get_ordered(self) -> List[AnalyzerSpec]:
        ordered = sorted(self.get_active(), key=lambda x: x.order)
        self._validate_dependencies(ordered)
        return ordered

    # -----------------------------------------------------
    # 🔥 DEPENDENCY VALIDATION (NEW)
    # -----------------------------------------------------

    def _validate_dependencies(self, specs: List[AnalyzerSpec]):

        names = {s.name for s in specs}

        for spec in specs:
            for dep in spec.requires:
                if dep not in names:
                    raise RuntimeError(
                        f"Analyzer '{spec.name}' requires missing '{dep}'"
                    )

        # cycle detection (simple DFS)
        visited = set()
        stack = set()

        def dfs(node: str):
            if node in stack:
                raise RuntimeError(f"Cyclic dependency detected at '{node}'")
            if node in visited:
                return

            stack.add(node)
            visited.add(node)

            for dep in self._registry[node].requires:
                dfs(dep)

            stack.remove(node)

        for s in specs:
            dfs(s.name)

    # -----------------------------------------------------
    # 🔥 MAIN EXECUTION (UPGRADED)
    # -----------------------------------------------------

    def run_all(
        self,
        ctx: FeatureContext,
        *,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, AnalyzerExecution]:

        extra_inputs = extra_inputs or {}
        results: Dict[str, AnalyzerExecution] = {}

        for spec in self.get_ordered():

            start = time.time()

            try:
                # Per-analyzer keyword arguments. We intentionally do NOT
                # forward the dependency dict as kwargs because each
                # analyzer's `analyze()` has its own keyword contract
                # (e.g. narrative analyzers want `hero_entities=` not
                # the full provider's output dict). Dependencies are
                # exposed via `extra_inputs` only when explicitly named.
                kwargs = dict(extra_inputs)

                # Invoke through the BaseAnalyzer __call__ wrapper when
                # available so caching, validation, and fallbacks run.
                runner = (
                    spec.analyzer
                    if callable(spec.analyzer)
                    else spec.analyzer.analyze
                )

                output = runner(ctx, **kwargs) if kwargs else runner(ctx)

                if not isinstance(output, dict):
                    raise TypeError("Analyzer output must be dict")

                latency = time.time() - start

                results[spec.name] = AnalyzerExecution(
                    output=output,
                    latency=latency,
                    success=True,
                )

            except Exception as e:

                latency = time.time() - start

                logger.exception("Analyzer failed: %s", spec.name)

                if spec.critical:
                    raise RuntimeError(
                        f"Critical analyzer failed: {spec.name}"
                    ) from e

                results[spec.name] = AnalyzerExecution(
                    output={},
                    latency=latency,
                    success=False,
                    error=str(e),
                )

        return results

    # -----------------------------------------------------

    def list(self) -> List[str]:
        return list(self._registry.keys())


# =========================================================
# DEFAULT REGISTRY (PRODUCTION SET)
# =========================================================

def build_default_registry() -> "AnalyzerRegistry":
    """
    Construct the default analyzer registry used by
    :class:`AnalysisPipeline`.

    Imports are local to keep the module lightweight at import time and
    to avoid cyclic imports between the registry and individual
    analyzers.
    """
    from src.analysis.rhetorical_device_detector import RhetoricalDeviceDetector
    from src.analysis.argument_mining import ArgumentMiningAnalyzer
    from src.analysis.context_omission_detector import ContextOmissionDetector
    from src.analysis.discourse_coherence_analyzer import (
        DiscourseCoherenceAnalyzer,
    )
    from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
    from src.analysis.framing_analysis import FramingAnalyzer
    from src.analysis.information_density_analyzer import (
        InformationDensityAnalyzer,
    )
    from src.analysis.information_omission_detector import (
        InformationOmissionDetector,
    )
    from src.analysis.ideological_language_detector import (
        IdeologicalLanguageDetector,
    )
    from src.analysis.narrative_role_extractor import NarrativeRoleExtractor
    from src.analysis.narrative_conflict import NarrativeConflictAnalyzer
    from src.analysis.narrative_propagation import (
        NarrativePropagationAnalyzer,
    )
    from src.analysis.narrative_temporal_analyzer import (
        NarrativeTemporalAnalyzer,
    )
    from src.analysis.source_attribution_analyzer import (
        SourceAttributionAnalyzer,
    )

    reg = AnalyzerRegistry()

    reg.register("rhetorical", RhetoricalDeviceDetector(), order=1)
    reg.register("argument", ArgumentMiningAnalyzer(), order=2)
    reg.register("context", ContextOmissionDetector(), order=3)
    reg.register("discourse", DiscourseCoherenceAnalyzer(), order=4)
    reg.register("emotion", EmotionTargetAnalyzer(), order=5)
    reg.register("framing", FramingAnalyzer(), order=6)
    reg.register("information", InformationDensityAnalyzer(), order=7)
    reg.register(
        "information_omission", InformationOmissionDetector(), order=8
    )
    reg.register("ideology", IdeologicalLanguageDetector(), order=9)
    reg.register("narrative_role", NarrativeRoleExtractor(), order=10)
    reg.register("narrative_conflict", NarrativeConflictAnalyzer(), order=11)
    reg.register(
        "narrative_propagation", NarrativePropagationAnalyzer(), order=12
    )
    reg.register("narrative_temporal", NarrativeTemporalAnalyzer(), order=13)
    reg.register("source", SourceAttributionAnalyzer(), order=14)

    return reg