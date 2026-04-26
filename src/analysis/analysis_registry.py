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
                # dependency injection
                kwargs = {
                    dep: results[dep].output
                    for dep in spec.requires
                }

                kwargs.update(extra_inputs)

                output = spec.analyzer.analyze(ctx, **kwargs)

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