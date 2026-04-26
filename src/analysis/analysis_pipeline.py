from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional

from spacy.tokens import Doc

from src.analysis.spacy_loader import get_shared_nlp
from src.analysis.feature_context import FeatureContext
from src.analysis.feature_merger import FeatureMerger
from src.analysis.analysis_config import AnalysisConfig, build_default_config
from src.analysis.analysis_registry import AnalyzerRegistry, AnalyzerExecution

logger = logging.getLogger(__name__)


# =========================================================
# PIPELINE (UPGRADED)
# =========================================================

class AnalysisPipeline:
    """
    Production-grade analysis pipeline.

    Features:
    - registry-driven execution
    - structured outputs
    - latency tracking
    - batch optimization
    - fail-safe execution
    """

    def __init__(
        self,
        registry: AnalyzerRegistry,
        *,
        config: Optional[AnalysisConfig] = None,
        nlp_mode: str = "safe",
    ):
        self.config = config or build_default_config()
        self.registry = registry

        self.nlp = get_shared_nlp(mode=nlp_mode)
        self.merger = FeatureMerger()

        logger.info(
            "AnalysisPipeline initialized | analyzers=%d | mode=%s",
            len(self.registry.list()),
            nlp_mode,
        )

    # =====================================================
    # SINGLE RUN
    # =====================================================

    def run(self, text: str) -> Dict[str, Any]:

        start = time.time()

        text = self._validate(text)

        doc = self.nlp(text)
        ctx = FeatureContext.from_doc(doc)

        results = self._execute(ctx)

        merged, vector, keys = self._post_process(results)

        return {
            "sections": {k: v.output for k, v in results.items()},
            "features": merged,
            "vector": vector,
            "feature_keys": keys,
            "meta": self._build_meta(results, start),
        }

    # =====================================================
    # BATCH RUN (OPTIMIZED)
    # =====================================================

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:

        if not texts:
            return []

        start = time.time()

        texts = [self._validate(t) for t in texts]

        docs = list(
            self.nlp.pipe(
                texts,
                batch_size=self.config.pipeline.batch_size,
                n_process=1,  # keep deterministic; can scale later
            )
        )

        results = []

        for doc in docs:
            ctx = FeatureContext.from_doc(doc)

            exec_results = self._execute(ctx)
            merged, vector, keys = self._post_process(exec_results)

            results.append({
                "sections": {k: v.output for k, v in exec_results.items()},
                "features": merged,
                "vector": vector,
                "feature_keys": keys,
                "meta": self._build_meta(exec_results, start),
            })

        return results

    # =====================================================
    # EXECUTION ENGINE (CORE)
    # =====================================================

    def _execute(
        self,
        ctx: FeatureContext,
    ) -> Dict[str, AnalyzerExecution]:

        return self.registry.run_all(
            ctx,
            extra_inputs=self._extra_inputs(ctx),
        )

    # =====================================================
    # POST PROCESSING
    # =====================================================

    def _post_process(
        self,
        results: Dict[str, AnalyzerExecution],
    ):

        sections = {k: v.output for k, v in results.items()}

        merged = self.merger.merge(sections)
        vector, keys = self.merger.to_vector(sections)

        return merged, vector, keys

    # =====================================================
    # META / OBSERVABILITY
    # =====================================================

    def _build_meta(
        self,
        results: Dict[str, AnalyzerExecution],
        start_time: float,
    ) -> Dict[str, Any]:

        total_time = time.time() - start_time

        failures = [
            k for k, v in results.items() if not v.success
        ]

        latencies = {
            k: v.latency for k, v in results.items()
        }

        return {
            "total_latency": total_time,
            "analyzer_latency": latencies,
            "failed_analyzers": failures,
            "num_analyzers": len(results),
        }

    # =====================================================
    # EXTRA INPUTS (EXTENSIBILITY)
    # =====================================================

    def _extra_inputs(self, ctx: FeatureContext) -> Dict[str, Any]:
        """
        Hook for passing extra shared inputs to analyzers.
        """
        return {}

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate(self, text: Any) -> str:

        if not isinstance(text, str):
            raise ValueError("Input must be string")

        text = text.strip()

        if not text:
            raise ValueError("Empty text")

        if len(text) > self.config.global_config.max_text_length:
            if self.config.global_config.truncate_text:
                text = text[: self.config.global_config.max_text_length]
            else:
                raise ValueError("Text too long")

        return text