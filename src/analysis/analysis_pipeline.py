# src/analysis/analysis_pipeline.py

from __future__ import annotations

import logging
from typing import Dict, List, Any

from spacy.tokens import Doc

from src.analysis.spacy_loader import get_shared_nlp
from src.analysis.feature_context import FeatureContext
from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_merger import FeatureMerger
from src.analysis.analysis_config import AnalysisConfig, build_default_config

logger = logging.getLogger(__name__)


# =========================================================
# Pipeline
# =========================================================

class AnalysisPipeline:

    def __init__(
        self,
        analyzers: List[BaseAnalyzer],
        *,
        config: AnalysisConfig | None = None,
        nlp_mode: str = "safe",
    ):
        if not analyzers:
            raise ValueError("At least one analyzer required")

        self.config = config or build_default_config()
        self.analyzers = analyzers
        self.nlp = get_shared_nlp(mode=nlp_mode)

        self.merger = FeatureMerger()

        logger.info(
            "Pipeline initialized | analyzers=%d | mode=%s",
            len(analyzers),
            nlp_mode,
        )

    # =====================================================
    # SINGLE
    # =====================================================

    def run(self, text: str) -> Dict[str, Any]:

        text = self._validate(text)

        doc = self.nlp(text)
        ctx = FeatureContext.from_doc(doc)

        sections = self._run_analyzers(ctx)

        merged = self.merger.merge(sections)
        vector, keys = self.merger.to_vector(sections)

        return {
            "sections": sections,
            "features": merged,
            "vector": vector,
            "feature_keys": keys,
        }

    # =====================================================
    # BATCH
    # =====================================================

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:

        if not texts:
            return []

        docs = list(self.nlp.pipe(
            texts,
            batch_size=self.config.pipeline.batch_size,
        ))

        results = []

        for doc in docs:
            ctx = FeatureContext.from_doc(doc)
            sections = self._run_analyzers(ctx)

            merged = self.merger.merge(sections)
            vector, keys = self.merger.to_vector(sections)

            results.append({
                "sections": sections,
                "features": merged,
                "vector": vector,
                "feature_keys": keys,
            })

        return results

    # =====================================================
    # ANALYZERS
    # =====================================================

    def _run_analyzers(
        self,
        ctx: FeatureContext,
    ) -> Dict[str, Dict[str, float]]:

        result: Dict[str, Dict[str, float]] = {}

        for analyzer in self.analyzers:

            name = analyzer.__class__.__name__

            if not self._is_enabled(name):
                continue

            try:
                output = analyzer.analyze(ctx)

                if not isinstance(output, dict):
                    continue

                group = self._get_group(analyzer)

                result.setdefault(group, {}).update(output)

            except Exception:
                logger.exception("Analyzer failed: %s", name)

                if self.config.global_config.fail_fast:
                    raise

        return result

    # =====================================================
    # GROUP RESOLUTION (FIXED)
    # =====================================================

    def _get_group(self, analyzer: BaseAnalyzer) -> str:
        """
        Each analyzer MUST define `group` attribute.
        """

        if hasattr(analyzer, "group"):
            return analyzer.group

        raise ValueError(
            f"{analyzer.__class__.__name__} missing 'group' attribute"
        )

    # =====================================================
    # CONFIG HELPERS
    # =====================================================

    def _is_enabled(self, name: str) -> bool:
        return self.config.is_enabled(name.lower())

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