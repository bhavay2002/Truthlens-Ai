from __future__ import annotations

import logging
from typing import List, Dict, Any

from spacy.tokens import Doc

from src.analysis._nlp import get_shared_nlp, process_docs
from src.analysis.feature_context import FeatureContext
from src.analysis.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


# =========================================================
# Pipeline
# =========================================================

class AnalysisPipeline:
    """
    Core feature extraction pipeline.

    Responsibilities:
    - Single spaCy pass
    - Build FeatureContext
    - Execute analyzers
    - Return structured feature groups
    """

    def __init__(
        self,
        analyzers: List[BaseAnalyzer],
        *,
        nlp_mode: str = "safe",  # "safe" | "fast"
    ):
        if not analyzers:
            raise ValueError("At least one analyzer required")

        self.analyzers = analyzers
        self.nlp = get_shared_nlp(mode=nlp_mode)

        logger.info(
            "AnalysisPipeline initialized | analyzers=%d | mode=%s",
            len(analyzers),
            nlp_mode,
        )

    # =====================================================
    # SINGLE TEXT
    # =====================================================

    def run(self, text: str) -> Dict[str, Dict[str, float]]:

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input must be non-empty string")

        doc = self.nlp(text)
        ctx = FeatureContext.from_doc(doc)

        return self._run_analyzers(ctx)

    # =====================================================
    # BATCH
    # =====================================================

    def run_batch(self, texts: List[str]) -> List[Dict[str, Dict[str, float]]]:

        if not texts:
            return []

        docs = process_docs(texts)

        results = []

        for doc in docs:
            ctx = FeatureContext.from_doc(doc)
            results.append(self._run_analyzers(ctx))

        return results

    # =====================================================
    # INTERNAL EXECUTION
    # =====================================================

    def _run_analyzers(self, ctx: FeatureContext) -> Dict[str, Dict[str, float]]:

        result = {
            "rhetorical": {},
            "argument": {},
            "context": {},
            "discourse": {},
            "emotion": {},
            "framing": {},
            "information": {},
            "ideology": {},
            "source": {},
            "narrative": {},
        }

        for analyzer in self.analyzers:

            try:
                output = analyzer.analyze(ctx)

                if not isinstance(output, dict):
                    continue

                self._route_output(analyzer, output, result)

            except Exception:
                logger.exception(
                    "Analyzer failed: %s", analyzer.__class__.__name__
                )

        return result

    # =====================================================
    # ROUTING LOGIC (CRITICAL)
    # =====================================================

    def _route_output(
        self,
        analyzer: BaseAnalyzer,
        output: Dict[str, float],
        result: Dict[str, Dict[str, float]],
    ) -> None:

        name = analyzer.__class__.__name__.lower()

        # 🔥 deterministic routing (NO heuristics later)

        if "rhetorical" in name:
            result["rhetorical"].update(output)

        elif "argument" in name:
            result["argument"].update(output)

        elif "context" in name:
            result["context"].update(output)

        elif "discourse" in name:
            result["discourse"].update(output)

        elif "emotion" in name:
            result["emotion"].update(output)

        elif "framing" in name:
            result["framing"].update(output)

        elif "informationdensity" in name or "information_density" in name:
            result["information"].update(output)

        elif "ideological" in name:
            result["ideology"].update(output)

        elif "source" in name:
            result["source"].update(output)

        elif "narrative" in name:
            result["narrative"].update(output)

        else:
            # fallback (rare)
            logger.debug("Unrouted analyzer output: %s", name)