#src\analysis\orchestrator.py

from __future__ import annotations

import logging
import time 
from typing import Dict, Any, Optional, List

from src.analysis.analysis_pipeline import AnalysisPipeline
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.analysis.propaganda_pattern_detector import PropagandaPatternDetector

logger = logging.getLogger(__name__)


# =========================================================
# ORCHESTRATOR
# =========================================================

class AnalysisOrchestrator:

    def __init__(
        self,
        pipeline: AnalysisPipeline,
        builder: Optional[BiasProfileBuilder] = None,
        propaganda_detector: Optional[PropagandaPatternDetector] = None,
        enable_timing: bool = True,
    ):
        self.pipeline = pipeline
        self.builder = builder or BiasProfileBuilder()
        self.propaganda = propaganda_detector or PropagandaPatternDetector()
        self.enable_timing = enable_timing

        logger.info("AnalysisOrchestrator initialized (final)")

    # =====================================================
    # SINGLE
    # =====================================================

    def run(self, text: str) -> Dict[str, Any]:

        start_total = time.perf_counter()

        try:
            text = self._validate_input(text)

            # -------------------------
            # PIPELINE
            # -------------------------
            t0 = time.perf_counter()
            raw = self.pipeline.run(text)
            t_pipeline = time.perf_counter() - t0

            if hasattr(raw, "model_dump"):
                raw = raw.model_dump()

            # -------------------------
            # POST PROCESS
            # -------------------------
            t1 = time.perf_counter()
            result = self._post_process(raw, text)
            t_post = time.perf_counter() - t1

            # -------------------------
            # META (TIMING)
            # -------------------------
            if self.enable_timing:
                result.setdefault("meta", {})
                result["meta"]["timing"] = {
                    "pipeline_ms": round(t_pipeline * 1000, 2),
                    "postprocess_ms": round(t_post * 1000, 2),
                    "total_ms": round((time.perf_counter() - start_total) * 1000, 2),
                }

            return result

        except Exception:
            logger.exception("Orchestrator run failed")
            return self._error_response()

    # =====================================================
    # BATCH
    # =====================================================

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:

        if not texts:
            return []

        start_total = time.perf_counter()

        try:
            indexed = list(enumerate(texts))

            t0 = time.perf_counter()
            raw_batch = self.pipeline.run_batch([t for _, t in indexed])
            t_pipeline = time.perf_counter() - t0

            results: List[Dict[str, Any]] = []

            for (idx, text), raw in zip(indexed, raw_batch):
                try:
                    if hasattr(raw, "model_dump"):
                        raw = raw.model_dump()

                    result = self._post_process(raw, text)

                    result.setdefault("meta", {})
                    result["meta"]["index"] = idx

                    results.append(result)

                except Exception:
                    logger.exception("Post-process failed")
                    results.append(self._error_response())

            # -------------------------
            # GLOBAL TIMING
            # -------------------------
            if self.enable_timing:
                total_time = (time.perf_counter() - start_total) * 1000

                for r in results:
                    r.setdefault("meta", {})
                    r["meta"]["batch_timing"] = {
                        "pipeline_total_ms": round(t_pipeline * 1000, 2),
                        "batch_total_ms": round(total_time, 2),
                    }

            return results

        except Exception:
            logger.exception("Batch failed → fallback to sequential")
            return [self.run(t) for t in texts]

    # =====================================================
    # POST PROCESS
    # =====================================================

    def _post_process(self, raw: Dict[str, Any], text: str) -> Dict[str, Any]:

        # -------------------------
        # SAFE EXTRACTION
        # -------------------------
        sections = {
            "rhetorical": raw.get("rhetorical", {}),
            "argument": raw.get("argument", {}),
            "context": raw.get("context", {}),
            "discourse": raw.get("discourse", {}),
            "emotion": raw.get("emotion", {}),
            "framing": raw.get("framing", {}),
            "information": raw.get("information", {}),
            "ideology": raw.get("ideology", {}),
        }

        # -------------------------
        # PROFILE
        # -------------------------
        profile = self.builder.build_profile(
            bias=sections["framing"],
            emotion=sections["emotion"],
            narrative=sections["framing"],
            discourse=sections["discourse"],
            ideology=sections["ideology"],
        )

        # -------------------------
        # PROPAGANDA
        # -------------------------
        propaganda = self.propaganda.analyze(
            emotion_features=sections["emotion"],
            narrative_features=sections["framing"],
            rhetorical_features=sections["rhetorical"],
            argument_features=sections["argument"],
            information_features=sections["information"],
        )

        # -------------------------
        # CONFIDENCE
        # -------------------------
        confidence = self._confidence(sections)

        # -------------------------
        # META
        # -------------------------
        meta = {
            "input_length": len(text),
            "num_features": sum(len(v) for v in sections.values()),
            "confidence": confidence,
        }

        return {
            "features": raw,
            "profile": profile,
            "propaganda": propaganda,
            "meta": meta,
        }

    # =====================================================
    # CONFIDENCE
    # =====================================================

    def _confidence(self, sections: Dict[str, Dict[str, float]]) -> float:

        values = []

        for section in sections.values():
            values.extend(
                v for v in section.values() if isinstance(v, (int, float))
            )

        if not values:
            return 0.0

        # robust signal strength
        mean_val = sum(values) / len(values)

        return float(min(max(mean_val, 0.0), 1.0))

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_input(self, text: Any) -> str:

        if not isinstance(text, str):
            raise ValueError("Input must be string")

        text = text.strip()

        if not text:
            raise ValueError("Empty text")

        return text

    # =====================================================
    # ERROR
    # =====================================================

    def _error_response(self) -> Dict[str, Any]:

        return {
            "features": {},
            "profile": {},
            "propaganda": {},
            "meta": {
                "error": True,
                "confidence": 0.0,
            },
        }