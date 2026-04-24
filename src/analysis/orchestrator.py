# src/analysis/orchestrator.py

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Optional, List

from src.analysis.analysis_pipeline import AnalysisPipeline
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.analysis.propaganda_pattern_detector import PropagandaPatternDetector

#  FIXED IMPORT PATH
try:
    from src.analysis.output_models import (
        PipelineOutput,
        FullAnalysisOutput,
        PropagandaFeatures,
        BiasProfile,
    )
    USE_MODELS = True
except Exception:
    USE_MODELS = False


logger = logging.getLogger(__name__)


# =========================================================
# Orchestrator
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

        logger.info("AnalysisOrchestrator initialized")

    # =====================================================
    # SINGLE TEXT
    # =====================================================

    def run(self, text: str) -> Dict[str, Any]:

        try:
            if not isinstance(text, str):
                raise ValueError("Input must be string")

            text = text.strip()
            if not text:
                raise ValueError("Input text cannot be empty")

            start_time = time.time()

            raw_results = self.pipeline.run(text)

            if hasattr(raw_results, "model_dump"):
                raw_results = raw_results.model_dump()

            result = self._post_process(raw_results)

            if self.enable_timing:
                result["meta"]["latency_ms"] = round(
                    (time.time() - start_time) * 1000, 2
                )

            return result

        except Exception:
            logger.exception("Orchestrator run failed")
            return self._error_response()

    # =====================================================
    # 🔥 BATCH SUPPORT (IMPORTANT)
    # =====================================================

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:

        try:
            raw_batch = self.pipeline.run_batch(texts)

            results = []
            for raw in raw_batch:
                results.append(self._post_process(raw))

            return results

        except Exception:
            logger.exception("Batch processing failed")

            # fallback to safe processing
            return [self.run(t) for t in texts]

    # =====================================================
    # CORE POST PROCESS
    # =====================================================

    def _post_process(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:

        # Safe extraction
        rhetorical = raw_results.get("rhetorical", {})
        argument = raw_results.get("argument", {})
        context = raw_results.get("context", {})
        discourse = raw_results.get("discourse", {})
        emotion = raw_results.get("emotion", {})
        framing = raw_results.get("framing", {})
        information = raw_results.get("information", {})
        ideology = raw_results.get("ideology", {})

        # -----------------------------
        # Profile
        # -----------------------------
        profile = self.builder.build_profile(
            bias=framing,
            emotion=emotion,
            narrative=framing,
            discourse=discourse,
            ideology=ideology,
        )

        # -----------------------------
        # Propaganda
        # -----------------------------
        propaganda = self.propaganda.analyze(
            emotion_features=emotion,
            narrative_features=framing,
            rhetorical_features=rhetorical,
            argument_features=argument,
            information_features=information,
        )

        # Convert if needed
        profile_dict = profile.model_dump() if hasattr(profile, "model_dump") else profile
        propaganda_dict = (
            propaganda.model_dump()
            if hasattr(propaganda, "model_dump")
            else propaganda
        )

        meta = {"text_length": len(str(raw_results))}

        # -----------------------------
        # Models (optional)
        # -----------------------------
        if USE_MODELS:
            try:
                return FullAnalysisOutput(
                    features=PipelineOutput(**raw_results),
                    profile=profile if isinstance(profile, BiasProfile) else profile_dict,
                    propaganda=PropagandaFeatures(**propaganda_dict),
                    meta=meta,
                )
            except Exception as e:
                logger.warning("Model conversion failed: %s", e)

        return {
            "features": raw_results,
            "profile": profile_dict,
            "propaganda": propaganda_dict,
            "meta": meta,
        }

    # =====================================================
    # ERROR RESPONSE
    # =====================================================

    def _error_response(self) -> Dict[str, Any]:
        return {
            "features": {},
            "profile": {},
            "propaganda": {},
            "meta": {"error": True},
        }