"""
File: prediction_service.py
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np

from src.inference.inference_engine import InferenceEngine
from src.inference.inference_logger import InferenceLogger
from src.inference.inference_cache import InferenceCache, InferenceCacheConfig
from src.inference.report_generator import ReportGenerator
from src.inference.result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


# =========================================================
# SERVICE
# =========================================================

class PredictionService:

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        cache: Optional[InferenceCache] = None,
        logger_: Optional[InferenceLogger] = None,
        report_generator: Optional[ReportGenerator] = None,
        formatter: Optional[ResultFormatter] = None,
    ):

        self.engine = engine

        self.cache = cache or InferenceCache(
            InferenceCacheConfig(enable_memory_cache=True)
        )

        self.logger = logger_ or InferenceLogger()
        self.report_generator = report_generator or ReportGenerator()
        self.formatter = formatter or ResultFormatter()

        logger.info("PredictionService initialized")

    # =====================================================
    # CORE PREDICT
    # =====================================================

    def predict(
        self,
        text: str,
        *,
        use_cache: bool = True,
    ) -> Dict[str, Any]:

        start = self.logger.start_timer()

        # ---------------- CACHE ----------------
        if use_cache:
            cached = self.cache.get(text)
            if cached:
                return cached

        # ---------------- INFERENCE ----------------
        outputs = self.engine.predict_for_evaluation([text])

        # ---------------- POSTPROCESS ----------------
        preds = self._postprocess(outputs)

        # ---------------- LOG ----------------
        self.logger.log_prediction(
            start_time=start,
            model_versions={},
            feature_count=0,
            predicted_label=preds.get("label"),
            prediction_confidence=preds.get("confidence"),
        )

        # ---------------- CACHE SAVE ----------------
        if use_cache:
            self.cache.set(text, preds)

        return preds

    # =====================================================
    # BATCH
    # =====================================================

    def predict_batch(
        self,
        texts: List[str],
    ) -> List[Dict[str, Any]]:

        outputs = self.engine.predict_for_evaluation(texts)

        probs_arr = outputs.get("calibrated_probabilities") or outputs.get("probabilities")
        preds_arr = outputs.get("predictions")

        results = []

        for i, text in enumerate(texts):

            conf = float(np.max(probs_arr[i])) if probs_arr is not None else None
            fake_prob = float(probs_arr[i][1]) if (probs_arr is not None and probs_arr.shape[1] > 1) else None

            results.append({
                "label": int(preds_arr[i]) if preds_arr is not None else None,
                "confidence": conf,
                "fake_probability": fake_prob,
            })

        return results

    # =====================================================
    # FULL PIPELINE
    # =====================================================

    def predict_full(
        self,
        text: str,
    ) -> Dict[str, Any]:

        outputs = self.engine.predict_for_evaluation([text])

        # ---------------- UNCERTAINTY ----------------
        uncertainty = self._compute_uncertainty(outputs)

        # ---------------- REPORT ----------------
        report = self.report_generator.generate_report(
            article_text=text,
            predictions=outputs,
            uncertainty=uncertainty,
        )

        return report

    # =====================================================
    # FORMATTED OUTPUT
    # =====================================================

    def predict_formatted(
        self,
        text: str,
        *,
        mode: str = "api",
    ) -> Dict[str, Any]:

        report = self.predict_full(text)

        if mode == "api":
            return self.formatter.format_api_response(report)

        elif mode == "dashboard":
            return self.formatter.format_dashboard_report(report)

        elif mode == "research":
            return self.formatter.format_research_export(report)

        else:
            raise ValueError("Invalid mode")

    # =====================================================
    # EVALUATION MODE
    # =====================================================

    def predict_for_evaluation(
        self,
        texts: List[str],
    ) -> Dict[str, Any]:

        return self.engine.predict_for_evaluation(texts)

    # =====================================================
    # POSTPROCESS
    # =====================================================

    def _postprocess(self, outputs):

        probs_arr = outputs.get("calibrated_probabilities") or outputs.get("probabilities")
        preds_arr = outputs.get("predictions")

        conf = float(np.max(probs_arr[0])) if probs_arr is not None else None
        fake_prob = float(probs_arr[0][1]) if (probs_arr is not None and probs_arr.shape[1] > 1) else None

        return {
            "label": int(preds_arr[0]) if preds_arr is not None else None,
            "confidence": conf,
            "fake_probability": fake_prob,
        }

    # =====================================================
    # UNCERTAINTY
    # =====================================================

    def _compute_uncertainty(self, outputs):

        results = {}

        for task, out in outputs.items():

            probs = out["probabilities"]

            if probs is None:
                continue

            probs = np.asarray(probs)

            entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

            results[task] = {
                "mean_entropy": float(np.mean(entropy)),
                "p95_entropy": float(np.percentile(entropy, 95)),
            }

        return results