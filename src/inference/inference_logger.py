"""
File Name: inference_logger.py
Module: Inference Structured Logging
Description:
    Provides structured logging utilities for the TruthLens inference system.
    This module enables production-grade logging for monitoring, debugging,
    and auditing model predictions at scale.

    Logged attributes include:
        • article_id
        • processing_time
        • model_versions
        • feature_count
        • prediction_confidence

    Logs are emitted in structured JSON format to facilitate ingestion by
    monitoring systems such as ELK, Datadog, Prometheus pipelines, or
    cloud logging infrastructure.
    
Dependencies:
    logging
    typing
    dataclasses
    json
    time
    uuid

Inputs:
    Inference metadata and prediction outputs.

Outputs:
    Structured JSON log entries.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from src.utils import current_datetime


logger = logging.getLogger(__name__)


@dataclass
class InferenceLogEntry:
    """
    Dataclass representing a single inference log record.
    """

    article_id: str
    processing_time_ms: float
    model_versions: Dict[str, str]
    feature_count: int
    prediction_confidence: Optional[float]
    timestamp: float


class InferenceLogger:
    """
    Production-grade structured logger for inference events.
    """

    def __init__(
        self,
        service_name: str = "truthlens-inference",
        enable_json_logs: bool = True,
    ) -> None:
        self.service_name = service_name
        self.enable_json_logs = enable_json_logs

        logger.info("InferenceLogger initialized for service: %s", service_name)

    def generate_article_id(self) -> str:
        """
        Generate a unique article identifier when not provided.
        """
        return str(uuid.uuid4())

    def start_timer(self) -> float:
        """
        Start processing timer.
        """
        return time.perf_counter()

    def stop_timer(self, start_time: float) -> float:
        """
        Stop timer and compute elapsed time in milliseconds.
        """
        elapsed = (time.perf_counter() - start_time) * 1000
        return elapsed

    def create_log_entry(
        self,
        article_id: str,
        processing_time_ms: float,
        model_versions: Dict[str, str],
        feature_count: int,
        prediction_confidence: Optional[float],
    ) -> InferenceLogEntry:
        """
        Create structured log entry.
        """

        if not isinstance(article_id, str):
            raise TypeError("article_id must be a string")

        if not isinstance(model_versions, dict):
            raise TypeError("model_versions must be a dictionary")

        if feature_count < 0:
            raise ValueError("feature_count cannot be negative")

        entry = InferenceLogEntry(
            article_id=article_id,
            processing_time_ms=processing_time_ms,
            model_versions=model_versions,
            feature_count=feature_count,
            prediction_confidence=prediction_confidence,
            timestamp=float(current_datetime().timestamp()),
        )

        return entry

    def log(
        self,
        entry: InferenceLogEntry,
        level: int = logging.INFO,
    ) -> None:
        """
        Emit structured log entry.
        """

        record = {
            "service": self.service_name,
            "event": "inference",
            **asdict(entry),
        }

        try:
            if self.enable_json_logs:
                message = json.dumps(record)
            else:
                message = str(record)

            logger.log(level, message)

        except Exception as exc:
            logger.exception("Failed to emit inference log: %s", exc)

    def log_prediction(
        self,
        article_id: Optional[str],
        start_time: float,
        model_versions: Dict[str, str],
        feature_count: int,
        prediction_confidence: Optional[float],
    ) -> None:
        """
        Convenience method to log a completed inference event.
        """

        if article_id is None:
            article_id = self.generate_article_id()

        processing_time_ms = self.stop_timer(start_time)

        entry = self.create_log_entry(
            article_id=article_id,
            processing_time_ms=processing_time_ms,
            model_versions=model_versions,
            feature_count=feature_count,
            prediction_confidence=prediction_confidence,
        )

        self.log(entry)
