"""
File Name: result_formatter.py
Module: Result Formatting
Description:
    Converts internal prediction and analysis outputs from the TruthLens
    inference pipeline into multiple standardized external formats suitable
    for different consumers.

    Supported formats:
        • API JSON responses
        • Dashboard JSON structures
        • Research export JSON

    This module ensures that internal representations remain decoupled from
    presentation layers while providing deterministic and validated output
    schemas.

    Example output targets:
        - TruthLensAPIResponse
        - TruthLensDashboardReport
        - TruthLensResearchExport

Dependencies:
    logging
    typing
    dataclasses
    json
    datetime

Inputs:
    Internal prediction and analysis dictionaries produced by the
    inference and report generation pipelines.

Outputs:
    Structured dictionaries compatible with APIs, dashboards,
    and research exports.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TruthLensAPIResponse:
    """
    API response structure used by production endpoints.
    """
    bias: Optional[str]
    ideology: Optional[str]
    propaganda_probability: Optional[float]
    credibility_score: Optional[float]
    timestamp: str


@dataclass
class TruthLensDashboardReport:
    """
    Dashboard-oriented report structure including expanded analysis
    for visualization and monitoring interfaces.
    """
    article_summary: Dict[str, Any]
    bias_analysis: Dict[str, Any]
    emotion_analysis: Dict[str, Any]
    narrative_structure: Dict[str, Any]
    entity_graph: Dict[str, Any]
    credibility_score: Optional[float]
    generated_at: str


@dataclass
class TruthLensResearchExport:
    """
    Research export format containing detailed signals and metadata
    used for experimentation and analysis.
    """
    article_summary: Dict[str, Any]
    predictions: Dict[str, Any]
    intermediate_features: Optional[Dict[str, Any]]
    model_metadata: Optional[Dict[str, Any]]
    generated_at: str


class ResultFormatter:
    """
    Responsible for converting internal system outputs into standardized
    formats for APIs, dashboards, and research workflows.
    """

    def __init__(self) -> None:
        logger.info("ResultFormatter initialized")

    def _timestamp(self) -> str:
        """
        Generate ISO timestamp.
        """
        return datetime.utcnow().isoformat()

    def format_api_response(
        self,
        prediction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert internal prediction output to API response format.
        """

        if not isinstance(prediction, dict):
            raise TypeError("prediction must be a dictionary")

        response = TruthLensAPIResponse(
            bias=prediction.get("bias"),
            ideology=prediction.get("ideology"),
            propaganda_probability=prediction.get("propaganda_probability"),
            credibility_score=prediction.get("credibility_score"),
            timestamp=self._timestamp(),
        )

        logger.debug("Formatted API response")

        return asdict(response)

    def format_dashboard_report(
        self,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert full report into dashboard-ready structure.
        """

        if not isinstance(report, dict):
            raise TypeError("report must be a dictionary")

        dashboard_report = TruthLensDashboardReport(
            article_summary=report.get("article_summary", {}),
            bias_analysis=report.get("bias_analysis", {}),
            emotion_analysis=report.get("emotion_analysis", {}),
            narrative_structure=report.get("narrative_structure", {}),
            entity_graph=report.get("entity_graph", {}),
            credibility_score=report.get("credibility_score"),
            generated_at=self._timestamp(),
        )

        logger.debug("Formatted dashboard report")

        return asdict(dashboard_report)

    def format_research_export(
        self,
        report: Dict[str, Any],
        prediction: Dict[str, Any],
        features: Optional[Dict[str, Any]] = None,
        model_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert system outputs into research-friendly export structure.
        """

        if not isinstance(report, dict):
            raise TypeError("report must be a dictionary")

        if not isinstance(prediction, dict):
            raise TypeError("prediction must be a dictionary")

        export = TruthLensResearchExport(
            article_summary=report.get("article_summary", {}),
            predictions=prediction,
            intermediate_features=features,
            model_metadata=model_metadata,
            generated_at=self._timestamp(),
        )

        logger.debug("Formatted research export")

        return asdict(export)

    def to_json(
        self,
        data: Dict[str, Any],
        pretty: bool = True,
    ) -> str:
        """
        Serialize formatted output to JSON.
        """

        try:
            if pretty:
                return json.dumps(data, indent=4, ensure_ascii=False)
            return json.dumps(data)

        except Exception as exc:
            logger.exception("JSON serialization failed")
            raise RuntimeError("Failed to serialize output") from exc