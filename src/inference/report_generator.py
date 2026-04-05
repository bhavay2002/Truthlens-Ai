"""
File Name: report_generator.py
Module: Report Generation
Description:
    Generates structured and human-readable reports from the outputs of the
    TruthLens analysis and prediction pipelines.

    The module aggregates outputs from multiple subsystems including bias
    detection, emotion analysis, narrative structure analysis, entity graph
    analysis, and credibility scoring. The resulting report can be used by:

        • APIs
        • dashboards
        • explainability systems
        • monitoring pipelines
        • PDF / HTML reporting tools

    Integrates ExplanationReportGenerator to persist explanation artifacts
    (JSON + HTML) whenever explainability data is present in the inference
    output. The 'explainability' section of the report mirrors the structured
    output from ExplainabilityLayer, including aggregated token importance,
    attention rollout scores, and cross-method consistency metrics.

Dependencies:
    logging
    typing
    dataclasses
    json
    datetime
    src.explainability.explanation_report_generator

Inputs:
    Prediction results and analysis outputs from inference pipelines.

Outputs:
    Structured human-readable report dictionary or JSON.
    Optional HTML + JSON explanation artifacts on disk.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.explainability.explanation_report_generator import ExplanationReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class ArticleSummary:
    """
    Summary information for analyzed article.
    """
    title: Optional[str]
    source: Optional[str]
    word_count: Optional[int]
    analyzed_at: str


@dataclass
class ReportConfig:
    """
    Configuration for report generation.
    """
    include_timestamp: bool = True
    pretty_json: bool = True
    validate_fields: bool = True
    save_explanation_artifacts: bool = False
    explanation_output_dir: str = "reports/explanations"


class ReportGenerator:
    """
    Generates structured analysis reports combining outputs from multiple
    analysis modules.

    Responsibilities:
        • Aggregate subsystem outputs
        • Validate report schema
        • Produce consistent structured output
        • Support JSON export
        • Persist HTML + JSON explanation artifacts via ExplanationReportGenerator
    """

    def __init__(self, config: Optional[ReportConfig] = None) -> None:
        self.config = config or ReportConfig()
        self._explanation_reporter = ExplanationReportGenerator(
            output_dir=self.config.explanation_output_dir
        )
        logger.info("ReportGenerator initialized")

    def _current_timestamp(self) -> str:
        """
        Generate ISO formatted timestamp.
        """
        return datetime.now(timezone.utc).isoformat()

    def _validate_section(self, section: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
        """
        Validate section structure.
        """
        if section is None:
            logger.debug("Section '%s' missing, substituting empty object", name)
            return {}

        if not isinstance(section, dict):
            raise TypeError(f"{name} must be a dictionary")

        return section

    def save_explanation_artifacts(
        self,
        article_id: str,
        explainability: Dict[str, Any],
        save_json: bool = True,
        save_html: bool = True,
    ) -> Dict[str, Path]:
        """
        Persist explainability data as JSON and HTML artifacts using
        ExplanationReportGenerator.

        Parameters
        ----------
        article_id : str
            Unique identifier for the article (used as filename prefix).
        explainability : dict
            Explainability output from ExplainabilityLayer.explain().
        save_json : bool
            Whether to write a JSON artifact.
        save_html : bool
            Whether to write an HTML dashboard artifact.

        Returns
        -------
        Dict mapping 'json' and/or 'html' to their file Paths.
        """
        try:
            paths = self._explanation_reporter.generate(
                article_id=article_id,
                explanation=explainability,
                save_json=save_json,
                save_html=save_html,
            )
            logger.info(
                "Explanation artifacts saved for article_id=%s: %s",
                article_id,
                list(paths.keys()),
            )
            return paths
        except Exception as exc:
            logger.warning("Failed to save explanation artifacts: %s", exc)
            return {}

    def generate_report(
        self,
        article_text: str,
        title: Optional[str] = None,
        source: Optional[str] = None,
        bias_analysis: Optional[Dict[str, Any]] = None,
        emotion_analysis: Optional[Dict[str, Any]] = None,
        narrative_structure: Optional[Dict[str, Any]] = None,
        entity_graph: Optional[Dict[str, Any]] = None,
        credibility_score: Optional[float] = None,
        explainability: Optional[Dict[str, Any]] = None,
        article_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate full analysis report.

        When explainability data is provided and save_explanation_artifacts
        is enabled in the config, HTML and JSON explanation artifacts are
        written to disk automatically.
        """

        if not isinstance(article_text, str):
            raise TypeError("article_text must be a string")

        word_count = len(article_text.split())

        summary = ArticleSummary(
            title=title,
            source=source,
            word_count=word_count,
            analyzed_at=self._current_timestamp() if self.config.include_timestamp else "",
        )

        report: Dict[str, Any] = {
            "article_summary": asdict(summary),
            "bias_analysis": self._validate_section(bias_analysis, "bias_analysis"),
            "emotion_analysis": self._validate_section(emotion_analysis, "emotion_analysis"),
            "narrative_structure": self._validate_section(narrative_structure, "narrative_structure"),
            "entity_graph": self._validate_section(entity_graph, "entity_graph"),
            "credibility_score": credibility_score,
            "explainability": self._validate_section(explainability, "explainability"),
        }

        if (
            explainability
            and self.config.save_explanation_artifacts
            and article_id
        ):
            self.save_explanation_artifacts(
                article_id=article_id,
                explainability=explainability,
            )

        logger.info("Report generated successfully")
        return report

    def generate_from_inference_outputs(
        self,
        *,
        article_text: str,
        prediction: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        source: Optional[str] = None,
        article_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Integrate report generation directly from src.inference outputs.

        When the prediction dict contains an 'explainability' key (produced
        by PredictionPipeline.predict_with_explanation), those contents are
        forwarded to generate_report() and optionally persisted as artifacts.
        """

        prediction = prediction or {}
        analysis = analysis or {}

        explainability = prediction.get("explainability") or analysis.get("explainability")

        return self.generate_report(
            article_text=article_text,
            title=title,
            source=source,
            bias_analysis={
                "features": analysis.get("bias_features", {}),
                "prediction": prediction.get("bias"),
            },
            emotion_analysis={
                "features": analysis.get("emotion_features", {}),
                "prediction": prediction.get("emotion"),
            },
            narrative_structure={
                "features": analysis.get("narrative_features", {}),
                "analysis_modules": analysis.get("analysis_modules", {}),
            },
            entity_graph={
                "graph_features": analysis.get("graph_features", {}),
                "graph_pipeline": analysis.get("graph_pipeline", {}),
            },
            credibility_score=prediction.get(
                "credibility_score",
                analysis.get("scores", {}).get("truthlens_credibility_score"),
            ),
            explainability=explainability,
            article_id=article_id,
        )

    def to_json(self, report: Dict[str, Any]) -> str:
        """
        Convert report to JSON string.
        """
        try:
            if self.config.pretty_json:
                return json.dumps(report, indent=4, ensure_ascii=False)
            return json.dumps(report)
        except Exception as exc:
            logger.exception("JSON serialization failed")
            raise RuntimeError("Report serialization failed") from exc

    def save_json(self, report: Dict[str, Any], filepath: str) -> None:
        """
        Save report to JSON file.
        """
        try:
            json_data = self.to_json(report)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_data)

            logger.info("Report saved to %s", filepath)

        except Exception as exc:
            logger.exception("Failed to save report")
            raise RuntimeError("Report save failed") from exc
