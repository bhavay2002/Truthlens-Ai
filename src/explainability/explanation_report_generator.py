"""
File Name: explanation_report_generator.py
Module: Explainability - Report Generation
Description:
    Generates structured explanation reports for TruthLens AI predictions.
    This module aggregates outputs from multiple explainability components
    (prediction, SHAP, LIME, bias, emotion, attention) and produces
    machine-readable JSON reports as well as human-readable HTML dashboards.

    Designed for research analysis, dashboards, and auditability.

Dependencies:
    logging
    json
    pathlib
    datetime
    typing
    html

Inputs:
    explanation package dictionary produced by explainability modules

Outputs:
    JSON explanation report
    HTML explanation dashboard
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExplanationReportGenerator:
    """
    Generates JSON and HTML explanation reports.
    """

    def __init__(
        self,
        output_dir: str | Path = "reports/explanations",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("ExplanationReportGenerator initialized")

    def _build_file_paths(self, article_id: str) -> Dict[str, Path]:
        safe_id = article_id.replace(" ", "_")
        json_path = self.output_dir / f"{safe_id}.json"
        html_path = self.output_dir / f"{safe_id}.html"

        return {
            "json": json_path,
            "html": html_path,
        }

    def save_json(
        self,
        article_id: str,
        explanation: Dict[str, Any],
    ) -> Path:
        """
        Save explanation report as JSON.
        """

        paths = self._build_file_paths(article_id)

        payload = {
            "article_id": article_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "explanation": explanation,
        }

        with paths["json"].open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("Saved explanation JSON report: %s", paths["json"])

        return paths["json"]

    def _render_section(
        self,
        title: str,
        content: Optional[Any],
    ) -> str:
        if content is None:
            return f"<h3>{escape(title)}</h3><p>No data available.</p>"

        pretty = json.dumps(content, indent=2)

        return f"""
        <h3>{escape(title)}</h3>
        <pre>{escape(pretty)}</pre>
        """

    def save_html(
        self,
        article_id: str,
        explanation: Dict[str, Any],
    ) -> Path:
        """
        Save explanation report as HTML dashboard.
        """

        paths = self._build_file_paths(article_id)

        prediction = explanation.get("prediction")
        shap_exp = explanation.get("shap_explanation")
        lime_exp = explanation.get("lime_explanation")
        bias_exp = explanation.get("bias_explanation")
        emotion_exp = explanation.get("emotion_explanation")
        attention = explanation.get("attention_scores")

        html_content = f"""
        <html>
        <head>
            <title>TruthLens Explanation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}

                h1 {{
                    color: #222;
                }}

                h2 {{
                    margin-top: 30px;
                    color: #333;
                }}

                h3 {{
                    color: #444;
                }}

                pre {{
                    background: #fff;
                    padding: 15px;
                    border: 1px solid #ddd;
                    overflow-x: auto;
                }}

                .section {{
                    margin-bottom: 30px;
                }}
            </style>
        </head>

        <body>

        <h1>TruthLens Explanation Report</h1>

        <div class="section">
        <h2>Prediction</h2>
        {self._render_section("Prediction Output", prediction)}
        </div>

        <div class="section">
        <h2>SHAP Explanation</h2>
        {self._render_section("SHAP", shap_exp)}
        </div>

        <div class="section">
        <h2>LIME Explanation</h2>
        {self._render_section("LIME", lime_exp)}
        </div>

        <div class="section">
        <h2>Bias Analysis</h2>
        {self._render_section("Bias Explanation", bias_exp)}
        </div>

        <div class="section">
        <h2>Emotion Analysis</h2>
        {self._render_section("Emotion Explanation", emotion_exp)}
        </div>

        <div class="section">
        <h2>Attention Analysis</h2>
        {self._render_section("Attention Scores", attention)}
        </div>

        </body>
        </html>
        """

        with paths["html"].open("w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Saved explanation HTML report: %s", paths["html"])

        return paths["html"]

    def generate(
        self,
        article_id: str,
        explanation: Dict[str, Any],
        save_json: bool = True,
        save_html: bool = True,
    ) -> Dict[str, Path]:
        """
        Generate explanation reports.
        """

        paths: Dict[str, Path] = {}

        if save_json:
            paths["json"] = self.save_json(article_id, explanation)

        if save_html:
            paths["html"] = self.save_html(article_id, explanation)

        return paths