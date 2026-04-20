from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExplanationReportGenerator:
    def __init__(self, output_dir: str | Path = "reports/explanations") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ExplanationReportGenerator initialized")

    def _safe_article_id(self, article_id: str) -> str:
        if not isinstance(article_id, str) or not article_id.strip():
            raise ValueError("article_id must be a non-empty string")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", article_id.strip())
        safe = safe.strip("._")
        return safe or "article"

    def _build_file_paths(self, article_id: str) -> Dict[str, Path]:
        safe_id = self._safe_article_id(article_id)
        return {"json": self.output_dir / f"{safe_id}.json", "html": self.output_dir / f"{safe_id}.html"}

    def save_json(self, article_id: str, explanation: Dict[str, Any]) -> Path:
        paths = self._build_file_paths(article_id)
        payload = {
            "article_id": article_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "explanation": explanation,
        }
        with paths["json"].open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return paths["json"]

    def _render_section(self, title: str, content: Optional[Any]) -> str:
        if content is None:
            return f"<h3>{escape(title)}</h3><p>No data available.</p>"
        return f"<h3>{escape(title)}</h3><pre>{escape(json.dumps(content, indent=2))}</pre>"

    def save_html(self, article_id: str, explanation: Dict[str, Any]) -> Path:
        paths = self._build_file_paths(article_id)
        html_content = f"""<html><head><title>TruthLens Explanation Report</title></head><body>
        <h1>TruthLens Explanation Report</h1>
        {self._render_section("Prediction Output", explanation.get("prediction"))}
        {self._render_section("SHAP", explanation.get("shap_explanation"))}
        {self._render_section("LIME", explanation.get("lime_explanation"))}
        {self._render_section("Bias Explanation", explanation.get("bias_explanation"))}
        {self._render_section("Emotion Explanation", explanation.get("emotion_explanation"))}
        {self._render_section("Attention Scores", explanation.get("attention_scores"))}
        </body></html>"""
        with paths["html"].open("w", encoding="utf-8") as f:
            f.write(html_content)
        return paths["html"]

    def generate(self, article_id: str, explanation: Dict[str, Any], save_json: bool = True, save_html: bool = True) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        if save_json:
            out["json"] = self.save_json(article_id, explanation)
        if save_html:
            out["html"] = self.save_html(article_id, explanation)
        return out
