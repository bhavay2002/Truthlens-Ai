"""
File Name: evaluation_dashboard.py
Module: TruthLens AI - Evaluation Dashboard
Description:
    Interactive evaluation dashboard for inspecting TruthLens AI model
    performance reports. The dashboard visualizes per-task metrics and
    global summaries using Streamlit. Designed for research diagnostics
    and experiment comparison.

    Integrates ExplanationReportGenerator to present saved explanation
    artifacts (HTML dashboards and JSON reports) directly within the
    evaluation dashboard when explanation data is present in the report.

Dependencies:
    streamlit
    pandas
    json
    logging
    pathlib
    src.explainability.explanation_report_generator

Inputs:
    report_path: Path to JSON evaluation report
Outputs:
    Interactive Streamlit dashboard
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    st = None

from src.explainability.explanation_report_generator import ExplanationReportGenerator
from src.visualization.visualize import plot_feature_importance

logger = logging.getLogger(__name__)


def _ensure_streamlit() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install 'streamlit' to launch dashboard."
        )


def _load_report(report_path: str | Path) -> Dict[str, Any]:
    """
    Load evaluation report from JSON file.
    """

    path = Path(report_path)

    if not path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON format in evaluation report") from exc

    if "tasks" not in report:
        raise ValueError("Report must contain 'tasks' field")

    return report


def _render_task_metrics(tasks: Dict[str, Dict[str, Any]]) -> None:
    """
    Render metrics for each task.
    """

    _ensure_streamlit()

    st.header("Task Metrics")

    for task, metrics in tasks.items():

        st.subheader(task)

        if not isinstance(metrics, dict):
            st.warning(f"Invalid metric structure for task: {task}")
            continue

        df = pd.DataFrame(
            list(metrics.items()),
            columns=["Metric", "Value"]
        )

        st.dataframe(df, use_container_width=True)

        numeric_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if numeric_metrics:
            fig, _ = plot_feature_importance(
                features=list(numeric_metrics.keys()),
                scores=list(numeric_metrics.values()),
                top_k=min(10, len(numeric_metrics)),
                save_path=None,
            )
            st.pyplot(fig, use_container_width=True)


def _render_summary(summary: Dict[str, Any]) -> None:
    """
    Render overall evaluation summary.
    """

    _ensure_streamlit()

    st.header("Overall Performance")

    try:
        df = pd.DataFrame(
            list(summary.items()),
            columns=["Metric", "Value"]
        )

        st.dataframe(df, use_container_width=True)

    except Exception:
        st.json(summary)

    numeric_summary = {
        k: float(v) for k, v in summary.items()
        if isinstance(v, (int, float))
    }
    if numeric_summary:
        fig, _ = plot_feature_importance(
            features=list(numeric_summary.keys()),
            scores=list(numeric_summary.values()),
            top_k=min(15, len(numeric_summary)),
            save_path=None,
        )
        st.pyplot(fig, use_container_width=True)


def _render_explanation_report(
    explainability: Dict[str, Any],
    article_id: Optional[str] = None,
    output_dir: str = "reports/explanations",
) -> None:
    """
    Render explainability data inside the Streamlit dashboard.

    Displays aggregated token importance, attention rollout scores, and
    cross-method consistency metrics. When HTML artifacts already exist
    on disk (saved by ReportGenerator.save_explanation_artifacts), they
    are embedded via an iframe. Otherwise, the explanation data is shown
    as structured JSON.

    Parameters
    ----------
    explainability : dict
        Explainability output from ExplainabilityLayer.explain().
    article_id : str, optional
        Article identifier used to locate saved HTML artifacts on disk.
    output_dir : str
        Directory where explanation artifacts are stored.
    """

    _ensure_streamlit()

    st.header("Explainability Report")

    if not explainability:
        st.info("No explainability data available for this report.")
        return

    if article_id:
        html_path = Path(output_dir) / f"{article_id}_explanation.html"
        if html_path.exists():
            try:
                html_content = html_path.read_text(encoding="utf-8")
                st.components.v1.html(html_content, height=600, scrolling=True)
                logger.info("Rendered explanation HTML artifact: %s", html_path)
                return
            except Exception as exc:
                logger.warning("Failed to render explanation HTML artifact: %s", exc)

    aggregated = explainability.get("aggregated_explanation")
    if aggregated:
        st.subheader("Aggregated Token Importance")
        tokens = [item.get("token", "") for item in aggregated]
        scores = [item.get("score", 0.0) for item in aggregated]
        df_agg = pd.DataFrame({"Token": tokens, "Importance": scores})
        st.dataframe(df_agg, use_container_width=True)

    rollout = explainability.get("attention_rollout")
    if rollout:
        st.subheader("Attention Rollout")
        aligned_tokens = rollout.get("aligned_tokens", rollout.get("tokens", []))
        aligned_scores = rollout.get("aligned_scores", rollout.get("rollout_scores", []))
        if aligned_tokens and aligned_scores:
            df_rollout = pd.DataFrame(
                {"Token": aligned_tokens, "Rollout Score": aligned_scores}
            )
            st.dataframe(df_rollout, use_container_width=True)

    consistency = explainability.get("consistency_metrics")
    if consistency:
        st.subheader("Cross-Method Consistency")
        df_con = pd.DataFrame(
            list(consistency.items()),
            columns=["Method Pair", "Correlation"],
        )
        st.dataframe(df_con, use_container_width=True)

    propaganda = explainability.get("propaganda_intensity")
    if propaganda is not None:
        st.subheader("Propaganda Intensity")
        st.metric("Propaganda Intensity", f"{propaganda:.4f}")

    with st.expander("Full Explainability Data (JSON)"):
        st.json(explainability)


def launch_dashboard(
    report_path: str | Path,
    explanation_dir: str = "reports/explanations",
) -> None:
    """
    Launch Streamlit evaluation dashboard.

    When the loaded report contains an 'explainability' section, the
    dashboard renders a dedicated Explainability Report panel, embedding
    the saved HTML artifact if available or falling back to structured
    table views of token importance, attention rollout, and consistency.
    """

    _ensure_streamlit()

    logger.info("Launching TruthLens evaluation dashboard")

    try:
        report = _load_report(report_path)

    except Exception as exc:
        st.error(f"Failed to load evaluation report: {exc}")
        logger.exception("Dashboard failed to load report")
        return

    st.set_page_config(
        page_title="TruthLens AI Evaluation Dashboard",
        layout="wide"
    )

    st.title("TruthLens AI Evaluation Dashboard")

    tasks = report.get("tasks", {})
    summary = report.get("summary")
    explainability = report.get("explainability")
    article_id = report.get("article_id")

    _render_task_metrics(tasks)

    if summary:
        _render_summary(summary)

    if explainability:
        _render_explanation_report(
            explainability=explainability,
            article_id=article_id,
            output_dir=explanation_dir,
        )
