"""
File Name: evaluation_dashboard.py
Module: TruthLens AI - Evaluation Dashboard
Description:
    Interactive evaluation dashboard for inspecting TruthLens AI model
    performance reports. The dashboard visualizes per-task metrics and
    global summaries using Streamlit. Designed for research diagnostics
    and experiment comparison.
Dependencies:
    streamlit
    pandas
    json
    logging
    pathlib
Inputs:
    report_path: Path to JSON evaluation report
Outputs:
    Interactive Streamlit dashboard
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd
try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    st = None

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


def launch_dashboard(report_path: str | Path) -> None:
    """
    Launch Streamlit evaluation dashboard.
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

    _render_task_metrics(tasks)

    if summary:
        _render_summary(summary)
