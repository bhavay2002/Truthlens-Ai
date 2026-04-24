"""
Final Production Dashboard (TruthLens)
"""

from __future__ import annotations

import json
import logging
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    import streamlit as st
except ImportError:
    st = None

logger = logging.getLogger(__name__)


# =========================================================
# UTILS
# =========================================================
def _ensure_streamlit():
    if st is None:
        raise RuntimeError("Install streamlit")


def load_report(path):
    with open(path) as f:
        return json.load(f)


# =========================================================
# TASK SELECTOR
# =========================================================
def select_task(tasks):
    return st.sidebar.selectbox("Task", list(tasks.keys()))


# =========================================================
# METRICS
# =========================================================
def render_metrics(metrics: Dict):

    st.subheader("Metrics")

    df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    st.dataframe(df, use_container_width=True)

    numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    if numeric:
        fig, ax = plt.subplots()
        ax.bar(numeric.keys(), numeric.values())
        plt.xticks(rotation=45)
        st.pyplot(fig)


# =========================================================
# CONFUSION (FIXED)
# =========================================================
def render_confusion(metrics):

    confusion = metrics.get("confusion")

    if not confusion:
        return

    st.subheader("Confusion Matrix")

    matrix = np.array(confusion["matrix"])

    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap="Blues")
    st.pyplot(fig)


# =========================================================
# CALIBRATION (FIXED)
# =========================================================
def render_calibration(report, task):

    cal = report.get("calibration", {}).get(task)

    if not cal:
        return

    st.subheader("Calibration")

    for k, v in cal.items():
        if isinstance(v, (int, float)):
            st.metric(k, f"{v:.4f}")


# =========================================================
# UNCERTAINTY
# =========================================================
def render_uncertainty(report, task):

    unc = report.get("uncertainty", {}).get(task)

    if not unc:
        return

    st.subheader("Uncertainty")

    df = pd.DataFrame(unc.items(), columns=["Metric", "Value"])
    st.dataframe(df)


# =========================================================
# CORRELATION
# =========================================================
def render_correlation(report):

    corr = report.get("task_correlation")

    if not corr:
        return

    st.subheader("Task Correlation")

    df = pd.DataFrame(corr)

    fig, ax = plt.subplots()
    cax = ax.matshow(df, cmap="coolwarm")
    fig.colorbar(cax)

    st.pyplot(fig)


# =========================================================
# EXPLAINABILITY
# =========================================================
def render_explainability(report):

    explain = report.get("explainability")

    if not explain:
        return

    st.subheader("Explainability")
    st.json(explain)


# =========================================================
# MAIN
# =========================================================
def launch_dashboard(report_path):

    _ensure_streamlit()

    st.set_page_config(layout="wide")

    report = load_report(report_path)

    st.title("TruthLens AI Dashboard")

    tasks = report["tasks"]

    task = select_task(tasks)

    col1, col2 = st.columns(2)

    with col1:
        render_metrics(tasks[task])
        render_calibration(report, task)

    with col2:
        render_uncertainty(report, task)
        render_confusion(tasks[task])

    st.divider()

    render_correlation(report)

    st.divider()

    render_explainability(report)