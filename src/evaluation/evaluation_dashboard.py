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

def render_metrics(task_data: Dict):

    metrics = task_data.get("metrics", {})

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
# DATASET STATS
# =========================================================

def render_dataset_stats(task_data):

    stats = task_data.get("dataset_stats", {})

    if not stats:
        return

    st.subheader("Dataset Statistics")
    st.json(stats)


# =========================================================
# CONFUSION MATRIX
# =========================================================

def render_confusion(task_data):

    metrics = task_data.get("metrics", {})
    confusion = metrics.get("confusion")

    if not confusion:
        return

    st.subheader("Confusion Matrix")

    matrix = np.array(confusion["matrix"])

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap="Blues")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, matrix[i, j], ha="center", va="center")

    fig.colorbar(im)
    st.pyplot(fig)


# =========================================================
#  RELIABILITY DIAGRAM
# =========================================================

def render_reliability(cal):

    rd = cal.get("reliability_diagram")

    if not rd:
        return

    st.subheader("Reliability Diagram")

    conf = rd.get("confidence")
    acc = rd.get("accuracy")

    if conf and acc:
        fig, ax = plt.subplots()

        ax.plot(conf, acc, marker="o", label="Model")
        ax.plot([0, 1], [0, 1], "--", label="Perfect")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.legend()

        st.pyplot(fig)


# =========================================================
#  CONFIDENCE DISTRIBUTION
# =========================================================

def render_confidence(cal):

    conf = cal.get("confidence")

    if not conf:
        return

    st.subheader("Confidence Distribution")

    fig, ax = plt.subplots()
    ax.hist(conf, bins=20)
    st.pyplot(fig)


# =========================================================
# CALIBRATION
# =========================================================

def render_calibration(report, task):

    cal = report.get("calibration", {}).get(task)

    if not cal:
        return

    st.subheader("Calibration")

    for k, v in cal.items():
        if isinstance(v, (int, float)):
            st.metric(k, f"{v:.4f}")

    render_confidence(cal)        #  NEW
    render_reliability(cal)       #  NEW


# =========================================================
#  ERROR ANALYSIS
# =========================================================

def render_error_analysis(report, task):

    err = report.get("error_analysis", {}).get(task)

    if not err:
        return

    st.subheader("Error Analysis")

    st.json(err)

    if "error_rate_per_class" in err:
        fig, ax = plt.subplots()
        ax.bar(err["error_rate_per_class"].keys(),
               err["error_rate_per_class"].values())
        st.pyplot(fig)


# =========================================================
# THRESHOLD
# =========================================================

def render_thresholds(report, task):

    th = report.get("optimal_thresholds", {}).get(task)

    if not th:
        return

    st.subheader("Optimal Threshold")

    st.metric("Threshold", f"{th:.4f}")


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

    keys = list(corr.keys())
    tasks = list(set(k.split("_")[0] for k in keys))

    matrix = pd.DataFrame(0.0, index=tasks, columns=tasks)

    for k, v in corr.items():
        t1, t2 = k.split("_")
        matrix.loc[t1, t2] = v
        matrix.loc[t2, t1] = v

    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap="coolwarm")
    fig.colorbar(cax)

    ax.set_xticks(range(len(tasks)))
    ax.set_yticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45)
    ax.set_yticklabels(tasks)

    st.pyplot(fig)


# =========================================================
# ADVANCED
# =========================================================

def render_advanced(report):

    adv = report.get("advanced_analysis")

    if not adv:
        return

    st.subheader("Advanced Analysis")
    st.json(adv)


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

    task_data = tasks[task]

    col1, col2 = st.columns(2)

    with col1:
        render_metrics(task_data)
        render_dataset_stats(task_data)
        render_calibration(report, task)
        render_thresholds(report, task)

    with col2:
        render_uncertainty(report, task)
        render_confusion(task_data)
        render_error_analysis(report, task)

    st.divider()

    render_correlation(report)

    st.divider()

    render_advanced(report)