from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any
import json
import datetime

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================
# SAFE SERIALIZATION
# =========================================================

def _make_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


# =========================================================
# VALIDATION
# =========================================================

def _validate_report(report: Dict[str, Any]):
    if not isinstance(report, dict):
        raise TypeError("Report must be dict")


# =========================================================
# PLOT UTILS
# =========================================================

def _plot_bar(data: Dict[str, float], save_path: Path):
    import matplotlib.pyplot as plt

    if not data:
        return

    fig, ax = plt.subplots()
    ax.bar(data.keys(), data.values())
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_list(values, save_path: Path, title: str = ""):
    import matplotlib.pyplot as plt

    if not values:
        return

    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# 🔥 NEW

def _plot_hist(values, save_path: Path, title: str = ""):
    import matplotlib.pyplot as plt

    if not values:
        return

    fig, ax = plt.subplots()
    ax.hist(values, bins=20)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_reliability(conf, acc, save_path: Path):
    import matplotlib.pyplot as plt

    if not conf or not acc:
        return

    fig, ax = plt.subplots()

    ax.plot(conf, acc, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "--", label="Perfect")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# =========================================================
# MAIN
# =========================================================

def save_report(
    report: Dict[str, Any],
    path: str | Path,
    generate_plots: bool = True,
) -> Path:

    _validate_report(report)

    base_path = Path(path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # METADATA
    # =====================================================
    metadata = report.get("metadata", {})

    metadata.update({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "evaluation_version": "v3",  # 🔥 upgraded
        "tasks": list(report.get("tasks", {}).keys()),
    })

    report["metadata"] = metadata

    # =====================================================
    # SERIALIZE
    # =====================================================
    safe_report = _make_serializable(report)

    with open(base_path, "w") as f:
        json.dump(safe_report, f, indent=4)

    logger.info("Saved report JSON: %s", base_path)

    if not generate_plots:
        return base_path

    # =====================================================
    # DIRECTORIES
    # =====================================================
    plots_dir = base_path.parent / "plots"

    summary_dir = plots_dir / "summary"
    task_dir = plots_dir / "tasks"
    calib_dir = plots_dir / "calibration"

    error_dir = plots_dir / "error_analysis"
    confidence_dir = plots_dir / "confidence"
    threshold_dir = plots_dir / "thresholds"
    monitoring_dir = plots_dir / "monitoring"

    for d in [
        plots_dir, summary_dir, task_dir, calib_dir,
        error_dir, confidence_dir, threshold_dir, monitoring_dir
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # SUMMARY
    # =====================================================
    summary = report.get("summary", {})
    numeric_summary = {
        k: float(v) for k, v in summary.items() if isinstance(v, (int, float))
    }

    if numeric_summary:
        _plot_bar(numeric_summary, summary_dir / "summary.png")

    # =====================================================
    # TASK METRICS
    # =====================================================
    tasks = report.get("tasks", {})

    for task, data in tasks.items():

        metrics = data.get("metrics", {})

        numeric = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

        if numeric:
            _plot_bar(numeric, task_dir / f"{task}_metrics.png")

        if "per_class_f1" in metrics:
            _plot_list(
                metrics["per_class_f1"],
                task_dir / f"{task}_per_class_f1.png",
                "Per Class F1",
            )

        if "per_label_f1" in metrics:
            _plot_list(
                metrics["per_label_f1"],
                task_dir / f"{task}_per_label_f1.png",
                "Per Label F1",
            )

    # =====================================================
    # CALIBRATION
    # =====================================================
    calibration = report.get("calibration", {})

    for task, cal in calibration.items():

        if not isinstance(cal, dict):
            continue

        numeric = {
            k: v for k, v in cal.items() if isinstance(v, (int, float))
        }

        if numeric:
            _plot_bar(numeric, calib_dir / f"{task}_calibration.png")

        if "classwise_ece" in cal:
            _plot_bar(
                cal["classwise_ece"],
                calib_dir / f"{task}_classwise_ece.png",
            )

        if "per_label_ece" in cal:
            _plot_list(
                cal["per_label_ece"],
                calib_dir / f"{task}_per_label_ece.png",
                "Per Label ECE",
            )

        # 🔥 confidence
        if "confidence" in cal:
            _plot_hist(
                cal["confidence"],
                confidence_dir / f"{task}_confidence.png",
                "Confidence Distribution"
            )

        # 🔥 reliability
        rd = cal.get("reliability_diagram")
        if rd:
            _plot_reliability(
                rd.get("confidence"),
                rd.get("accuracy"),
                calib_dir / f"{task}_reliability.png"
            )

    # =====================================================
    # ERROR ANALYSIS
    # =====================================================
    error_analysis = report.get("error_analysis", {})

    for task, err in error_analysis.items():

        if not isinstance(err, dict):
            continue

        if "error_rate_per_class" in err:
            _plot_bar(
                err["error_rate_per_class"],
                error_dir / f"{task}_error_rate.png"
            )

    # =====================================================
    # THRESHOLDS
    # =====================================================
    thresholds = report.get("optimal_thresholds", {})

    for task, th in thresholds.items():

        if isinstance(th, (int, float)):
            _plot_bar(
                {"threshold": float(th)},
                threshold_dir / f"{task}_threshold.png"
            )

    # =====================================================
    # MONITORING
    # =====================================================
    monitoring = report.get("monitoring", {})

    for key, val in monitoring.items():

        if isinstance(val, (list, np.ndarray)):
            _plot_list(
                val,
                monitoring_dir / f"{key}.png",
                key
            )

    logger.info("Report artifacts generated")

    return base_path