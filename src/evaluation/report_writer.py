"""
File: report_writer.py
"""

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
    elif hasattr(obj, "item"):
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
# PLOT HELPERS
# =========================================================
def _plot_bar(data: Dict[str, float], save_path: Path):

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(data.keys(), data.values())
    plt.xticks(rotation=45)
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

    # ---------------------------
    # ADD METADATA
    # ---------------------------
    report["metadata"] = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    # ---------------------------
    # SAFE SERIALIZATION
    # ---------------------------
    safe_report = _make_serializable(report)

    # ---------------------------
    # SAVE JSON
    # ---------------------------
    with open(base_path, "w") as f:
        json.dump(safe_report, f, indent=4)

    logger.info("Saved report JSON: %s", base_path)

    if not generate_plots:
        return base_path

    # ---------------------------
    # CREATE STRUCTURE
    # ---------------------------
    plots_dir = base_path.parent / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ---------------------------
    # SUMMARY PLOT
    # ---------------------------
    summary = report.get("summary", {})
    numeric_summary = {
        k: float(v) for k, v in summary.items() if isinstance(v, (int, float))
    }

    if numeric_summary:
        _plot_bar(numeric_summary, plots_dir / "summary.png")

    # ---------------------------
    # PER TASK PLOTS
    # ---------------------------
    tasks = report.get("tasks", {})

    for task, metrics in tasks.items():

        numeric = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }

        if numeric:
            _plot_bar(numeric, plots_dir / f"{task}_metrics.png")

    # ---------------------------
    # CALIBRATION PLOTS (IF EXISTS)
    # ---------------------------
    calibration = report.get("calibration", {})

    for task, val in calibration.items():
        if isinstance(val, (int, float)):
            _plot_bar({task: val}, plots_dir / f"{task}_calibration.png")

    logger.info("Report artifacts generated")

    return base_path