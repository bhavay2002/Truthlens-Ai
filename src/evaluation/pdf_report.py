"""
File: pdf_report.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

logger = logging.getLogger(__name__)


# =========================================================
# TABLE UTILS
# =========================================================
def dict_to_table(data: Dict[str, Any]):

    rows = [["Metric", "Value"]]

    for k, v in data.items():
        if isinstance(v, dict):
            rows.append([k, ""])
            for sub_k, sub_v in v.items():
                rows.append([f"  {sub_k}", str(sub_v)])
        else:
            rows.append([k, str(v)])

    table = Table(rows, colWidths=[3 * inch, 3 * inch])

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ])
    )

    return table


# =========================================================
# SECTION RENDERERS
# =========================================================
def render_tasks(elements, tasks, styles):
    elements.append(Paragraph("Task Performance", styles["Heading1"]))
    elements.append(Spacer(1, 10))

    for task, metrics in tasks.items():
        elements.append(Paragraph(f"Task: {task}", styles["Heading2"]))
        elements.append(dict_to_table(metrics))
        elements.append(Spacer(1, 12))


def render_summary(elements, summary, styles):
    elements.append(Paragraph("Overall Summary", styles["Heading1"]))
    elements.append(dict_to_table(summary))
    elements.append(Spacer(1, 12))


def render_section(elements, title, data, styles):
    if not data:
        return

    elements.append(Paragraph(title, styles["Heading1"]))
    elements.append(dict_to_table(data))
    elements.append(Spacer(1, 12))


# =========================================================
# MAIN
# =========================================================
def generate_pdf_report(report: Dict[str, Any], output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("TruthLens AI Evaluation Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # ---------------------------
    # TASKS
    # ---------------------------
    render_tasks(elements, report.get("tasks", {}), styles)

    # ---------------------------
    # SUMMARY
    # ---------------------------
    if "summary" in report:
        render_summary(elements, report["summary"], styles)

    # ---------------------------
    # CALIBRATION
    # ---------------------------
    render_section(
        elements,
        "Calibration",
        report.get("calibration"),
        styles
    )

    # ---------------------------
    # UNCERTAINTY
    # ---------------------------
    render_section(
        elements,
        "Uncertainty",
        report.get("uncertainty"),
        styles
    )

    # ---------------------------
    # TASK CORRELATION
    # ---------------------------
    render_section(
        elements,
        "Task Correlation",
        report.get("task_correlation"),
        styles
    )

    # ---------------------------
    # BUILD PDF
    # ---------------------------
    doc = SimpleDocTemplate(str(output_path))
    doc.build(elements)

    logger.info("PDF report generated: %s", output_path)

    return output_path