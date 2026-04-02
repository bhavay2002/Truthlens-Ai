"""
File Name: pdf_report.py
Module: TruthLens AI - PDF Evaluation Report
Description:
    Generates a structured PDF report from TruthLens evaluation results.
    The report includes task-level metrics and overall summary metrics.
    Designed for experiment documentation, research artifacts, and
    reproducible evaluation reports.
Dependencies:
    reportlab.platypus
    reportlab.lib.styles
    reportlab.lib.units
    pathlib
    logging
    typing
Inputs:
    report: dictionary containing evaluation results
    output_path: path where the PDF report will be saved
Outputs:
    PDF report file
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


def _validate_report(report: Dict[str, Any]) -> None:
    """
    Validate report structure.
    """

    if not isinstance(report, dict):
        raise TypeError("Report must be a dictionary.")

    if "tasks" not in report:
        raise ValueError("Report must contain 'tasks' section.")


def _metrics_table(metrics: Dict[str, Any]):
    """
    Convert metrics dictionary into a ReportLab table.
    """

    data = [["Metric", "Value"]]

    for key, value in metrics.items():
        data.append([str(key), str(value)])

    table = Table(data, colWidths=[3 * inch, 3 * inch])

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        )
    )

    return table


def generate_pdf_report(
    report: Dict[str, Any],
    output_path: str | Path
) -> Path:
    """
    Generate PDF evaluation report.
    """

    _validate_report(report)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph("TruthLens AI Evaluation Report", styles["Title"])
    )

    elements.append(Spacer(1, 20))

    tasks = report.get("tasks", {})

    for task, metrics in tasks.items():

        elements.append(
            Paragraph(f"Task: {task}", styles["Heading2"])
        )

        if isinstance(metrics, dict):
            elements.append(_metrics_table(metrics))
        else:
            elements.append(
                Paragraph(str(metrics), styles["BodyText"])
            )

        elements.append(Spacer(1, 16))

    if "summary" in report:

        elements.append(
            Paragraph("Overall Summary", styles["Heading2"])
        )

        elements.append(
            _metrics_table(report["summary"])
        )

        elements.append(Spacer(1, 16))

    try:
        doc = SimpleDocTemplate(str(output_path))
        doc.build(elements)
        logger.info("PDF evaluation report generated at %s", output_path)
    except Exception as exc:
        logger.exception("Failed to generate PDF report")
        raise RuntimeError("PDF report generation failed") from exc

    return output_path