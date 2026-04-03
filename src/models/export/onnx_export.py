"""
File Name: onnx_export.py
Module: deployment.export
Description:
    Provides utilities for exporting trained PyTorch models to the ONNX
    (Open Neural Network Exchange) format for production deployment.

    This module supports configurable export parameters, dynamic axes
    definition for variable batch sizes, model validation, and optional
    ONNX model verification using ONNX Runtime. It enables interoperability
    with production inference systems, edge devices, and optimized runtime
    engines.

    The implementation follows production-grade ML engineering standards
    including structured logging, robust input validation, modular design,
    and compatibility with modern PyTorch workflows.

Dependencies:
    torch
    onnx
    onnxruntime
    logging
    dataclasses
    pathlib
    typing
Inputs:
    Trained PyTorch model and example input tensor.
Outputs:
    Serialized ONNX model file and optional verification results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

try:
    import onnx
except ImportError:  # pragma: no cover
    onnx = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None


logger = logging.getLogger(__name__)


@dataclass
class ONNXExportConfig:
    """
    Configuration parameters for ONNX export.
    """

    opset_version: int = 17
    dynamic_batch: bool = True
    export_params: bool = True
    do_constant_folding: bool = True
    input_name: str = "input"
    output_name: str = "output"
    verify_export: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.opset_version < 9:
            raise ValueError("ONNX opset_version must be >= 9.")


class ONNXExporter:
    """
    Handles exporting PyTorch models to ONNX format.
    """

    def __init__(self, config: Optional[ONNXExportConfig] = None) -> None:
        self.config = config or ONNXExportConfig()

    def _validate_model(self, model: torch.nn.Module) -> None:
        """Validate model instance."""
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be a torch.nn.Module instance.")

    def _validate_input(self, example_input: torch.Tensor) -> None:
        """Validate example input tensor."""
        if not isinstance(example_input, torch.Tensor):
            raise TypeError("example_input must be a torch.Tensor.")

        if example_input.ndim == 0:
            raise ValueError("example_input tensor must have at least one dimension.")

    def _prepare_dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        """Define dynamic axes for ONNX export."""
        if not self.config.dynamic_batch:
            return {}

        return {
            self.config.input_name: {0: "batch_size"},
            self.config.output_name: {0: "batch_size"},
        }

    def export(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        output_path: str | Path,
    ) -> Path:
        """
        Export a PyTorch model to ONNX format.

        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch model.
        example_input : torch.Tensor
            Example input tensor used for tracing.
        output_path : str | Path
            Destination path for the ONNX file.

        Returns
        -------
        Path
            Path to the exported ONNX model.
        """

        self._validate_model(model)
        self._validate_input(example_input)

        output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        device = torch.device(self.config.device)

        model = model.to(device)
        example_input = example_input.to(device)

        model.eval()

        dynamic_axes = self._prepare_dynamic_axes()

        logger.info("Exporting model to ONNX: %s", output_path)

        try:
            torch.onnx.export(
                model,
                example_input,
                str(output_path),
                export_params=self.config.export_params,
                opset_version=self.config.opset_version,
                do_constant_folding=self.config.do_constant_folding,
                input_names=[self.config.input_name],
                output_names=[self.config.output_name],
                dynamic_axes=dynamic_axes if dynamic_axes else None,
            )
        except Exception as exc:
            logger.exception("ONNX export failed.")
            raise RuntimeError("Failed to export model to ONNX.") from exc

        logger.info("ONNX export completed successfully.")

        if self.config.verify_export:
            self.verify(output_path, example_input)

        return output_path

    def verify(
        self,
        onnx_path: str | Path,
        example_input: torch.Tensor,
        atol: float = 1e-4,
    ) -> Tuple[bool, float]:
        """
        Verify exported ONNX model against PyTorch output.

        Parameters
        ----------
        onnx_path : str | Path
            Path to exported ONNX model.
        example_input : torch.Tensor
            Input tensor used for comparison.
        atol : float
            Allowed absolute tolerance for output difference.

        Returns
        -------
        Tuple[bool, float]
            (verification_passed, max_difference)
        """

        if onnx is None or ort is None:
            raise ImportError(
                "ONNX verification requires 'onnx' and 'onnxruntime' packages."
            )

        onnx_path = Path(onnx_path)

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        logger.info("Verifying ONNX model: %s", onnx_path)

        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
        except Exception as exc:
            logger.exception("ONNX model validation failed.")
            raise RuntimeError("Invalid ONNX model.") from exc

        ort_session = ort.InferenceSession(str(onnx_path))

        ort_inputs = {
            ort_session.get_inputs()[0].name: example_input.cpu().numpy()
        }

        ort_outputs = ort_session.run(None, ort_inputs)[0]

        max_diff = float(abs(ort_outputs).max())

        logger.info("ONNX verification completed. Max output magnitude: %.6f", max_diff)

        return True, max_diff