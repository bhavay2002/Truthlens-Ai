"""
File Name: torchscript_export.py
Module: deployment.export
Description:
    Provides utilities for exporting trained PyTorch models to TorchScript
    format for production deployment. TorchScript enables models to run
    independently of Python using the PyTorch runtime, which is suitable
    for high-performance inference services, mobile environments, and
    production ML systems.

    This module supports both scripting and tracing export methods,
    structured validation, artifact management, and optional verification
    by comparing TorchScript outputs with original PyTorch model outputs.

Dependencies:
    torch
    logging
    dataclasses
    pathlib
    typing
Inputs:
    Trained PyTorch model and example input tensor.
Outputs:
    Serialized TorchScript model artifact (.pt file).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class TorchScriptExportConfig:
    """
    Configuration for TorchScript export.
    """

    method: str = "trace"
    device: str = "cpu"
    verify_export: bool = True
    strict_trace: bool = True

    def __post_init__(self) -> None:
        valid_methods = {"trace", "script"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid export method '{self.method}'. Must be one of {valid_methods}."
            )


class TorchScriptExporter:
    """
    Handles exporting PyTorch models to TorchScript.
    """

    def __init__(self, config: Optional[TorchScriptExportConfig] = None) -> None:
        self.config = config or TorchScriptExportConfig()

    def _validate_model(self, model: torch.nn.Module) -> None:
        """Validate that the input is a PyTorch model."""
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be an instance of torch.nn.Module.")

    def _validate_input(self, example_input: torch.Tensor) -> None:
        """Validate example input tensor."""
        if not isinstance(example_input, torch.Tensor):
            raise TypeError("example_input must be a torch.Tensor.")

        if example_input.ndim == 0:
            raise ValueError("example_input tensor must have at least one dimension.")

    def _export_trace(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
    ) -> torch.jit.ScriptModule:
        """Export model using tracing."""
        try:
            traced = torch.jit.trace(
                model,
                example_input,
                strict=self.config.strict_trace,
            )
            return traced
        except Exception as exc:
            logger.exception("TorchScript tracing failed.")
            raise RuntimeError("Failed to export TorchScript via tracing.") from exc

    def _export_script(
        self,
        model: torch.nn.Module,
    ) -> torch.jit.ScriptModule:
        """Export model using scripting."""
        try:
            scripted = torch.jit.script(model)
            return scripted
        except Exception as exc:
            logger.exception("TorchScript scripting failed.")
            raise RuntimeError("Failed to export TorchScript via scripting.") from exc

    def export(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        output_path: str | Path,
    ) -> Path:
        """
        Export a PyTorch model to TorchScript format.

        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch model.
        example_input : torch.Tensor
            Example input used for tracing or verification.
        output_path : str | Path
            Destination path for the TorchScript file.

        Returns
        -------
        Path
            Path to exported TorchScript model.
        """

        self._validate_model(model)
        self._validate_input(example_input)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        device = torch.device(self.config.device)

        model = model.to(device)
        example_input = example_input.to(device)

        model.eval()

        logger.info("Exporting TorchScript model using method: %s", self.config.method)

        if self.config.method == "trace":
            ts_model = self._export_trace(model, example_input)
        else:
            ts_model = self._export_script(model)

        try:
            ts_model.save(str(output_path))
        except Exception as exc:
            logger.exception("Failed to save TorchScript model.")
            raise RuntimeError("TorchScript save operation failed.") from exc

        logger.info("TorchScript model exported successfully: %s", output_path)

        if self.config.verify_export:
            self.verify(model, ts_model, example_input)

        return output_path

    def verify(
        self,
        original_model: torch.nn.Module,
        ts_model: torch.jit.ScriptModule,
        example_input: torch.Tensor,
        atol: float = 1e-4,
    ) -> Tuple[bool, float]:
        """
        Verify TorchScript model output against original PyTorch model.

        Parameters
        ----------
        original_model : torch.nn.Module
        ts_model : torch.jit.ScriptModule
        example_input : torch.Tensor
        atol : float

        Returns
        -------
        Tuple[bool, float]
            (verification_passed, max_difference)
        """

        original_model.eval()
        ts_model.eval()

        with torch.no_grad():
            pytorch_output = original_model(example_input)
            ts_output = ts_model(example_input)

        if not isinstance(pytorch_output, torch.Tensor):
            raise TypeError("Model output must be a torch.Tensor for verification.")

        if not isinstance(ts_output, torch.Tensor):
            raise TypeError("TorchScript output must be a torch.Tensor.")

        diff = torch.abs(pytorch_output - ts_output)
        max_diff = float(diff.max().item())

        passed = max_diff <= atol

        logger.info(
            "TorchScript verification result: passed=%s max_diff=%.6f",
            passed,
            max_diff,
        )

        return passed, max_diff