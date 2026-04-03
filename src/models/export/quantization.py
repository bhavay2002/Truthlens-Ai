"""
File Name: quantization.py
Module: deployment.optimization
Description:
    Provides utilities for applying model quantization to PyTorch models
    for efficient deployment. Quantization reduces model size and improves
    inference latency by converting model weights and activations from
    floating-point precision to lower precision formats such as INT8.

    This module supports multiple quantization strategies including:
    • Dynamic Quantization
    • Post-Training Static Quantization
    • Quantization Aware Training (QAT) preparation utilities

    The implementation is suitable for production ML systems and includes
    structured logging, input validation, device management, and modular
    configuration.

Dependencies:
    torch
    torch.nn
    torch.quantization
    logging
    dataclasses
    typing
Inputs:
    Trained PyTorch model and optional calibration data.
Outputs:
    Quantized PyTorch model ready for efficient inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Type

import torch
import torch.nn as nn
import torch.quantization as quantization

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.
    """

    method: str = "dynamic"
    dtype: torch.dtype = torch.qint8
    backend: str = "fbgemm"
    device: str = "cpu"

    def __post_init__(self) -> None:
        valid_methods = {"dynamic", "static", "qat"}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid quantization method '{self.method}'. "
                f"Supported methods: {valid_methods}"
            )


class QuantizationEngine:
    """
    Handles quantization of PyTorch models using different strategies.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None) -> None:
        self.config = config or QuantizationConfig()

        if self.config.backend not in torch.backends.quantized.supported_engines:
            raise ValueError(
                f"Unsupported quantization backend: {self.config.backend}"
            )

        torch.backends.quantized.engine = self.config.backend

    def _validate_model(self, model: nn.Module) -> None:
        """Validate that model is a PyTorch module."""
        if not isinstance(model, nn.Module):
            raise TypeError("Model must be an instance of torch.nn.Module.")

    def dynamic_quantization(
        self,
        model: nn.Module,
        layers: Optional[Iterable[Type[nn.Module]]] = None,
    ) -> nn.Module:
        """
        Apply dynamic quantization.

        Parameters
        ----------
        model : nn.Module
        layers : Optional iterable of layer types

        Returns
        -------
        nn.Module
            Dynamically quantized model.
        """

        self._validate_model(model)

        if layers is None:
            layers = {nn.Linear}

        logger.info("Applying dynamic quantization.")

        try:
            quantized_model = quantization.quantize_dynamic(
                model,
                layers,
                dtype=self.config.dtype,
            )
        except Exception as exc:
            logger.exception("Dynamic quantization failed.")
            raise RuntimeError("Dynamic quantization failed.") from exc

        logger.info("Dynamic quantization completed.")

        return quantized_model

    def static_quantization_prepare(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Prepare model for post-training static quantization.

        Parameters
        ----------
        model : nn.Module

        Returns
        -------
        nn.Module
            Model prepared for calibration.
        """

        self._validate_model(model)

        model.eval()

        model.qconfig = quantization.get_default_qconfig(self.config.backend)

        logger.info("Preparing model for static quantization.")

        try:
            prepared_model = quantization.prepare(model, inplace=False)
        except Exception as exc:
            logger.exception("Static quantization preparation failed.")
            raise RuntimeError("Static quantization preparation failed.") from exc

        return prepared_model

    def static_quantization_convert(
        self,
        prepared_model: nn.Module,
    ) -> nn.Module:
        """
        Convert calibrated model to static quantized model.

        Parameters
        ----------
        prepared_model : nn.Module

        Returns
        -------
        nn.Module
            Quantized model.
        """

        prepared_model.eval()

        logger.info("Converting model to static quantized version.")

        try:
            quantized_model = quantization.convert(prepared_model, inplace=False)
        except Exception as exc:
            logger.exception("Static quantization conversion failed.")
            raise RuntimeError("Static quantization conversion failed.") from exc

        logger.info("Static quantization conversion completed.")

        return quantized_model

    def prepare_qat(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Prepare model for Quantization Aware Training (QAT).

        Parameters
        ----------
        model : nn.Module

        Returns
        -------
        nn.Module
            Model prepared for QAT.
        """

        self._validate_model(model)

        model.train()

        model.qconfig = quantization.get_default_qat_qconfig(self.config.backend)

        logger.info("Preparing model for Quantization Aware Training.")

        try:
            qat_model = quantization.prepare_qat(model, inplace=False)
        except Exception as exc:
            logger.exception("QAT preparation failed.")
            raise RuntimeError("QAT preparation failed.") from exc

        return qat_model

    def apply(
        self,
        model: nn.Module,
        calibration_data: Optional[Iterable[torch.Tensor]] = None,
    ) -> nn.Module:
        """
        Apply quantization according to the configured method.

        Parameters
        ----------
        model : nn.Module
        calibration_data : Optional iterable of tensors used for calibration

        Returns
        -------
        nn.Module
            Quantized model.
        """

        method = self.config.method

        if method == "dynamic":
            return self.dynamic_quantization(model)

        if method == "static":
            if calibration_data is None:
                raise ValueError(
                    "Calibration data is required for static quantization."
                )

            prepared = self.static_quantization_prepare(model)

            logger.info("Running calibration for static quantization.")

            prepared.eval()

            with torch.no_grad():
                for batch in calibration_data:
                    prepared(batch)

            return self.static_quantization_convert(prepared)

        if method == "qat":
            return self.prepare_qat(model)

        raise RuntimeError(f"Unsupported quantization method: {method}")