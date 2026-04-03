"""
File Name: temperature_scaling.py
Module: calibration
Description:
    Implements temperature scaling for post-hoc calibration of classification
    models. Temperature scaling is a simple but effective method for improving
    the calibration of predicted probabilities without modifying the underlying
    model parameters.

    The technique learns a single scalar temperature parameter that rescales
    model logits before applying the softmax function. This module supports
    PyTorch models and integrates with standard ML evaluation pipelines.

    The implementation follows research-grade engineering standards used in
    modern ML systems and supports GPU computation, reproducibility, and
    structured logging.

Dependencies:
    torch
    torch.nn
    torch.optim
    logging
    typing
Inputs:
    Model logits and ground truth labels.
Outputs:
    Calibrated probability predictions and optimized temperature parameter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


logger = logging.getLogger(__name__)


@dataclass
class TemperatureScalingConfig:
    """
    Configuration for temperature scaling optimization.
    """

    lr: float = 0.01
    max_iter: int = 50
    tolerance: float = 1e-6
    device: str = "cpu"


class TemperatureScaler(nn.Module):
    """
    Temperature scaling module for calibrating model logits.

    References
    ----------
    Guo et al. (2017)
    "On Calibration of Modern Neural Networks"
    """

    def __init__(self, config: TemperatureScalingConfig) -> None:
        super().__init__()

        if config.lr <= 0:
            raise ValueError("Learning rate must be positive.")

        if config.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        self.config = config
        self.temperature = nn.Parameter(torch.ones(1))
        self.device = torch.device(config.device)

        self.to(self.device)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Parameters
        ----------
        logits : torch.Tensor
            Raw logits from the model.

        Returns
        -------
        torch.Tensor
            Scaled logits.
        """

        if logits.ndim < 2:
            raise ValueError("Logits must have shape [batch_size, num_classes].")

        temperature = self.temperature.expand_as(logits)

        return logits / temperature

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert scaled logits to calibrated probabilities.

        Parameters
        ----------
        logits : torch.Tensor

        Returns
        -------
        torch.Tensor
        """

        scaled_logits = self.forward(logits)
        return torch.softmax(scaled_logits, dim=1)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Optimize the temperature parameter using validation data.

        Parameters
        ----------
        logits : torch.Tensor
            Model logits from validation set.

        labels : torch.Tensor
            Ground truth labels.

        Returns
        -------
        float
            Optimized temperature value.
        """

        if logits.shape[0] != labels.shape[0]:
            raise ValueError("Logits and labels must have matching batch size.")

        logits = logits.to(self.device)
        labels = labels.to(self.device)

        nll_criterion = nn.CrossEntropyLoss()

        optimizer = optim.LBFGS(
            [self.temperature],
            lr=self.config.lr,
            max_iter=self.config.max_iter,
        )

        logger.info("Starting temperature scaling optimization.")

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()

            loss = nll_criterion(self.forward(logits), labels)

            loss.backward()

            return loss

        optimizer.step(_closure)

        temperature_value = float(self.temperature.detach().cpu().item())

        if temperature_value <= 0:
            raise RuntimeError("Optimized temperature must be positive.")

        logger.info("Optimized temperature: %.6f", temperature_value)

        return temperature_value

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Fit temperature and return calibrated probabilities.

        Parameters
        ----------
        logits : torch.Tensor
        labels : torch.Tensor

        Returns
        -------
        Tuple[torch.Tensor, float]
            Calibrated probabilities and learned temperature.
        """

        temperature = self.fit(logits, labels)

        calibrated_probs = self.predict_proba(logits)

        return calibrated_probs, temperature