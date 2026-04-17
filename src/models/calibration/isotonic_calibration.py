"""
File Name: isotonic_calibration.py
Module: calibration
Description:
    Implements isotonic regression-based post-hoc calibration for classification
    models. Isotonic regression is a non-parametric method that fits a
    monotonically non-decreasing function to map model confidence scores to
    calibrated probabilities.

    This module supports binary and multi-class classification models using
    a one-vs-rest strategy. It integrates with sklearn's IsotonicRegression
    and NumPy for efficient probability estimation.

    The implementation follows research-grade engineering standards used in
    modern ML systems and supports structured logging and reproducibility.

Dependencies:
    numpy
    sklearn
    logging
    typing
Inputs:
    Model logits or probability scores and ground truth labels.
Outputs:
    Calibrated probability predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression


logger = logging.getLogger(__name__)


@dataclass
class IsotonicCalibrationConfig:
    """
    Configuration for isotonic regression calibration.
    """

    out_of_bounds: str = "clip"
    increasing: bool = True


class IsotonicCalibrator:
    """
    Isotonic regression calibrator for post-hoc probability calibration.

    Fits a separate IsotonicRegression model per class using a one-vs-rest
    strategy. Supports binary and multi-class classification.

    References
    ----------
    Zadrozny & Elkan (2002)
    "Transforming Classifier Scores into Accurate Multiclass Probability Estimates"
    """

    def __init__(
        self,
        config: Optional[IsotonicCalibrationConfig] = None,
    ) -> None:
        self.config = config or IsotonicCalibrationConfig()
        self._calibrators: List[IsotonicRegression] = []
        self._num_classes: int = 0
        self._is_fitted: bool = False
        self._binary_vector_input: bool = False

    def _prepare_input(self, probabilities: np.ndarray, fit_stage: bool = False) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)

        if probabilities.ndim == 1:
            p1 = probabilities.reshape(-1, 1)
            if np.any((p1 < 0) | (p1 > 1)):
                raise ValueError("Binary probabilities must be in [0, 1].")
            probabilities = np.hstack([1.0 - p1, p1])
            if fit_stage:
                self._binary_vector_input = True

        if probabilities.ndim != 2:
            raise ValueError("probabilities must be shape (n_samples, n_classes) or (n_samples,)")

        if np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("All probabilities must be in [0, 1].")

        return probabilities

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "IsotonicCalibrator":
        """
        Fit isotonic regression calibrators on validation set probabilities.

        Parameters
        ----------
        probabilities : np.ndarray, shape (n_samples, n_classes) or (n_samples,)
            Predicted probabilities from uncalibrated model.
        labels : np.ndarray, shape (n_samples,)
            Ground truth class indices.

        Returns
        -------
        IsotonicCalibrator
            Fitted calibrator instance.
        """

        probabilities = self._prepare_input(probabilities, fit_stage=True)
        labels = np.asarray(labels, dtype=np.int64)

        if probabilities.shape[0] != labels.shape[0]:
            raise ValueError(
                "probabilities and labels must have the same number of samples."
            )

        n_classes = probabilities.shape[1]
        if labels.min() < 0 or labels.max() >= n_classes:
            raise ValueError(f"labels must be in [0, {n_classes - 1}]")

        self._num_classes = n_classes
        self._calibrators = []

        for class_idx in range(n_classes):
            binary_labels = (labels == class_idx).astype(np.float64)
            class_scores = probabilities[:, class_idx]

            calibrator = IsotonicRegression(
                out_of_bounds=self.config.out_of_bounds,
                increasing=self.config.increasing,
            )
            calibrator.fit(class_scores, binary_labels)
            self._calibrators.append(calibrator)

        self._is_fitted = True
        logger.info("IsotonicCalibrator fitted for %d classes.", n_classes)

        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Produce calibrated probability estimates.

        Parameters
        ----------
        probabilities : np.ndarray, shape (n_samples, n_classes) or (n_samples,)
            Raw probability scores from uncalibrated model.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Calibrated and row-normalized probability estimates.
        """

        if not self._is_fitted:
            raise RuntimeError(
                "IsotonicCalibrator must be fitted before calling predict_proba."
            )

        probabilities = self._prepare_input(probabilities, fit_stage=False)

        if probabilities.shape[1] != self._num_classes:
            raise ValueError(
                f"Expected {self._num_classes} classes but got {probabilities.shape[1]}."
            )

        calibrated = np.zeros_like(probabilities)

        for class_idx, calibrator in enumerate(self._calibrators):
            calibrated[:, class_idx] = calibrator.predict(probabilities[:, class_idx])

        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        calibrated = calibrated / row_sums

        return calibrated

    def calibrate(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """
        Fit calibrator and return calibrated probabilities on the same set.

        Parameters
        ----------
        probabilities : np.ndarray
            Validation set probabilities from uncalibrated model.
        labels : np.ndarray
            Ground truth labels.

        Returns
        -------
        np.ndarray
            Calibrated probability estimates.
        """

        self.fit(probabilities, labels)
        return self.predict_proba(probabilities)
