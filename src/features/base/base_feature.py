"""
File Name: base_feature.py
Module: Feature Engineering Base Abstractions
Description:
    Defines the base abstraction used for implementing feature extractors
    across the TruthLens feature engineering system. All feature modules
    must inherit from BaseFeature and implement the required interface.

    The abstraction ensures consistent feature extraction behavior,
    standardized outputs, structured logging, validation, and integration
    with the feature registry and feature pipelines.

Dependencies:
    dataclasses
    typing
    logging

Inputs:
    text: str = ""
    metadata: Optional[Dict]

Outputs:
    Dict[str, Any] representing extracted feature values
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class FeatureContext:
    """
    Container object used to pass contextual information
    during feature extraction.

    Attributes
    ----------
    text : str
        Input text used for feature extraction.

    metadata : Optional[Dict[str, Any]]
        Additional metadata (source, timestamp, etc.).

    tokens : Optional[List[str]]
        Pre-tokenized representation of text if available.

    embeddings : Optional[Any]
        Optional precomputed embeddings.

    cache : Optional[Dict[str, Any]]
        Shared cache across feature modules.
    """

    text: str
    metadata: Optional[Dict[str, Any]] = None
    tokens: Optional[List[str]] = None
    embeddings: Optional[Any] = None
    cache: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseFeature:
    """
    Abstract base class for all feature extractors.

    Each feature module must subclass BaseFeature and implement
    the `extract()` method.

    Example
    -------
    class SentimentFeature(BaseFeature):

        def extract(self, context: FeatureContext) -> Dict[str, float]:
            return {"sentiment_score": 0.85}
    """

    name: str
    version: str = "1.0"
    description: Optional[str] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate feature configuration after initialization."""
        if not self.name:
            raise ValueError("Feature must have a valid name")

        logger.debug(
            "Initialized feature: %s | version=%s | enabled=%s",
            self.name,
            self.version,
            self.enabled,
        )

    # ------------------------------------------------------------------
    # Core Feature Interface
    # ------------------------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, Any]:
        """
        Extract features from input context.

        Parameters
        ----------
        context : FeatureContext
            Feature extraction context.

        Returns
        -------
        Dict[str, Any]
            Dictionary of feature name -> value.

        Raises
        ------
        NotImplementedError
            If subclass does not implement extraction logic.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement extract() method."
        )

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def validate_output(self, features: Dict[str, Any]) -> None:
        """
        Validate extracted feature outputs.

        Parameters
        ----------
        features : Dict[str, Any]

        Raises
        ------
        ValueError
            If output format is invalid.
        """

        if not isinstance(features, dict):
            raise ValueError(
                f"Feature '{self.name}' must return a dictionary."
            )

        for key in features:
            if not isinstance(key, str):
                raise ValueError(
                    f"Feature key must be string, got {type(key)}"
                )

    def safe_extract(self, context: FeatureContext) -> Dict[str, Any]:
        """
        Safe wrapper around feature extraction with logging
        and error handling.

        Parameters
        ----------
        context : FeatureContext

        Returns
        -------
        Dict[str, Any]
        """

        if not self.enabled:
            logger.debug("Feature '%s' is disabled", self.name)
            return {}

        if not isinstance(context.text, str):
            raise TypeError("FeatureContext.text must be a string")

        try:
            features = self.extract(context)

            self.validate_output(features)

            logger.debug(
                "Feature '%s' extracted %d values",
                self.name,
                len(features),
            )

            return features

        except Exception as exc:
            logger.exception(
                "Feature extraction failed for feature '%s'",
                self.name,
            )
            raise RuntimeError(
                f"Feature extraction failed for {self.name}"
            ) from exc

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata describing this feature module.

        Returns
        -------
        Dict[str, Any]
        """

        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled,
            "class": self.__class__.__name__,
        }

    # ------------------------------------------------------------------
    # Optional Hooks
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Optional initialization hook.

        Called before feature extraction begins.
        """
        logger.debug("Initializing feature '%s'", self.name)

    def teardown(self) -> None:
        """
        Optional teardown hook.

        Called when feature pipeline completes.
        """
        logger.debug("Tearing down feature '%s'", self.name)