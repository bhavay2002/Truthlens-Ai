"""
File Name: feature_registry.py
Module: Feature Engineering Registry
Description:
    Implements a centralized registry for feature extractors used in the
    TruthLens feature engineering system. The registry enables automatic
    discovery, registration, and retrieval of feature classes used by the
    feature pipelines.

    This design supports modular feature development, dynamic feature loading,
    and configuration-driven feature activation.

Dependencies:
    dataclasses
    typing
    logging

Inputs:
    Feature classes inheriting from BaseFeature

Outputs:
    Registry of available feature extractors
"""

from __future__ import annotations

import logging
from typing import Dict, Type, List, Optional

from src.features.base.base_feature import BaseFeature

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Central registry responsible for managing feature extractors.

    The registry maps feature names to their implementing classes,
    enabling dynamic instantiation and configuration-driven pipelines.
    """

    _registry: Dict[str, Type[BaseFeature]] = {}

    # ------------------------------------------------------------------
    # Registration Methods
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, feature_cls: Type[BaseFeature]) -> Type[BaseFeature]:
        """
        Register a feature extractor class.

        Parameters
        ----------
        feature_cls : Type[BaseFeature]

        Returns
        -------
        Type[BaseFeature]
            The registered feature class.

        Raises
        ------
        ValueError
            If feature class is invalid or duplicate.
        """

        if not issubclass(feature_cls, BaseFeature):
            raise ValueError(
                f"{feature_cls.__name__} must inherit from BaseFeature"
            )

        feature_name = getattr(feature_cls, "name", feature_cls.__name__)

        if feature_name in cls._registry:
            raise ValueError(
                f"Feature '{feature_name}' already registered"
            )

        cls._registry[feature_name] = feature_cls

        logger.debug("Registered feature: %s", feature_name)

        return feature_cls

    # ------------------------------------------------------------------
    # Retrieval Methods
    # ------------------------------------------------------------------

    @classmethod
    def get_feature(cls, name: str) -> Type[BaseFeature]:
        """
        Retrieve a registered feature class.

        Parameters
        ----------
        name : str
            Feature name.

        Returns
        -------
        Type[BaseFeature]

        Raises
        ------
        KeyError
            If feature is not registered.
        """

        if name not in cls._registry:
            raise KeyError(f"Feature '{name}' not found in registry")

        return cls._registry[name]

    @classmethod
    def create_feature(cls, name: str, **kwargs) -> BaseFeature:
        """
        Instantiate a feature from the registry.

        Parameters
        ----------
        name : str
            Registered feature name.
        kwargs : dict
            Parameters passed to feature constructor.

        Returns
        -------
        BaseFeature
        """

        feature_cls = cls.get_feature(name)

        feature = feature_cls(**kwargs)

        logger.debug("Instantiated feature: %s", name)

        return feature

    # ------------------------------------------------------------------
    # Registry Inspection
    # ------------------------------------------------------------------

    @classmethod
    def list_features(cls) -> List[str]:
        """
        List all registered features.

        Returns
        -------
        List[str]
        """

        return sorted(cls._registry.keys())

    @classmethod
    def has_feature(cls, name: str) -> bool:
        """
        Check if a feature exists in registry.

        Parameters
        ----------
        name : str

        Returns
        -------
        bool
        """

        return name in cls._registry

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear the registry (mainly used in tests).
        """

        cls._registry.clear()
        logger.warning("Feature registry cleared")

    # ------------------------------------------------------------------
    # Bulk Registration
    # ------------------------------------------------------------------

    @classmethod
    def register_many(cls, features: List[Type[BaseFeature]]) -> None:
        """
        Register multiple feature classes.

        Parameters
        ----------
        features : List[Type[BaseFeature]]
        """

        for feature in features:
            cls.register(feature)

    # ------------------------------------------------------------------
    # Debug Utilities
    # ------------------------------------------------------------------

    @classmethod
    def describe_registry(cls) -> Dict[str, str]:
        """
        Return registry metadata useful for debugging.

        Returns
        -------
        Dict[str, str]
        """

        description = {}

        for name, feature_cls in cls._registry.items():
            description[name] = feature_cls.__module__

        return description


# ----------------------------------------------------------------------
# Decorator for Automatic Registration
# ----------------------------------------------------------------------

def register_feature(feature_cls: Type[BaseFeature]) -> Type[BaseFeature]:
    """
    Decorator used for automatic feature registration.

    Example
    -------
    @register_feature
    class BiasFeature(BaseFeature):
        ...
    """

    return FeatureRegistry.register(feature_cls)