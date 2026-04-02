"""
File Name: feature_config.py
Module: Feature Engineering Configuration
Description:
    Defines configuration structures and validation utilities for the
    TruthLens feature extraction system. This module enables configuration-
    driven feature activation, parameterization, and grouping using
    dataclasses compatible with YAML configuration files.

    The configuration layer integrates with the FeatureRegistry to build
    feature pipelines dynamically and ensures consistent, validated feature
    initialization across experiments and production inference systems.

Dependencies:
    dataclasses
    typing
    logging

Inputs:
    YAML configuration dictionaries loaded by config_loader

Outputs:
    Validated FeatureConfig objects used by feature pipelines
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.features.base.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """
    Defines configuration for a single feature extractor.
    """

    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate the feature definition.
        """

        if not self.name:
            raise ValueError("FeatureDefinition must include a feature name")

        if not isinstance(self.params, dict):
            raise ValueError(
                f"Feature '{self.name}' params must be a dictionary"
            )

        if not FeatureRegistry.has_feature(self.name):
            raise ValueError(
                f"Feature '{self.name}' is not registered in FeatureRegistry"
            )


@dataclass
class FeatureGroupConfig:
    """
    Represents a logical group of features.

    Example groups:
    - bias
    - emotion
    - narrative
    - propaganda
    """

    group_name: str
    enabled: bool = True
    features: List[FeatureDefinition] = field(default_factory=list)

    def validate(self) -> None:
        """
        Validate the group configuration.
        """

        if not self.group_name:
            raise ValueError("FeatureGroupConfig must define group_name")

        for feature in self.features:
            feature.validate()


@dataclass
class FeaturePipelineConfig:
    """
    Top-level configuration controlling the entire feature pipeline.
    """

    groups: List[FeatureGroupConfig] = field(default_factory=list)
    global_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate pipeline configuration.
        """

        if not isinstance(self.groups, list):
            raise ValueError("FeaturePipelineConfig.groups must be a list")

        for group in self.groups:
            group.validate()

    def enabled_features(self) -> List[str]:
        """
        Return names of all enabled features.
        """

        enabled = []

        for group in self.groups:
            if not group.enabled:
                continue

            for feature in group.features:
                if feature.enabled:
                    enabled.append(feature.name)

        return enabled

    def feature_parameters(self, feature_name: str) -> Dict[str, Any]:
        """
        Retrieve parameters for a specific feature.

        Parameters
        ----------
        feature_name : str

        Returns
        -------
        Dict[str, Any]
        """

        for group in self.groups:
            for feature in group.features:
                if feature.name == feature_name:
                    return feature.params

        raise KeyError(f"No parameters defined for feature '{feature_name}'")


class FeatureConfigLoader:
    """
    Utility class responsible for converting raw dictionaries
    (usually loaded from YAML files) into validated FeaturePipelineConfig
    objects.
    """

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> FeaturePipelineConfig:
        """
        Construct FeaturePipelineConfig from a dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]

        Returns
        -------
        FeaturePipelineConfig
        """

        if "groups" not in config_dict:
            raise ValueError("Feature config must contain 'groups' field")

        groups: List[FeatureGroupConfig] = []

        for group_data in config_dict["groups"]:

            feature_defs: List[FeatureDefinition] = []

            for feature_data in group_data.get("features", []):
                feature_def = FeatureDefinition(
                    name=feature_data["name"],
                    enabled=feature_data.get("enabled", True),
                    params=feature_data.get("params", {}),
                )
                feature_defs.append(feature_def)

            group = FeatureGroupConfig(
                group_name=group_data["group_name"],
                enabled=group_data.get("enabled", True),
                features=feature_defs,
            )

            groups.append(group)

        pipeline_config = FeaturePipelineConfig(
            groups=groups,
            global_params=config_dict.get("global_params", {}),
        )

        pipeline_config.validate()

        logger.info(
            "Loaded feature configuration with %d groups and %d enabled features",
            len(groups),
            len(pipeline_config.enabled_features()),
        )

        return pipeline_config