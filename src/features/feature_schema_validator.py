"""
File Name: feature_schema_validator.py
Module: Feature Engineering - Schema Validation
Description:
    Validates feature dictionaries produced by the TruthLens feature
    pipeline against an expected schema. The validator ensures that:

        • required features are present
        • feature values are numeric
        • feature keys match the defined schema
        • missing or unexpected features are detected
        • feature order consistency is maintained for ML pipelines

    The validator supports both strict and permissive validation modes
    and is intended to safeguard training and inference pipelines against
    schema drift or corrupted feature outputs.

Dependencies:
    dataclasses
    typing
    logging
    numbers

Inputs:
    Feature dictionaries (Dict[str, float])
    Expected feature schema (List[str])

Outputs:
    Validated feature dictionaries
"""

from __future__ import annotations

import logging
import numbers
from dataclasses import dataclass, field
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]


@dataclass
class FeatureSchemaValidator:
    """
    Validates feature vectors against a defined schema.
    """

    expected_features: List[str]
    strict: bool = True
    allow_missing: bool = False
    allow_extra: bool = False

    _expected_set: Set[str] = field(init=False)

    def __post_init__(self) -> None:
        if not self.expected_features:
            raise ValueError("Expected feature schema cannot be empty")

        self._expected_set = set(self.expected_features)

        logger.info(
            "FeatureSchemaValidator initialized | features=%d strict=%s",
            len(self.expected_features),
            self.strict,
        )

    def validate(self, features: FeatureVector) -> FeatureVector:
        """
        Validate a single feature dictionary.
        """

        if not isinstance(features, dict):
            raise TypeError("Features must be a dictionary")

        feature_keys = set(features.keys())

        missing = self._expected_set - feature_keys
        extra = feature_keys - self._expected_set

        if missing and not self.allow_missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")

        if extra and not self.allow_extra:
            raise ValueError(f"Unexpected extra features: {sorted(extra)}")

        validated: FeatureVector = {}

        for key in self.expected_features:

            value = features.get(key, 0.0)

            if not isinstance(value, numbers.Number):
                raise TypeError(f"Feature '{key}' must be numeric")

            validated[key] = float(value)

        return validated

    def validate_batch(
        self,
        feature_list: List[FeatureVector],
    ) -> List[FeatureVector]:
        """
        Validate multiple feature vectors.
        """

        if not feature_list:
            raise ValueError("Feature list cannot be empty")

        validated = []

        for idx, fv in enumerate(feature_list):

            try:
                validated.append(self.validate(fv))

            except Exception:  # noqa: BLE001
                logger.error("Feature validation failed at index %d", idx)

                if self.strict:
                    raise

                validated.append({})

        logger.info(
            "Batch feature validation completed | samples=%d",
            len(validated),
        )

        return validated

    def enforce_order(self, features: FeatureVector) -> List[float]:
        """
        Convert validated feature dictionary into ordered feature vector.
        """

        validated = self.validate(features)

        return [validated[key] for key in self.expected_features]

    def enforce_order_batch(
        self,
        feature_list: List[FeatureVector],
    ) -> List[List[float]]:
        """
        Convert batch of feature dictionaries into ordered vectors.
        """

        return [self.enforce_order(f) for f in feature_list]

    def schema_summary(self) -> Dict[str, int]:
        """
        Return metadata about the feature schema.
        """

        return {
            "num_features": len(self.expected_features),
        }
