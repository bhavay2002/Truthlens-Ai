"""
File Name: feature_preparer.py
Module: Feature Preparation Pipeline
Description:
    Converts extracted dictionary-based feature representations into
    model-ready numeric vectors used during inference and training.

    The module guarantees strict consistency between training and inference
    feature ordering and preprocessing by applying the same feature schema,
    scaling pipeline, and feature selection pipeline used during model
    training.

    Processing pipeline:

        raw feature dict
            ↓
        ordered feature vector (schema-aligned)
            ↓
        numpy feature matrix
            ↓
        scaling transformation
            ↓
        feature selection
            ↓
        model-ready feature tensor

    Designed for production ML systems where reproducibility and deterministic
    feature pipelines are required.

Dependencies:
    logging
    typing
    dataclasses
    numpy
    torch
Inputs:
    Feature dictionaries extracted from upstream feature pipelines.
Outputs:
    Model-ready feature arrays or tensors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from src.graph.graph_pipeline import GraphPipeline

logger = logging.getLogger(__name__)


@dataclass
class FeaturePreparationConfig:
    """
    Configuration for feature preparation pipeline.
    """
    feature_schema: List[str]
    apply_scaling: bool = True
    apply_feature_selection: bool = True
    return_tensor: bool = True
    dtype: str = "float32"
    derive_graph_features: bool = True


class FeaturePreparer:
    """
    Responsible for transforming extracted features into model-ready format.

    Responsibilities:
    - enforce deterministic feature ordering
    - convert dictionaries to numeric vectors
    - apply scaling transformation
    - apply feature selection
    - validate feature integrity
    """

    def __init__(
        self,
        config: FeaturePreparationConfig,
        scaler: Optional[Any] = None,
        selector: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.config = config
        self.scaler = scaler
        self.selector = selector
        self.device = device
        self.graph_pipeline: GraphPipeline | None = None

        if not config.feature_schema:
            raise ValueError("Feature schema cannot be empty")

        self.feature_index = {name: idx for idx, name in enumerate(config.feature_schema)}
        if self.config.derive_graph_features:
            try:
                self.graph_pipeline = GraphPipeline()
            except Exception as exc:  # noqa: BLE001
                logger.warning("GraphPipeline unavailable in FeaturePreparer: %s", exc)
                self.graph_pipeline = None

        logger.info(
            "FeaturePreparer initialized with %d features",
            len(self.config.feature_schema),
        )

    def _validate_feature_dict(self, features: Dict[str, Any]) -> None:
        """
        Validate input feature dictionary.
        """
        if not isinstance(features, dict):
            raise TypeError("Features must be a dictionary")

        for key, value in features.items():
            if not isinstance(key, str):
                raise TypeError("Feature keys must be strings")

            if not isinstance(value, (int, float)):
                raise TypeError(f"Feature value must be numeric: {key}")

    def _flatten_numeric(self, prefix: str, value: Any, output: Dict[str, float]) -> None:
        if isinstance(value, (int, float)):
            output[prefix] = float(value)
            return

        if isinstance(value, (list, tuple, set)):
            output[f"{prefix}_count"] = float(len(value))
            return

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                clean_key = str(sub_key).strip().replace(" ", "_")
                next_prefix = f"{prefix}_{clean_key}" if prefix else clean_key
                self._flatten_numeric(next_prefix, sub_value, output)

    def _prepare_flat_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        flat: Dict[str, float] = {}

        for key, value in features.items():
            if key == "text":
                continue
            self._flatten_numeric(str(key), value, flat)

        if self.graph_pipeline is not None:
            text = features.get("text")
            if isinstance(text, str) and text.strip():
                try:
                    graph_output = self.graph_pipeline.run(text)

                    graph_features = graph_output.get("graph_features", {})
                    if isinstance(graph_features, dict):
                        for k, v in graph_features.items():
                            if isinstance(v, (int, float)):
                                flat.setdefault(str(k), float(v))

                    entity_metrics = graph_output.get("entity_graph_metrics", {})
                    if isinstance(entity_metrics, dict):
                        for k, v in entity_metrics.items():
                            if isinstance(v, (int, float)):
                                flat.setdefault(f"graph_pipeline_{k}", float(v))

                    narrative_metrics = graph_output.get("narrative_graph_metrics", {})
                    if isinstance(narrative_metrics, dict):
                        for k, v in narrative_metrics.items():
                            if isinstance(v, (int, float)):
                                flat.setdefault(f"graph_pipeline_{k}", float(v))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Graph feature derivation skipped: %s", exc)

        return flat

    def _dict_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to ordered feature vector.
        """
        vector = np.zeros(len(self.config.feature_schema), dtype=self.config.dtype)

        for feature_name, value in features.items():
            if feature_name not in self.feature_index:
                logger.debug("Ignoring unknown feature: %s", feature_name)
                continue

            idx = self.feature_index[feature_name]
            vector[idx] = float(value)

        return vector

    def _apply_scaling(self, X: np.ndarray) -> np.ndarray:
        """
        Apply scaling transformation.
        """
        if not self.config.apply_scaling or self.scaler is None:
            return X

        try:
            X_scaled = self.scaler.transform(X)
            return X_scaled
        except Exception as exc:
            logger.exception("Scaling transformation failed")
            raise RuntimeError("Feature scaling failed") from exc

    def _apply_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """
        Apply feature selection transformation.
        """
        if not self.config.apply_feature_selection or self.selector is None:
            return X

        try:
            X_selected = self.selector.transform(X)
            return X_selected
        except Exception as exc:
            logger.exception("Feature selection failed")
            raise RuntimeError("Feature selection transformation failed") from exc

    def prepare_single(self, features: Dict[str, Any]) -> np.ndarray | torch.Tensor:
        """
        Prepare a single feature dictionary.
        """
        flat_features = self._prepare_flat_features(features)
        self._validate_feature_dict(flat_features)

        vector = self._dict_to_vector(flat_features)
        matrix = vector.reshape(1, -1)

        matrix = self._apply_scaling(matrix)
        matrix = self._apply_feature_selection(matrix)

        if self.config.return_tensor:
            tensor = torch.tensor(matrix, dtype=torch.float32)

            if self.device is not None:
                tensor = tensor.to(self.device)

            return tensor

        return matrix

    def prepare_batch(
        self,
        feature_dicts: List[Dict[str, Any]],
    ) -> np.ndarray | torch.Tensor:
        """
        Prepare batch feature dictionaries.
        """
        if not isinstance(feature_dicts, list):
            raise TypeError("feature_dicts must be a list")

        if len(feature_dicts) == 0:
            raise ValueError("feature_dicts list cannot be empty")

        vectors: List[np.ndarray] = []

        for features in feature_dicts:
            flat_features = self._prepare_flat_features(features)
            self._validate_feature_dict(flat_features)
            vec = self._dict_to_vector(flat_features)
            vectors.append(vec)

        matrix = np.vstack(vectors)

        matrix = self._apply_scaling(matrix)
        matrix = self._apply_feature_selection(matrix)

        if self.config.return_tensor:
            tensor = torch.tensor(matrix, dtype=torch.float32)

            if self.device is not None:
                tensor = tensor.to(self.device)

            return tensor

        return matrix

    def get_feature_schema(self) -> List[str]:
        """
        Return feature schema used for ordering.
        """
        return self.config.feature_schema

    def feature_dimension(self) -> int:
        """
        Return final feature dimension after selection.
        """
        dummy = np.zeros((1, len(self.config.feature_schema)))

        dummy = self._apply_scaling(dummy)
        dummy = self._apply_feature_selection(dummy)

        return dummy.shape[1]
