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

    Explicit integration of bias, framing, and ideological feature extractors:

        BiasFeatures (src.features.bias.bias_features)
            10 features: bias_loaded_language_ratio, bias_subjective_ratio,
            bias_uncertainty_ratio, bias_polarization_ratio,
            bias_evaluative_ratio, bias_phrase_count, bias_exclamation_density,
            bias_caps_ratio, bias_intensity, bias_diversity

        FramingFeatures (src.features.bias.framing_features)
            10 features: frame_economic_ratio, frame_moral_ratio,
            frame_security_ratio, frame_human_interest_ratio,
            frame_conflict_ratio, frame_phrase_count, frame_quote_density,
            frame_diversity, frame_dominance, frame_entropy

        IdeologicalFeatures (src.features.bias.ideological_features)
            8 features: ideology_left_ratio, ideology_right_ratio,
            ideology_balance, ideology_entropy, ideology_polarization_ratio,
            ideology_group_reference_ratio, ideology_phrase_count,
            ideology_signal_strength

    Use build_bias_schema() to create a FeaturePreparationConfig whose
    feature_schema contains all 28 bias-module features. Use
    prepare_from_text() to run all three extractors on raw text and
    return a model-ready tensor in one call.

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
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from src.graph.graph_pipeline import GraphPipeline

from src.features.base.base_feature import FeatureContext
from src.features.bias.bias_features import BiasFeatures
from src.features.bias.framing_features import FramingFeatures
from src.features.bias.ideological_features import IdeologicalFeatures
from src.features.dataset_feature_generator import DatasetFeatureGenerator
from src.features.feature_schema_validator import FeatureSchemaValidator
from src.features.feature_statistics import FeatureStatistics
from src.features.pipelines.feature_pipeline import ALL_BIAS_MODULE_FEATURE_NAMES

logger = logging.getLogger(__name__)


def _prepare_flat_features_batch(features: Dict[str, Any]) -> Dict[str, float]:
    if all(isinstance(value, (int, float)) for value in features.values()):
        return {
            key: float(value)
            for key, value in features.items()
            if key != "text"
        }

    flat: Dict[str, float] = {}
    stack = list(features.items())
    pop = stack.pop
    while stack:
        key, value = pop()
        if key == "text":
            continue
        if isinstance(value, (int, float)):
            flat[key] = float(value)
        elif isinstance(value, (list, tuple, set)):
            flat[f"{key}_count"] = float(len(value))
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                clean_key = str(sub_key).strip().replace(" ", "_")
                next_prefix = f"{key}_{clean_key}" if key else clean_key
                stack.append((next_prefix, sub_value))

    return flat


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

    Bias module integration:
        BiasFeatures, FramingFeatures, and IdeologicalFeatures are directly
        imported. Their output keys are used by build_bias_schema() to
        construct a 28-feature schema. prepare_from_text() runs all three
        extractors on raw text and returns a model-ready tensor without
        requiring an upstream feature pipeline.
    """

    def __init__(
        self,
        config: FeaturePreparationConfig,
        scaler: Optional[Any] = None,
        selector: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.scaler = scaler
        self.selector = selector
        self.graph_pipeline: GraphPipeline | None = None
        self._pool: Optional[mp.pool.Pool] = None

        if not config.feature_schema:
            raise ValueError("Feature schema cannot be empty")

        self.feature_dim = len(config.feature_schema)
        self.feature_index = {name: idx for idx, name in enumerate(config.feature_schema)}
        self.schema_validator = FeatureSchemaValidator(
            expected_features=config.feature_schema,
            strict=False,
            allow_missing=True,
            allow_extra=True,
        )
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

    def _get_pool(self) -> mp.pool.Pool:
        if self._pool is None:
            try:
                ctx = mp.get_context("fork")
            except ValueError:
                ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(4)
        return self._pool

    def __del__(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Schema builders
    # -----------------------------------------------------------------------

    @classmethod
    def build_bias_schema(cls) -> "FeaturePreparationConfig":
        """
        Build a FeaturePreparationConfig whose schema covers all 28 features
        produced by the three bias module extractors:

            BiasFeatures       → 10 features (bias_*)
            FramingFeatures    → 10 features (frame_*)
            IdeologicalFeatures → 8 features (ideology_*)

        Returns a config instance ready for use in FeaturePreparer().

        Example
        -------
        config = FeaturePreparer.build_bias_schema()
        preparer = FeaturePreparer(config)
        tensor = preparer.prepare_from_text("Breaking news: ...")
        """
        return FeaturePreparationConfig(
            feature_schema=ALL_BIAS_MODULE_FEATURE_NAMES,
            apply_scaling=False,
            apply_feature_selection=False,
            return_tensor=True,
            derive_graph_features=False,
        )

    # -----------------------------------------------------------------------
    # Core preparation helpers
    # -----------------------------------------------------------------------

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

    def _prepare_flat_features(
        self,
        features: Dict[str, Any],
        *,
        batch_mode: bool = False,
    ) -> Dict[str, float]:
        """
        Flatten a raw feature dict into a schema-compatible float dict.

        Handles bias_*, frame_*, and ideology_* keys natively — they are
        already numeric floats produced by BiasFeatures, FramingFeatures,
        and IdeologicalFeatures respectively. Non-numeric values are
        flattened recursively; the 'text' key is skipped.

        When derive_graph_features is enabled and a 'text' key is present,
        graph features are also derived and merged under setdefault.
        """
        if all(isinstance(value, (int, float)) for value in features.values()):
            return {
                key: float(value)
                for key, value in features.items()
                if key != "text"
            }

        flat: Dict[str, float] = {}

        for key, value in features.items():
            if key == "text":
                continue
            self._flatten_numeric(str(key), value, flat)

        graph_pipeline = self.graph_pipeline
        use_graph = graph_pipeline is not None
        if use_graph and not batch_mode:
            text = features.get("text")
            if isinstance(text, str) and text.strip():
                try:
                    graph_output = graph_pipeline.run(text)

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
                except Exception:
                    pass

        for key in list(flat.keys()):
            flat[key] = float(flat[key])

        return flat

    def _dict_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """
        Convert feature dictionary to ordered feature vector.
        """
        vector = np.zeros(len(self.config.feature_schema), dtype=self.config.dtype)

        for feature_name, value in features.items():
            idx = self.feature_index.get(feature_name)
            if idx is None:
                continue
            vector[idx] = float(value)

        return vector

    def _apply_scaling(self, X: np.ndarray) -> np.ndarray:
        """
        Apply scaling transformation.
        """
        if not self.config.apply_scaling or self.scaler is None:
            return X

        try:
            try:
                return self.scaler.transform(X, copy=False)
            except TypeError:
                return self.scaler.transform(X)
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
            try:
                return self.selector.transform(X, copy=False)
            except TypeError:
                return self.selector.transform(X)
        except Exception as exc:
            logger.exception("Feature selection failed")
            raise RuntimeError("Feature selection transformation failed") from exc

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def prepare_single(self, features: Dict[str, Any]) -> np.ndarray | torch.Tensor:
        """
        Prepare a single feature dictionary.

        The input dict may contain any mix of bias_*, frame_*, ideology_*,
        and other feature keys. Keys not present in the feature schema are
        silently ignored; missing schema keys default to 0.0.

        FeatureSchemaValidator is applied to the flattened feature dict to
        ensure all values are numeric and only expected keys are present.
        Validation runs in permissive mode — missing or extra keys are allowed
        and logged rather than raised as errors.
        """
        flat_features = self._prepare_flat_features(features, batch_mode=False)
        vector = np.zeros(self.feature_dim, dtype=np.float32)
        feature_index = self.feature_index
        for key, value in flat_features.items():
            idx = feature_index.get(key)
            if idx is not None:
                vector[idx] = value
        matrix = vector[None, :]

        matrix = self._apply_scaling(matrix)
        matrix = self._apply_feature_selection(matrix)

        if self.config.return_tensor:
            if not self.config.apply_scaling and not self.config.apply_feature_selection:
                return torch.as_tensor(matrix, dtype=torch.float32).pin_memory()

            return torch.as_tensor(matrix, dtype=torch.float32).pin_memory()

        return matrix

    def prepare_from_text(
        self,
        text: str,
        include_bias: bool = True,
        include_framing: bool = True,
        include_ideology: bool = True,
    ) -> torch.Tensor | np.ndarray:
        """
        Run bias-module extractors on raw text and return a prepared tensor.

        Instantiates BiasFeatures, FramingFeatures, and IdeologicalFeatures
        directly, runs them on the provided text, merges their outputs, and
        calls prepare_single() on the combined feature dict.

        This is the most direct path from raw text to a model-ready tensor
        when only bias-module features are needed — no FeaturePipeline or
        BatchFeaturePipeline required.

        Parameters
        ----------
        text : str
            Raw article text.
        include_bias : bool
            Whether to run BiasFeatures (bias_*).
        include_framing : bool
            Whether to run FramingFeatures (frame_*).
        include_ideology : bool
            Whether to run IdeologicalFeatures (ideology_*).

        Returns
        -------
        torch.Tensor of shape (1, n_features) when return_tensor=True,
        numpy array otherwise.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        context = FeatureContext(text=text)
        combined: Dict[str, float] = {"text": text}

        if include_bias:
            try:
                bias_extractor = BiasFeatures()
                combined.update(bias_extractor.extract(context))
            except Exception as exc:
                logger.warning("BiasFeatures extraction failed: %s", exc)

        if include_framing:
            try:
                framing_extractor = FramingFeatures()
                combined.update(framing_extractor.extract(context))
            except Exception as exc:
                logger.warning("FramingFeatures extraction failed: %s", exc)

        if include_ideology:
            try:
                ideology_extractor = IdeologicalFeatures()
                combined.update(ideology_extractor.extract(context))
            except Exception as exc:
                logger.warning("IdeologicalFeatures extraction failed: %s", exc)

        return self.prepare_single(combined)

    def prepare_batch(
        self,
        feature_dicts: List[Dict[str, Any]],
    ) -> np.ndarray | torch.Tensor:
        """
        Prepare batch feature dictionaries.

        Each dict in the list may contain any combination of bias_*, frame_*,
        ideology_*, and other feature keys. Use prepare_from_text() for a
        single raw-text entry point to the bias module features.
        """
        if not isinstance(feature_dicts, list):
            raise TypeError("feature_dicts must be a list")

        if len(feature_dicts) == 0:
            raise ValueError("feature_dicts list cannot be empty")

        rows = len(feature_dicts)
        feature_dim = self.feature_dim
        dtype = np.float32 if self.config.dtype == "float32" else np.float16
        matrix = np.zeros((rows, feature_dim), dtype=dtype)
        feature_index = self.feature_index
        get_idx = feature_index.get

        if rows < 32:
            flat_list = [_prepare_flat_features_batch(item) for item in feature_dicts]
        else:
            pool = self._get_pool()
            flat_list = pool.map(_prepare_flat_features_batch, feature_dicts)

        for i, flat_features in enumerate(flat_list):
            for key, value in flat_features.items():
                idx = get_idx(key)
                if idx is not None:
                    matrix[i, idx] = value

        scaler = self.scaler
        selector = self.selector

        if scaler is not None:
            try:
                matrix = scaler.transform(matrix, copy=False)
            except TypeError:
                matrix = scaler.transform(matrix)

        if selector is not None:
            try:
                matrix = selector.transform(matrix, copy=False)
            except TypeError:
                matrix = selector.transform(matrix)

        if self.config.return_tensor:
            return torch.from_numpy(matrix).pin_memory()

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

    def generate_from_texts(
        self,
        texts: List[str],
        batch_pipeline: Any,
    ) -> "np.ndarray | torch.Tensor":
        """
        Generate model-ready feature matrix directly from raw texts.

        Uses DatasetFeatureGenerator to extract a feature matrix from the
        provided texts, then calls prepare_batch() to apply schema ordering,
        scaling, and selection.

        Parameters
        ----------
        texts : List[str]
            Raw article texts.
        batch_pipeline : BatchFeaturePipeline
            An initialized or uninitialised BatchFeaturePipeline instance.
            DatasetFeatureGenerator will call initialize() automatically if
            needed.

        Returns
        -------
        np.ndarray or torch.Tensor of shape (n_samples, n_features_selected).
        """
        if not texts:
            raise ValueError("texts must not be empty")

        generator = DatasetFeatureGenerator(pipeline=batch_pipeline)
        _, feature_names = generator.generate(texts)

        contexts = generator._build_contexts(texts)
        feature_dicts = batch_pipeline._sequential_extract(contexts)

        logger.info(
            "generate_from_texts | samples=%d features=%d",
            len(feature_dicts),
            len(feature_names),
        )

        return self.prepare_batch(feature_dicts)

    def compute_feature_statistics(
        self,
        feature_dicts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute descriptive statistics for a batch of feature dictionaries.

        Uses FeatureStatistics to produce a dataset-level summary and detect
        constant (zero-variance) features. Also validates each dict against the
        configured schema via FeatureSchemaValidator.

        Parameters
        ----------
        feature_dicts : List[Dict[str, Any]]
            Feature dictionaries as produced by the upstream feature pipeline.

        Returns
        -------
        Dict[str, Any] with keys:
            "summary"           — dataset-level statistics (mean_variance, etc.)
            "constant_features" — list of zero-variance feature names
            "schema_summary"    — feature schema metadata
        """
        if not feature_dicts:
            raise ValueError("feature_dicts must not be empty")

        flat_dicts = [self._prepare_flat_features(f) for f in feature_dicts]

        stats = FeatureStatistics()
        summary = stats.dataset_summary(flat_dicts)
        constant = stats.detect_constant_features(flat_dicts)

        logger.info(
            "Feature statistics | samples=%d features=%d "
            "mean_variance=%.6f constant=%d",
            int(summary["num_samples"]),
            int(summary["num_features"]),
            summary["mean_variance"],
            len(constant),
        )

        try:
            self.schema_validator.validate_batch(flat_dicts)
        except Exception as _val_exc:
            logger.warning("Schema validation warning: %s", _val_exc)

        return {
            "summary": summary,
            "constant_features": constant,
            "schema_summary": self.schema_validator.schema_summary(),
        }
