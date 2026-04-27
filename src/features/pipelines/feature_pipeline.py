from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch

from src.features.base.feature_registry import FeatureRegistry
from src.features.feature_bootstrap import bootstrap_feature_registry
from src.features.fusion.feature_fusion import FeatureFusion
from src.features.feature_schema_validator import FeatureSchemaValidator
from src.graph.graph_pipeline import GraphPipeline, get_default_pipeline
from src.features.base.base_feature import FeatureContext
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)


# =========================================================
# PARTITIONING
# =========================================================

_TEXT_PREFIXES = (
    "embedding_", "vocabulary_", "hapax_", "token_",
    "unique_token_", "type_token_", "avg_token_", "max_token_",
    "repetition_", "sentence_", "avg_sentence_",
    "noun_", "verb_", "adjective_", "adverb_",
    "punctuation_", "lexical_", "average_word_",
)


def partition_feature_sections(features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    sections = {
        "bias": {},
        "framing": {},
        "ideology": {},
        "emotion": {},
        "discourse": {},
        "graph": {},
        "narrative": {},
        "propaganda": {},
        "text": {},
        "other": {},
    }

    for k, v in features.items():
        if k.startswith("bias_"):
            sections["bias"][k] = v
        elif k.startswith("frame_"):
            sections["framing"][k] = v
        elif k.startswith("ideology_"):
            sections["ideology"][k] = v
        elif k.startswith(("emotion_", "lexicon_emotion_")):
            sections["emotion"][k] = v
        elif k.startswith(("discourse_", "argument_")):
            sections["discourse"][k] = v
        elif k.startswith(("entity_", "interaction_", "graph_", "graph_pipeline_")):
            sections["graph"][k] = v
        elif k.startswith(("narrative_", "conflict_")):
            sections["narrative"][k] = v
        elif k.startswith(("propaganda_", "manipulation_")):
            sections["propaganda"][k] = v
        elif k.startswith(_TEXT_PREFIXES):
            sections["text"][k] = v
        else:
            sections["other"][k] = v

    return sections


# =========================================================
# PIPELINE
# =========================================================

@dataclass
class FeaturePipeline:

    validator: Optional[FeatureSchemaValidator] = None

    features: List = field(default_factory=list)
    fusion: Optional[FeatureFusion] = None
    graph_pipeline: Optional[GraphPipeline] = None

    model: Optional[torch.nn.Module] = None

    _initialized: bool = False

    # Audit fix #1.8 — surface graph-merge failures instead of swallowing
    # them at debug level. Counter is exposed for metrics scrape.
    graph_merge_failures: int = 0

    # -----------------------------------------------------

    def initialize(self):

        if self._initialized:
            return

        bootstrap_feature_registry()

        # CUDA optimizations
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            except Exception:
                logger.debug("Flash attention setup skipped")

        feature_names = FeatureRegistry.list_features()

        self.features = [
            FeatureRegistry.create_feature(name)
            for name in feature_names
        ]

        self.fusion = FeatureFusion(self.features)

        try:
            # G-R1: shared singleton — first instantiation pays the
            # full setup cost, subsequent ``FeaturePipeline``s reuse it.
            self.graph_pipeline = get_default_pipeline()
        except Exception as e:
            logger.warning("GraphPipeline unavailable: %s", e)
            self.graph_pipeline = None

        # Model optimization (optional)
        if self.model is not None:
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled")
            except Exception:
                logger.debug("torch.compile skipped")

            try:
                if hasattr(self.model, "gradient_checkpointing_enable"):
                    self.model.gradient_checkpointing_enable()
            except Exception:
                logger.debug("Gradient checkpointing skipped")

        self._initialized = True

        logger.info("FeaturePipeline initialized | features=%d", len(self.features))

    # -----------------------------------------------------

    def _merge_graph_features(self, ctx: FeatureContext, features: Dict[str, float]) -> None:
        """
        Merge graph features into feature dict (cached).
        """

        if not self.graph_pipeline:
            return

        try:
            cache = ctx.cache.setdefault("_graph", {})

            if "output" not in cache:
                cache["output"] = self.graph_pipeline.run(ctx.text)

            graph_output = cache["output"]

            # Core graph
            for k, v in graph_output.get("graph_features", {}).items():
                if isinstance(v, (int, float)):
                    features[k] = float(v)

            # Entity metrics
            for k, v in graph_output.get("entity_graph_metrics", {}).items():
                if isinstance(v, (int, float)):
                    features[f"graph_pipeline_entity_{k}"] = float(v)

            # Narrative metrics
            for k, v in graph_output.get("narrative_graph_metrics", {}).items():
                if isinstance(v, (int, float)):
                    features[f"graph_pipeline_narrative_{k}"] = float(v)

        except Exception as e:
            # Audit fix #1.8 — broken graph pipeline used to log at
            # debug level and yield an empty graph-features sub-dict;
            # downstream model then saw all-zero graph signal and
            # treated it as legitimate "no signal" instead of a bug.
            self.graph_merge_failures += 1
            logger.warning(
                "Graph merge failed (count=%d): %s",
                self.graph_merge_failures, e,
            )
            if os.getenv("TRUTHLENS_STRICT_FEATURES"):
                raise

    # -----------------------------------------------------

    def extract(self, ctx: FeatureContext) -> Dict[str, float]:

        if not self._initialized:
            self.initialize()

        # Tokenize ONCE at the top of the pipeline. Every downstream
        # extractor reads `ctx.tokens_word` instead of re-running the
        # same regex against the same text.
        ensure_tokens_word(ctx)

        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    features = self.fusion.extract(ctx)
            else:
                features = self.fusion.extract(ctx)

        # Graph merge
        self._merge_graph_features(ctx, features)

        return features

    # -----------------------------------------------------

    def batch_extract(self, contexts: List[FeatureContext]):

        if not contexts:
            return []

        if not self._initialized:
            self.initialize()

        #  Shared batch cache (NEW)
        shared_cache: Dict[str, Any] = {}

        for ctx in contexts:
            if not isinstance(ctx.cache, dict):
                ctx.cache = {}
            ctx.shared = shared_cache
            # Tokenize ONCE per context at the top of the pipeline.
            ensure_tokens_word(ctx)

        # Vectorized per-feature dispatch through fusion.extract_batch
        # (lexicon extractors override extract_batch for ~10-50x speedup).
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast("cuda"):
                    batch_features = self.fusion.extract_batch(contexts)
            else:
                batch_features = self.fusion.extract_batch(contexts)

        # Graph features remain per-sample (graph build is text-dependent
        # and the per-sample cache already prevents recomputation).
        for ctx, features in zip(contexts, batch_features):
            self._merge_graph_features(ctx, features)

        return batch_features

    # -----------------------------------------------------

    def process(self, contexts: List[FeatureContext]):

        features = self.batch_extract(contexts)

        if self.validator:
            features = self.validator.validate_batch(features)

        return features

# =========================================================
# Backward-compat re-exports (canonical source: feature_schema.py)
# =========================================================
from src.features.feature_schema import (
    BIAS_FEATURES as BIAS_FEATURE_NAMES,
    FRAMING_FEATURES as FRAMING_FEATURE_NAMES,
    IDEOLOGICAL_FEATURES as IDEOLOGICAL_FEATURE_NAMES,
)

# Union of every bias-adjacent feature group, as expected by
# src.inference.feature_preparer.
ALL_BIAS_MODULE_FEATURE_NAMES: list[str] = sorted(
    set(BIAS_FEATURE_NAMES + FRAMING_FEATURE_NAMES + IDEOLOGICAL_FEATURE_NAMES)
)

