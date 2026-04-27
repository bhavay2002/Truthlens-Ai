"""Canonical multi-task TruthLens model.

A single shared transformer encoder feeds a ``ModuleDict`` of task heads
(``ClassificationHead`` for multi-class targets, ``MultiLabelHead`` for
multi-label targets). The model can be constructed three ways:

  1. ``MultiTaskTruthLensModel(encoder, task_heads)`` — raw modules.
  2. ``MultiTaskTruthLensModel(config=MultiTaskTruthLensConfig(...))``
     — convenience path that builds both the encoder and a default set
     of TruthLens task heads from a small dataclass config. Used by the
     audit-issue tests and by `TruthLensMultiTaskModel`.
  3. ``MultiTaskTruthLensModel.from_model_config(MultiTaskModelConfig)``
     — full-fidelity path driven by the YAML-backed
     :class:`~src.models.config.MultiTaskModelConfig`. Used by the
     model registry and the inference engine.

The forward pass returns a dict that contains BOTH:

  * a per-task entry, e.g. ``outputs["bias"] = {"logits": ..., ...}``
    (the contract the test-evaluation pipeline relies on); and
  * an ``outputs["task_logits"]`` mapping ``{task_name: tensor}``
    (the contract the training step / loss / evaluation engines rely on).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# CANONICAL TASK METADATA
# =========================================================

# A single source of truth for the default TruthLens head sizes and
# label vocabularies. ``MultiTaskTruthLensConfig.task_num_labels`` may
# override these per-task; everything below is just the fallback.

_DEFAULT_TASK_SPEC: Dict[str, Dict[str, Any]] = {
    "bias": {
        "task_type": "multi_class",
        "labels": ["non_bias", "bias"],
    },
    "ideology": {
        "task_type": "multi_class",
        "labels": ["left", "center", "right"],
    },
    "propaganda": {
        "task_type": "multi_class",
        "labels": ["non_propaganda", "propaganda"],
    },
    "narrative": {
        "task_type": "multi_label",
        "labels": ["hero", "villain", "victim"],
    },
    "narrative_frame": {
        "task_type": "multi_label",
        "labels": ["RE", "HI", "CO", "MO", "EC"],
    },
    "emotion": {
        "task_type": "multi_label",
        "labels": [f"emotion_{i}" for i in range(20)],
    },
}


# =========================================================
# CONFIG
# =========================================================

@dataclass
class MultiTaskTruthLensConfig:
    """Lightweight config for the convenience-construction path.

    .. note::
       **A6 — naming caveat.** The TruthLens codebase has *two*
       multi-task configuration objects and they are NOT interchangeable:

         * :class:`MultiTaskTruthLensConfig` (this class) — a
           narrow, hand-tuned dataclass used by the convenience
           constructor ``MultiTaskTruthLensModel(config=...)``. Builds
           the canonical TruthLens task set with default head sizes;
           cannot express per-task type / loss / regression overrides.
         * :class:`~src.models.config.MultiTaskModelConfig` — the
           YAML-backed, fully-structured config used by the model
           registry, the inference engine and the training pipeline
           via :meth:`MultiTaskTruthLensModel.from_model_config`.

       Mixing them (e.g. passing a ``MultiTaskModelConfig`` as
       ``config=`` here) raises a ``TypeError`` rather than silently
       constructing the wrong model. Unknown keyword arguments are
       rejected by the underlying dataclass init.
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None

    # When True we build the underlying transformer from a HF *config*
    # rather than downloading pretrained weights — useful for fast unit
    # tests that just want a forward-pass-compatible model.
    init_from_config_only: bool = False

    # Per-task loss weights (consumed by downstream training code).
    bias_weight: float = 1.0
    ideology_weight: float = 1.0
    propaganda_weight: float = 1.0
    narrative_weight: float = 1.0
    emotion_weight: float = 1.0

    # Optional overrides for the default per-task head sizes.
    task_num_labels: Optional[Dict[str, int]] = None

    # Optional restriction to a subset of the canonical task list.
    enabled_tasks: Optional[List[str]] = None

    # Reserved for future per-task knobs (kept for forward-compat with
    # the old free-form `**kwargs` shim).
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# MULTI-TASK TRUTHLENS MODEL
# =========================================================

class MultiTaskTruthLensModel(nn.Module):
    """Shared-encoder, multi-headed TruthLens model."""

    # -----------------------------------------------------
    # Class-level metadata used by downstream label-helper code
    # and by the test suite (``tests/test_multitask_label_helpers.py``).
    # These mirror ``_DEFAULT_TASK_SPEC`` above.
    # -----------------------------------------------------

    BIAS_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["bias"]["labels"])
    IDEOLOGY_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["ideology"]["labels"])
    PROPAGANDA_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["propaganda"]["labels"])
    NARRATIVE_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["narrative"]["labels"])
    FRAME_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["narrative_frame"]["labels"])
    EMOTION_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["emotion"]["labels"])

    NUM_BIAS: int = len(BIAS_LABELS)
    NUM_IDEOLOGY: int = len(IDEOLOGY_LABELS)
    NUM_PROPAGANDA: int = len(PROPAGANDA_LABELS)
    NUM_NARRATIVE: int = len(NARRATIVE_LABELS)
    NUM_NARRATIVE_FRAMES: int = len(FRAME_LABELS)
    NUM_EMOTIONS: int = len(EMOTION_LABELS)

    # =====================================================
    # CONSTRUCTION
    # =====================================================

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        task_heads: Optional[Dict[str, nn.Module]] = None,
        *,
        config: Optional[MultiTaskTruthLensConfig] = None,
    ) -> None:
        super().__init__()

        # -------------------------------------------------
        # Convenience-config construction path
        # -------------------------------------------------
        if config is not None:
            if encoder is not None or task_heads is not None:
                raise ValueError(
                    "Pass either (encoder, task_heads) or `config=`, "
                    "not both."
                )

            if not isinstance(config, MultiTaskTruthLensConfig):
                raise TypeError(
                    "config must be a MultiTaskTruthLensConfig instance "
                    f"(got {type(config).__name__})"
                )

            encoder, task_heads = self._build_from_truthlens_config(config)
            self.config = config

        else:
            self.config = None

        # -------------------------------------------------
        # Validation (shared by both paths)
        # -------------------------------------------------
        if encoder is None:
            raise ValueError("encoder is required")

        if not isinstance(task_heads, dict) or not task_heads:
            raise ValueError("task_heads must be a non-empty dict")

        self.encoder = encoder
        self.task_heads = nn.ModuleDict(task_heads)

        logger.info(
            "MultiTaskTruthLensModel initialized | tasks=%s",
            list(self.task_heads.keys()),
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, **inputs: Any) -> Dict[str, Any]:

        # -------------------------------------------------
        # ENCODER
        # -------------------------------------------------
        encoder_outputs = self.encoder(**inputs)

        pooled = self._extract_pooled(encoder_outputs)

        # -------------------------------------------------
        # TASK HEADS
        # -------------------------------------------------
        outputs: Dict[str, Any] = {}
        task_logits: Dict[str, torch.Tensor] = {}

        for task_name, head in self.task_heads.items():
            try:
                head_output = head(pooled)
            except Exception as e:
                raise RuntimeError(
                    f"Head '{task_name}' forward failed: {e}"
                ) from e

            # Task heads (ClassificationHead / MultiLabelHead) return a
            # dict containing at least a "logits" tensor; keep that
            # contract explicit so accidental tensor-only heads fail
            # loudly instead of polluting the per-task output keys.
            if isinstance(head_output, dict):
                if "logits" not in head_output:
                    raise RuntimeError(
                        f"Head '{task_name}' returned a dict without "
                        f"'logits' (keys={list(head_output)})"
                    )
                logits = head_output["logits"]
                outputs[task_name] = head_output

            elif torch.is_tensor(head_output):
                logits = head_output
                outputs[task_name] = {"logits": head_output}

            else:
                raise TypeError(
                    f"Head '{task_name}' must return a Tensor or dict "
                    f"with 'logits' (got {type(head_output).__name__})"
                )

            task_logits[task_name] = logits

        # -------------------------------------------------
        # OUTPUT
        # -------------------------------------------------
        outputs["task_logits"] = task_logits
        return outputs

    # =====================================================
    # ENCODER POOL HELPER
    # =====================================================

    @staticmethod
    def _extract_pooled(encoder_outputs: Any) -> torch.Tensor:
        """Best-effort pooled embedding extraction.

        Supports both raw ``dict`` outputs (e.g. our ``TransformerEncoder``
        wrapper, which returns ``{"pooled_output": ..., "last_hidden_state": ...}``)
        and HuggingFace-style ``ModelOutput`` objects.
        """

        if isinstance(encoder_outputs, dict):
            pooled = (
                encoder_outputs.get("pooled_output")
                or encoder_outputs.get("pooler_output")
            )
            hidden = encoder_outputs.get("last_hidden_state")
        else:
            pooled = (
                getattr(encoder_outputs, "pooled_output", None)
                or getattr(encoder_outputs, "pooler_output", None)
            )
            hidden = getattr(encoder_outputs, "last_hidden_state", None)

        if pooled is None:
            if hidden is None:
                raise RuntimeError(
                    "Encoder did not return a pooled embedding or "
                    "`last_hidden_state`; cannot feed task heads."
                )
            # Fall back to the [CLS] token for models like RoBERTa that
            # do not expose a dedicated pooler.
            pooled = hidden[:, 0]

        return pooled

    # =====================================================
    # CONSTRUCTION HELPERS
    # =====================================================

    @classmethod
    def _build_from_truthlens_config(
        cls,
        config: MultiTaskTruthLensConfig,
    ) -> "tuple[nn.Module, Dict[str, nn.Module]]":
        """Build (encoder, task_heads) from a ``MultiTaskTruthLensConfig``."""

        from src.models.encoder.encoder_config import EncoderConfig
        from src.models.encoder.encoder_factory import EncoderFactory

        encoder = EncoderFactory.create_transformer_encoder(
            EncoderConfig(
                model_type="transformer",
                model_name=config.model_name,
                pooling=config.pooling,
                device=config.device,
                init_from_config_only=config.init_from_config_only,
            )
        )

        hidden_size = int(getattr(encoder, "hidden_size"))

        task_specs = cls._resolve_task_specs(
            num_labels_overrides=config.task_num_labels,
            enabled_tasks=config.enabled_tasks,
        )

        task_heads = cls._build_default_heads(
            task_specs=task_specs,
            hidden_size=hidden_size,
            dropout=config.dropout,
        )

        return encoder, task_heads

    @classmethod
    def _resolve_task_specs(
        cls,
        num_labels_overrides: Optional[Dict[str, int]] = None,
        enabled_tasks: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Merge default task spec with optional overrides."""

        names = enabled_tasks or list(_DEFAULT_TASK_SPEC.keys())

        unknown = [n for n in names if n not in _DEFAULT_TASK_SPEC]
        if unknown:
            raise ValueError(
                f"Unknown TruthLens task name(s): {unknown}. "
                f"Known: {list(_DEFAULT_TASK_SPEC.keys())}"
            )

        resolved: Dict[str, Dict[str, Any]] = {}

        for name in names:
            base = _DEFAULT_TASK_SPEC[name]
            num_labels = (
                (num_labels_overrides or {}).get(name, len(base["labels"]))
            )
            resolved[name] = {
                "task_type": base["task_type"],
                "num_labels": int(num_labels),
            }

        return resolved

    @staticmethod
    def _build_default_heads(
        task_specs: Dict[str, Dict[str, Any]],
        hidden_size: int,
        dropout: float,
    ) -> Dict[str, nn.Module]:
        """Instantiate one head per task spec."""

        from src.models.heads.classification_head import (
            ClassificationHead,
            ClassificationHeadConfig,
        )
        from src.models.heads.multilabel_head import (
            MultiLabelHead,
            MultiLabelHeadConfig,
        )

        heads: Dict[str, nn.Module] = {}

        for name, spec in task_specs.items():
            task_type = spec["task_type"]
            num_labels = spec["num_labels"]

            if task_type == "multi_class":
                heads[name] = ClassificationHead(
                    ClassificationHeadConfig(
                        input_dim=hidden_size,
                        num_classes=num_labels,
                        dropout=dropout,
                    )
                )

            elif task_type == "multi_label":
                heads[name] = MultiLabelHead(
                    MultiLabelHeadConfig(
                        input_dim=hidden_size,
                        num_labels=num_labels,
                        dropout=dropout,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported task_type {task_type!r} for task {name!r}"
                )

        return heads

    # =====================================================
    # PUBLIC FACTORY: from MultiTaskModelConfig (YAML path)
    # =====================================================

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
    ) -> "MultiTaskTruthLensModel":
        """Build a model from a :class:`MultiTaskModelConfig`.

        This is the high-fidelity construction path used by the model
        registry, the inference engine and the YAML-driven training
        pipeline. Each entry in ``model_config.tasks`` is materialised
        as a head whose width comes from ``task_cfg.num_labels`` and
        whose head type comes from ``task_cfg.task_type``.
        """

        from src.models.config import MultiTaskModelConfig
        from src.models.encoder.encoder_factory import EncoderFactory
        from src.models.heads.classification_head import (
            ClassificationHead,
            ClassificationHeadConfig,
        )
        from src.models.heads.multilabel_head import (
            MultiLabelHead,
            MultiLabelHeadConfig,
        )

        if not isinstance(model_config, MultiTaskModelConfig):
            raise TypeError(
                "model_config must be a MultiTaskModelConfig "
                f"(got {type(model_config).__name__})"
            )

        if not model_config.tasks:
            raise ValueError("model_config.tasks must be non-empty")

        encoder = EncoderFactory.create_from_model_config(model_config)
        hidden_size = int(getattr(encoder, "hidden_size"))

        task_heads: Dict[str, nn.Module] = {}

        for task_name, task_cfg in model_config.tasks.items():
            num_labels = int(task_cfg.num_labels)

            if num_labels <= 0:
                raise ValueError(
                    f"Task {task_name!r}: num_labels must be positive "
                    f"(got {num_labels})"
                )

            if task_cfg.task_type == "multi_label":
                task_heads[task_name] = MultiLabelHead(
                    MultiLabelHeadConfig(
                        input_dim=hidden_size,
                        num_labels=num_labels,
                        dropout=model_config.dropout,
                    )
                )

            else:  # default: multi_class
                task_heads[task_name] = ClassificationHead(
                    ClassificationHeadConfig(
                        input_dim=hidden_size,
                        num_classes=num_labels,
                        dropout=model_config.dropout,
                    )
                )

        return cls(encoder=encoder, task_heads=task_heads)

    # =====================================================
    # UTILITIES
    # =====================================================

    def get_task_names(self) -> List[str]:
        return list(self.task_heads.keys())

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen")

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen")

    def freeze_heads(self) -> None:
        for head in self.task_heads.values():
            for p in head.parameters():
                p.requires_grad = False
        logger.info("All task heads frozen")

    def unfreeze_heads(self) -> None:
        for head in self.task_heads.values():
            for p in head.parameters():
                p.requires_grad = True
        logger.info("All task heads unfrozen")

    def freeze_task(self, task_name: str) -> None:
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")

        for p in self.task_heads[task_name].parameters():
            p.requires_grad = False

        logger.info("Task '%s' frozen", task_name)

    def unfreeze_task(self, task_name: str) -> None:
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")

        for p in self.task_heads[task_name].parameters():
            p.requires_grad = True

        logger.info("Task '%s' unfrozen", task_name)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def extra_repr(self) -> str:
        return f"tasks={list(self.task_heads.keys())}"
