"""Re-export shim for the canonical TransformerEncoder.

The real `TransformerEncoder` implementation lives in
`src.models.inference.model_wrapper`. This module exists purely so callers
can keep importing it from `src.models.encoder.transformer_encoder` without
caring where the class is defined.

Historically this file also contained a duplicate `EncoderFactory` class
that shadowed the canonical one in `src.models.encoder.encoder_factory`
and silently drifted out of sync. That duplicate has been removed; use
`src.models.encoder.encoder_factory.EncoderFactory` instead.
"""

from __future__ import annotations

from src.models.inference.model_wrapper import TransformerEncoder

__all__ = ["TransformerEncoder"]
