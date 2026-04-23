"""
Defensive training instrumentation for TruthLens.

Provides:
  * LossTracker        — bias-corrected EMA per task, NaN/inf rejection
  * LossStats          — windowed mean + variance per task
  * GradTracker        — per-step total / mean grad-norm with windowed history
  * detect_grad_anomaly — exploding / vanishing classifier
  * SpikeDetector      — ratio-or-z-score hybrid
  * validate_labels    — fail-fast tensor-level label validation
  * check_optimizer    — multi-param-group LR snapshot
  * apply_clipping     — wrapper that returns the pre-clip total norm
  * dump_batch         — atomic .pt dump of a payload for post-mortem

This module is deliberately framework-agnostic: callers pass tensors or
dicts and receive plain Python types back. Heavy detail (per-parameter
norms) is computed lazily and only kept in a bounded ring buffer so the
instrumentation never grows memory unboundedly during long runs.
"""
from __future__ import annotations

import math
import os
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional

import torch


# -----------------------------------------------------------------------------
# Loss tracking
# -----------------------------------------------------------------------------

class LossTracker:
    """Bias-corrected EMA of per-task losses.

    The bias correction matters early: a raw EMA initialized at the first
    observation under-weights subsequent steps; ``ema / (1 - (1-alpha)**t)``
    fixes the warm-up bias the same way Adam corrects its moments.
    """

    def __init__(self, tasks: Iterable[str], alpha: float = 0.1, eps: float = 1e-8):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.ema: Dict[str, Optional[float]] = {t: None for t in tasks}
        self.steps: Dict[str, int] = {t: 0 for t in tasks}

    def update(self, losses: Dict[str, Any]) -> Dict[str, float]:
        smoothed: Dict[str, float] = {}
        for task, val in losses.items():
            v = float(val.detach().item()) if torch.is_tensor(val) else float(val)
            if not math.isfinite(v):
                raise ValueError(f"Non-finite loss in task {task!r}: {v}")

            if task not in self.ema:
                # Tolerate task names not declared at construction time.
                self.ema[task] = None
                self.steps[task] = 0

            self.steps[task] += 1
            # Adam-style: initialize at 0 so the bias correction below
            # actually produces an unbiased estimate. Initializing at the
            # first observation (as the audit doc's pseudocode does) makes
            # the bias correction wrong on every step.
            prev = self.ema[task] if self.ema[task] is not None else 0.0
            self.ema[task] = self.alpha * v + (1 - self.alpha) * prev

            bias_correction = 1.0 - (1.0 - self.alpha) ** self.steps[task]
            smoothed[task] = self.ema[task] / (bias_correction + self.eps)
        return smoothed


class LossStats:
    """Windowed mean / variance per task. Detects instability that an EMA misses."""

    def __init__(self, tasks: Iterable[str], window: int = 50):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        self.window = int(window)
        self.history: Dict[str, Deque[float]] = {t: deque(maxlen=self.window) for t in tasks}

    def update(self, losses: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for task, val in losses.items():
            v = float(val.detach().item()) if torch.is_tensor(val) else float(val)
            if task not in self.history:
                self.history[task] = deque(maxlen=self.window)
            hist = self.history[task]
            hist.append(v)
            if len(hist) > 1:
                # Use unbiased variance to match torch.var default behavior.
                var = float(torch.tensor(list(hist)).var(unbiased=True).item())
            else:
                var = 0.0
            out[task] = {"mean": sum(hist) / len(hist), "var": var}
        return out


# -----------------------------------------------------------------------------
# Gradient tracking
# -----------------------------------------------------------------------------

class GradTracker:
    """Bounded ring buffer of per-step gradient summaries.

    ``total_norm`` matches ``torch.nn.utils.clip_grad_norm_`` (L2 over all
    parameters); ``mean_norm`` and ``mean_var`` are per-parameter averages
    useful for spotting one layer dominating the global norm.
    """

    def __init__(self, window: int = 50):
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = int(window)
        self.history: Deque[Dict[str, float]] = deque(maxlen=self.window)

    def update(self, model: torch.nn.Module) -> Dict[str, float]:
        norms: list[float] = []
        vars_: list[float] = []
        sq_total = 0.0

        for _, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            n = float(g.norm().item())
            norms.append(n)
            # var() requires >= 2 elements; single-element params have 0 variance.
            vars_.append(float(g.var(unbiased=False).item()) if g.numel() > 1 else 0.0)
            sq_total += n * n

        record = {
            "total_norm": math.sqrt(sq_total),
            "mean_norm": sum(norms) / max(1, len(norms)),
            "mean_var": sum(vars_) / max(1, len(vars_)),
            "n_params": float(len(norms)),
        }
        self.history.append(record)
        return record


def detect_grad_anomaly(
    grad_stats: Dict[str, float],
    explode_th: float = 1000.0,
    vanish_th: float = 1e-6,
) -> str:
    """Classify a grad record as ``EXPLODING`` / ``VANISHING`` / ``NORMAL``."""
    norm = float(grad_stats.get("total_norm", 0.0))
    if not math.isfinite(norm) or norm > explode_th:
        return "EXPLODING"
    if norm < vanish_th:
        return "VANISHING"
    return "NORMAL"


# -----------------------------------------------------------------------------
# Validation utilities
# -----------------------------------------------------------------------------

def validate_labels(labels: torch.Tensor, num_classes: int, *, name: str = "labels") -> None:
    """Fail-fast on malformed label tensors.

    Catches the failure mode where a CSV-coercion bug emits ``-1`` or
    ``num_classes + 1`` and the loss silently produces garbage gradients.
    """
    if not torch.is_tensor(labels):
        raise TypeError(f"{name} must be a tensor, got {type(labels).__name__}")
    if labels.numel() == 0:
        raise ValueError(f"{name} is empty")
    lo = int(labels.min().item())
    hi = int(labels.max().item())
    if lo < 0 or hi >= num_classes:
        raise ValueError(
            f"{name} out of range: [{lo}, {hi}] not in [0, {num_classes - 1}]"
        )


def check_optimizer(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """Multi-param-group LR snapshot — useful when differential LR is in use."""
    lrs = [float(pg["lr"]) for pg in optimizer.param_groups]
    if not lrs:
        return {"min_lr": 0.0, "max_lr": 0.0, "mean_lr": 0.0, "n_groups": 0.0}
    return {
        "min_lr": min(lrs),
        "max_lr": max(lrs),
        "mean_lr": sum(lrs) / len(lrs),
        "n_groups": float(len(lrs)),
    }


def apply_clipping(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """Clip and return the pre-clip total norm (for logging)."""
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total.detach().item()) if torch.is_tensor(total) else float(total)


# -----------------------------------------------------------------------------
# Spike detection
# -----------------------------------------------------------------------------

class SpikeDetector:
    """Hybrid ratio + z-score spike detector.

    ``ratio = loss / ema`` catches multiplicative blowups; the z-score
    catches the case where ``ema`` is near zero but variance is small,
    which the ratio alone would flag spuriously every step.
    """

    def __init__(self, threshold: float = 2.5, eps: float = 1e-8):
        if threshold <= 0:
            raise ValueError(f"threshold must be > 0, got {threshold}")
        self.threshold = float(threshold)
        self.eps = float(eps)

    def detect(
        self,
        loss: float,
        ema_loss: float,
        var: Optional[float] = None,
    ) -> bool:
        loss = float(loss)
        ema_loss = float(ema_loss)
        if not math.isfinite(loss):
            return True
        ratio = loss / (ema_loss + self.eps) if ema_loss > 0 else 0.0
        if var is not None and var > 0:
            z = (loss - ema_loss) / (math.sqrt(var) + self.eps)
        else:
            z = 0.0
        return ratio > self.threshold or z > self.threshold


# -----------------------------------------------------------------------------
# Batch dump (post-mortem evidence)
# -----------------------------------------------------------------------------

def dump_batch(debug_path: str | os.PathLike, payload: Dict[str, Any]) -> str:
    """Atomically write a torch payload to ``debug_path`` and return the path.

    Tensors are detached and moved to CPU first so the dump can never hold
    a CUDA-graph reference. Filenames are millisecond-stamped so concurrent
    dumps don't collide.
    """
    debug_path = os.fspath(debug_path)
    os.makedirs(debug_path, exist_ok=True)

    safe: Dict[str, Any] = {}
    for k, v in payload.items():
        if torch.is_tensor(v):
            safe[k] = v.detach().cpu()
        elif isinstance(v, dict):
            safe[k] = {
                kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                for kk, vv in v.items()
            }
        else:
            safe[k] = v

    final_path = os.path.join(debug_path, f"spike_{int(time.time() * 1000)}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(safe, tmp_path)
    os.replace(tmp_path, final_path)
    return final_path


__all__ = [
    "LossTracker",
    "LossStats",
    "GradTracker",
    "detect_grad_anomaly",
    "SpikeDetector",
    "validate_labels",
    "check_optimizer",
    "apply_clipping",
    "dump_batch",
]
