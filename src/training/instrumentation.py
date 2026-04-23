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


# -----------------------------------------------------------------------------
# Phase 4 — Multi-signal anomaly classification
# -----------------------------------------------------------------------------

class AnomalyClassifier:
    """Priority-ordered multi-signal anomaly classifier.

    Order matters: hard numerical failures must be reported before
    optimization pathologies, and optimization issues before data /
    instability ones, otherwise a NaN loss could be reported as
    ``loss_spike`` and the actual cause masked.

    Returns one of:
      ``nan_loss``, ``nan_logits``, ``exploding_gradients``,
      ``vanishing_gradients``, ``negative_labels``, ``invalid_labels``,
      ``high_variance``, ``loss_spike``, ``logit_collapse``, ``normal``.
    """

    def __init__(
        self,
        spike_ratio: float = 2.5,
        explode_th: float = 1000.0,
        vanish_th: float = 1e-7,
        var_th: float = 5.0,
        logit_collapse_th: float = 1e-4,
        eps: float = 1e-8,
    ):
        self.spike_ratio = float(spike_ratio)
        self.explode_th = float(explode_th)
        self.vanish_th = float(vanish_th)
        self.var_th = float(var_th)
        self.logit_collapse_th = float(logit_collapse_th)
        self.eps = float(eps)

    def classify(
        self,
        loss: float,
        ema_loss: float,
        grad_stats: Optional[Dict[str, float]] = None,
        loss_var: Optional[float] = None,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None,
    ) -> str:
        # 1. Hard numerical failures — these must short-circuit. A NaN loss
        #    with a finite gradient norm should still report nan_loss.
        loss_v = float(loss)
        if not math.isfinite(loss_v):
            return "nan_loss"
        if logits is not None and torch.is_tensor(logits) and not torch.isfinite(logits).all():
            return "nan_logits"

        # 2. Gradient pathologies.
        if grad_stats is not None:
            gn = float(grad_stats.get("total_norm", 0.0))
            if not math.isfinite(gn) or gn > self.explode_th:
                return "exploding_gradients"
            if gn < self.vanish_th:
                return "vanishing_gradients"

        # 3. Data / label issues.
        if labels is not None and torch.is_tensor(labels) and labels.numel() > 0:
            if (labels < 0).any().item():
                return "negative_labels"
            if num_classes is not None and (labels >= num_classes).any().item():
                return "invalid_labels"

        # 4. Instability — variance first (catches noisy training that
        #    happens to cross the spike threshold), then ratio.
        if loss_var is not None and float(loss_var) > self.var_th:
            return "high_variance"
        ema_v = float(ema_loss)
        if ema_v > 0:
            ratio = loss_v / (ema_v + self.eps)
            if ratio > self.spike_ratio:
                return "loss_spike"

        # 5. Logit collapse — silent failure where the head outputs a near
        #    constant for every input (e.g. dead ReLU, frozen head).
        if logits is not None and torch.is_tensor(logits) and logits.numel() > 1:
            if float(logits.float().std().item()) < self.logit_collapse_th:
                return "logit_collapse"

        return "normal"


def anomaly_severity(loss: float, ema_loss: float, grad_norm: float) -> str:
    """Coarse severity bucket — pairs with AnomalyClassifier for triage."""
    loss = float(loss)
    ema_loss = float(ema_loss)
    grad_norm = float(grad_norm)
    if not math.isfinite(loss) or not math.isfinite(grad_norm) or grad_norm > 1000:
        return "critical"
    ratio = loss / (ema_loss + 1e-8) if ema_loss > 0 else 0.0
    if ratio > 5:
        return "high"
    if ratio > 2:
        return "medium"
    return "low"


# -----------------------------------------------------------------------------
# Phase 5 — GradNorm (Chen et al., 2018)
# -----------------------------------------------------------------------------

class GradNorm:
    """Paper-aligned GradNorm task weighting.

    Produces a per-task weight dict that satisfies ``sum(weights) == T``
    (where T is the number of tasks), preserving the magnitude of the
    aggregate loss while rebalancing relative contributions.

    NOTE: GradNorm is opt-in. Wiring it changes convergence dynamics —
    only enable it if multi-task imbalance is observed in the dominance
    detector. See ``compute_task_grad_norms`` for the required helper.
    """

    def __init__(self, tasks: Iterable[str], alpha: float = 0.5, eps: float = 1e-8):
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise ValueError("GradNorm requires at least one task")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.initial_losses: Optional[Dict[str, float]] = None
        self.weights: Dict[str, float] = {t: 1.0 for t in self.tasks}

    def compute(
        self,
        task_losses: Dict[str, Any],
        task_grad_norms: Dict[str, float],
    ) -> Dict[str, float]:
        # Normalize floats once.
        losses = {
            t: float(v.detach().item()) if torch.is_tensor(v) else float(v)
            for t, v in task_losses.items()
        }
        norms = {t: float(task_grad_norms[t]) for t in self.tasks if t in task_grad_norms}

        if self.initial_losses is None:
            # Guard against zero-init losses that would make ratios explode.
            self.initial_losses = {t: max(losses[t], self.eps) for t in self.tasks}

        loss_ratios = {t: losses[t] / (self.initial_losses[t] + self.eps) for t in self.tasks}
        avg_ratio = sum(loss_ratios.values()) / len(loss_ratios)

        # Inverse training rate: tasks falling behind get larger targets.
        targets = {t: avg_ratio * (loss_ratios[t] ** self.alpha) for t in self.tasks}

        new_weights = {
            t: targets[t] / (norms.get(t, self.eps) + self.eps) for t in self.tasks
        }
        # Renormalize so sum(weights) == T (the GradNorm constraint).
        total = sum(new_weights.values())
        scale = len(self.tasks) / (total + self.eps)
        self.weights = {t: w * scale for t, w in new_weights.items()}
        return self.weights


def compute_task_grad_norms(
    losses: Dict[str, torch.Tensor],
    shared_params: Iterable[torch.nn.Parameter],
) -> Dict[str, float]:
    """Per-task gradient norm computed over the **shared backbone**.

    GradNorm specifically requires the gradient of each task loss w.r.t.
    the shared parameters, not the full model. ``retain_graph=True`` keeps
    the graph alive for the subsequent backward pass on the weighted sum.
    """
    shared_list = [p for p in shared_params if p.requires_grad]
    if not shared_list:
        raise ValueError("compute_task_grad_norms: no trainable shared params")
    out: Dict[str, float] = {}
    for task, loss in losses.items():
        if not torch.is_tensor(loss) or not loss.requires_grad:
            out[task] = 0.0
            continue
        grads = torch.autograd.grad(
            loss, shared_list, retain_graph=True, allow_unused=True,
        )
        sq = 0.0
        for g in grads:
            if g is not None:
                sq += float(g.detach().norm().item()) ** 2
        out[task] = math.sqrt(sq)
    return out


# -----------------------------------------------------------------------------
# Phase 6 — Per-parameter gradient hooks
# -----------------------------------------------------------------------------

class GradHookManager:
    """Attach per-parameter backward hooks; aggregate over a window.

    Hooks fire on every ``backward()`` call, so callers must invoke
    ``reset()`` on a known cadence (e.g. every log interval) to avoid
    unbounded memory growth.
    """

    def __init__(self):
        self.buffer: Dict[str, list[float]] = {}
        self._handles: list[Any] = []

    def attach(
        self,
        model: torch.nn.Module,
        filter_fn: Optional[Any] = None,
    ) -> int:
        """Register hooks. Returns the number of hooks attached."""
        n = 0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if filter_fn is not None and not filter_fn(name):
                continue
            self._handles.append(p.register_hook(self._make_hook(name)))
            n += 1
        return n

    def _make_hook(self, name: str):
        def hook(grad: torch.Tensor):
            if grad is None:
                return
            self.buffer.setdefault(name, []).append(float(grad.detach().norm().item()))
        return hook

    def aggregate(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, vals in self.buffer.items():
            if not vals:
                continue
            out[name] = {
                "mean": sum(vals) / len(vals),
                "max": max(vals),
                "min": min(vals),
                "n": float(len(vals)),
            }
        return out

    def reset(self) -> None:
        self.buffer = {}

    def detach(self) -> None:
        """Remove all hooks. Call before deleting the model to avoid leaks."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []
        self.buffer = {}


# -----------------------------------------------------------------------------
# Phase 7 — Task dominance detection
# -----------------------------------------------------------------------------

class TaskDominanceDetector:
    """EMA-smoothed grad-norm dominance detector.

    Returns a dict ``{dominant, suppressed, ratio}`` when the smoothed
    max/min ratio crosses ``dominance_ratio``, else ``None``. EMA
    smoothing prevents flapping on a single noisy step.
    """

    def __init__(self, alpha: float = 0.1, dominance_ratio: float = 5.0, eps: float = 1e-8):
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.ratio = float(dominance_ratio)
        self.eps = float(eps)
        self.ema: Dict[str, float] = {}

    def update(self, task_grads: Dict[str, float]) -> Optional[Dict[str, Any]]:
        for t, g in task_grads.items():
            gv = float(g)
            if not math.isfinite(gv):
                continue
            if t in self.ema:
                self.ema[t] = self.alpha * gv + (1 - self.alpha) * self.ema[t]
            else:
                self.ema[t] = gv

        if len(self.ema) < 2:
            return None
        max_t = max(self.ema, key=self.ema.get)  # type: ignore[arg-type]
        min_t = min(self.ema, key=self.ema.get)  # type: ignore[arg-type]
        if self.ema[min_t] <= self.eps:
            return None
        r = self.ema[max_t] / self.ema[min_t]
        if r > self.ratio:
            return {"dominant": max_t, "suppressed": min_t, "ratio": r}
        return None


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
    "AnomalyClassifier",
    "anomaly_severity",
    "GradNorm",
    "compute_task_grad_norms",
    "GradHookManager",
    "TaskDominanceDetector",
]
