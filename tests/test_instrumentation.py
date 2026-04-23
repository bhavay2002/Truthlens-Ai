"""Tests for the defensive training instrumentation subsystem."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.training.instrumentation import (
    GradTracker,
    LossStats,
    LossTracker,
    SpikeDetector,
    apply_clipping,
    check_optimizer,
    detect_grad_anomaly,
    dump_batch,
    validate_labels,
)


# ----- LossTracker -----------------------------------------------------------

def test_loss_tracker_bias_correction_first_step_is_input():
    lt = LossTracker(tasks=["a"], alpha=0.1)
    out = lt.update({"a": 5.0})
    # After 1 step, bias_correction = alpha, so corrected = 5.0 * alpha / alpha = 5.0
    assert math.isclose(out["a"], 5.0, rel_tol=1e-5)


def test_loss_tracker_rejects_non_finite():
    lt = LossTracker(tasks=["a"])
    with pytest.raises(ValueError, match="Non-finite"):
        lt.update({"a": float("nan")})
    with pytest.raises(ValueError, match="Non-finite"):
        lt.update({"a": float("inf")})


def test_loss_tracker_unknown_task_is_tolerated():
    lt = LossTracker(tasks=["a"])
    out = lt.update({"a": 1.0, "b_new": 2.0})
    assert "b_new" in out


def test_loss_tracker_accepts_tensors():
    lt = LossTracker(tasks=["a"])
    out = lt.update({"a": torch.tensor(3.0)})
    assert math.isclose(out["a"], 3.0, rel_tol=1e-5)


# ----- LossStats -------------------------------------------------------------

def test_loss_stats_window_bounded():
    ls = LossStats(tasks=["a"], window=3)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        out = ls.update({"a": v})
    # Only last 3 values [3,4,5] should be in the window
    assert math.isclose(out["a"]["mean"], 4.0, rel_tol=1e-5)
    assert out["a"]["var"] > 0


def test_loss_stats_zero_var_with_one_sample():
    ls = LossStats(tasks=["a"], window=10)
    out = ls.update({"a": 1.0})
    assert out["a"]["var"] == 0.0


# ----- GradTracker -----------------------------------------------------------

def test_grad_tracker_records_total_norm():
    model = torch.nn.Linear(4, 2)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()

    gt = GradTracker(window=5)
    rec = gt.update(model)
    assert rec["total_norm"] > 0
    assert rec["n_params"] == 2  # weight + bias
    assert len(gt.history) == 1


def test_grad_tracker_history_window_bounded():
    model = torch.nn.Linear(2, 2)
    gt = GradTracker(window=3)
    for _ in range(5):
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        gt.update(model)
    assert len(gt.history) == 3


# ----- detect_grad_anomaly ---------------------------------------------------

def test_detect_grad_anomaly_classifies():
    assert detect_grad_anomaly({"total_norm": 10.0}) == "NORMAL"
    assert detect_grad_anomaly({"total_norm": 1e10}) == "EXPLODING"
    assert detect_grad_anomaly({"total_norm": 1e-9}) == "VANISHING"
    assert detect_grad_anomaly({"total_norm": float("inf")}) == "EXPLODING"


# ----- SpikeDetector ---------------------------------------------------------

def test_spike_detector_ratio_path():
    sd = SpikeDetector(threshold=2.5)
    assert sd.detect(loss=10.0, ema_loss=1.0) is True
    assert sd.detect(loss=1.5, ema_loss=1.0) is False


def test_spike_detector_zscore_path():
    sd = SpikeDetector(threshold=2.5)
    # ratio is small (~1.5x), but z-score of 5σ should fire
    assert sd.detect(loss=1.5, ema_loss=1.0, var=0.01) is True


def test_spike_detector_handles_non_finite_loss():
    sd = SpikeDetector()
    assert sd.detect(loss=float("nan"), ema_loss=1.0) is True


# ----- validate_labels -------------------------------------------------------

def test_validate_labels_passes_in_range():
    validate_labels(torch.tensor([0, 1, 2]), num_classes=3)


def test_validate_labels_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        validate_labels(torch.tensor([0, 3]), num_classes=3)
    with pytest.raises(ValueError, match="out of range"):
        validate_labels(torch.tensor([-1, 0]), num_classes=3)


def test_validate_labels_rejects_empty_and_non_tensor():
    with pytest.raises(ValueError, match="empty"):
        validate_labels(torch.tensor([], dtype=torch.long), num_classes=3)
    with pytest.raises(TypeError):
        validate_labels([0, 1, 2], num_classes=3)  # type: ignore[arg-type]


# ----- check_optimizer / apply_clipping --------------------------------------

def test_check_optimizer_multi_group():
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.SGD([
        {"params": [model.weight], "lr": 1e-3},
        {"params": [model.bias], "lr": 1e-2},
    ])
    snap = check_optimizer(opt)
    assert snap["min_lr"] == 1e-3
    assert snap["max_lr"] == 1e-2
    assert snap["n_groups"] == 2


def test_apply_clipping_returns_preclip_norm():
    model = torch.nn.Linear(4, 2)
    for p in model.parameters():
        p.grad = torch.full_like(p, 10.0)
    pre = apply_clipping(model, max_norm=1.0)
    assert pre > 1.0  # we created big gradients
    # After clipping, parameter grads should have norm ~ 1.0
    post = sum((p.grad.norm() ** 2).item() for p in model.parameters()) ** 0.5
    assert math.isclose(post, 1.0, rel_tol=1e-3)


# ----- dump_batch ------------------------------------------------------------

def test_dump_batch_writes_cpu_tensors(tmp_path: Path):
    payload = {
        "step": 42,
        "inputs": {"input_ids": torch.tensor([[1, 2, 3]])},
        "losses": {"bias": 0.1, "ideology": 0.5},
    }
    out_path = dump_batch(str(tmp_path), payload)
    assert Path(out_path).exists()
    loaded = torch.load(out_path, weights_only=False)
    assert loaded["step"] == 42
    assert loaded["inputs"]["input_ids"].device.type == "cpu"
    assert loaded["losses"]["bias"] == 0.1
