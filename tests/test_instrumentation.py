"""Tests for the defensive training instrumentation subsystem."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.training.instrumentation import (
    AnomalyClassifier,
    GradHookManager,
    GradNorm,
    GradTracker,
    LossStats,
    LossTracker,
    SpikeDetector,
    TaskDominanceDetector,
    anomaly_severity,
    apply_clipping,
    check_optimizer,
    compute_task_grad_norms,
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


# ----- AnomalyClassifier -----------------------------------------------------

def test_anomaly_classifier_priority_nan_loss_first():
    c = AnomalyClassifier()
    # NaN loss should win even when grad_stats look exploding.
    assert c.classify(
        loss=float("nan"), ema_loss=1.0,
        grad_stats={"total_norm": 1e9},
    ) == "nan_loss"


def test_anomaly_classifier_nan_logits():
    c = AnomalyClassifier()
    bad = torch.tensor([1.0, float("nan")])
    assert c.classify(loss=1.0, ema_loss=1.0, logits=bad) == "nan_logits"


def test_anomaly_classifier_grad_pathologies():
    c = AnomalyClassifier(explode_th=100.0, vanish_th=1e-6)
    assert c.classify(1.0, 1.0, grad_stats={"total_norm": 1e3}) == "exploding_gradients"
    assert c.classify(1.0, 1.0, grad_stats={"total_norm": 1e-9}) == "vanishing_gradients"


def test_anomaly_classifier_label_issues():
    c = AnomalyClassifier()
    assert c.classify(1.0, 1.0, labels=torch.tensor([-1, 0])) == "negative_labels"
    assert c.classify(
        1.0, 1.0, labels=torch.tensor([0, 5]), num_classes=3,
    ) == "invalid_labels"


def test_anomaly_classifier_high_variance_before_spike():
    c = AnomalyClassifier(spike_ratio=2.0, var_th=1.0)
    # ratio ~ 5x AND high variance — variance reported first by design.
    assert c.classify(loss=5.0, ema_loss=1.0, loss_var=10.0) == "high_variance"


def test_anomaly_classifier_loss_spike():
    c = AnomalyClassifier(spike_ratio=2.0)
    assert c.classify(loss=5.0, ema_loss=1.0) == "loss_spike"


def test_anomaly_classifier_logit_collapse():
    c = AnomalyClassifier()
    flat = torch.full((8, 4), 1.0)
    # Calm loss + flat logits => collapse.
    assert c.classify(loss=1.0, ema_loss=1.0, logits=flat) == "logit_collapse"


def test_anomaly_classifier_normal():
    c = AnomalyClassifier()
    assert c.classify(loss=1.0, ema_loss=1.0) == "normal"


# ----- anomaly_severity ------------------------------------------------------

def test_anomaly_severity_buckets():
    assert anomaly_severity(1.0, 1.0, 5000.0) == "critical"
    assert anomaly_severity(float("nan"), 1.0, 1.0) == "critical"
    assert anomaly_severity(10.0, 1.0, 1.0) == "high"
    assert anomaly_severity(3.0, 1.0, 1.0) == "medium"
    assert anomaly_severity(1.5, 1.0, 1.0) == "low"


# ----- GradNorm --------------------------------------------------------------

def test_gradnorm_weights_sum_to_num_tasks():
    gn = GradNorm(tasks=["a", "b", "c"], alpha=0.5)
    losses = {"a": 1.0, "b": 2.0, "c": 0.5}
    grad_norms = {"a": 0.5, "b": 1.0, "c": 0.2}
    w = gn.compute(losses, grad_norms)
    assert math.isclose(sum(w.values()), 3.0, rel_tol=1e-5)
    assert set(w) == {"a", "b", "c"}


def test_gradnorm_lagging_task_gets_more_weight():
    gn = GradNorm(tasks=["fast", "slow"], alpha=1.0)
    # Initial step seeds reference losses.
    gn.compute({"fast": 1.0, "slow": 1.0}, {"fast": 1.0, "slow": 1.0})
    # Now "slow" is still at 1.0 (high relative loss), "fast" dropped to 0.1.
    w = gn.compute({"fast": 0.1, "slow": 1.0}, {"fast": 1.0, "slow": 1.0})
    assert w["slow"] > w["fast"]


def test_compute_task_grad_norms_uses_shared_params_only():
    shared = torch.nn.Linear(4, 8)
    head_a = torch.nn.Linear(8, 2)
    head_b = torch.nn.Linear(8, 3)
    x = torch.randn(4, 4)
    feat = shared(x)
    out_a = head_a(feat)
    out_b = head_b(feat)
    losses = {
        "a": torch.nn.functional.cross_entropy(out_a, torch.randint(0, 2, (4,))),
        "b": torch.nn.functional.cross_entropy(out_b, torch.randint(0, 3, (4,))),
    }
    norms = compute_task_grad_norms(losses, shared.parameters())
    assert set(norms) == {"a", "b"}
    assert all(n > 0 for n in norms.values())


# ----- GradHookManager -------------------------------------------------------

def test_grad_hook_manager_records_and_resets():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    mgr = GradHookManager()
    n = mgr.attach(model)
    assert n > 0

    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()

    agg = mgr.aggregate()
    assert len(agg) > 0
    for stats in agg.values():
        assert {"mean", "max", "min", "n"} <= stats.keys()
        assert stats["n"] >= 1.0

    mgr.reset()
    assert mgr.aggregate() == {}
    mgr.detach()


def test_grad_hook_manager_filter_fn():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Linear(8, 2),
    )
    mgr = GradHookManager()
    # Only keep the second linear's params (index "1.").
    n = mgr.attach(model, filter_fn=lambda name: name.startswith("1."))
    assert n == 2  # weight + bias of the second linear
    mgr.detach()


# ----- TaskDominanceDetector -------------------------------------------------

def test_task_dominance_detector_fires_above_ratio():
    d = TaskDominanceDetector(alpha=1.0, dominance_ratio=3.0)  # alpha=1 => no smoothing
    out = d.update({"a": 10.0, "b": 1.0})
    assert out is not None
    assert out["dominant"] == "a"
    assert out["suppressed"] == "b"
    assert out["ratio"] >= 3.0


def test_task_dominance_detector_silent_when_balanced():
    d = TaskDominanceDetector(alpha=1.0, dominance_ratio=5.0)
    assert d.update({"a": 1.0, "b": 1.2}) is None


def test_task_dominance_detector_returns_none_with_one_task():
    d = TaskDominanceDetector()
    assert d.update({"a": 100.0}) is None
