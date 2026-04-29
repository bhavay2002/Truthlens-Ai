"""
Three-layer imbalance strategy tests.

    Layer 1: TaskScheduler            — between-task imbalance
    Layer 2: data-level (oversampling, samplers)
    Layer 3: loss-level balancing     — class weights, pos_weight, focal
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch

from src.training.task_scheduler import TaskScheduler, TaskSchedulerConfig
from src.training.loss_functions import (
    FocalLoss,
    compute_class_weights,
    compute_pos_weight,
)
from src.training.loss_balancer import (
    LossBalancerConfig,
    plan_for_dataframe,
    plan_for_labels,
)
from src.training.loss_engine import LossEngine, LossEngineConfig
from src.data_processing.class_balance import (
    analyze_classification,
    analyze_multilabel,
)


# =========================================================
# LAYER 1 — TASK SCHEDULER
# =========================================================

class TestTaskBalance:
    def test_weighted_strategy_normalizes_weights(self):
        sched = TaskScheduler(
            tasks=["a", "b", "c"],
            config=TaskSchedulerConfig(
                strategy="weighted",
                task_weights={"a": 2.0, "b": 1.0, "c": 1.0},
            ),
        )
        weights = sched.get_weights()
        assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0
        assert weights["a"] > weights["b"]
        assert weights["b"] == pytest.approx(weights["c"])

    def test_weighted_strategy_samples_proportionally(self):
        sched = TaskScheduler(
            tasks=["heavy", "light"],
            config=TaskSchedulerConfig(
                strategy="weighted",
                task_weights={"heavy": 4.0, "light": 1.0},
                seed=123,
            ),
        )
        counts = Counter(sched.next_task() for _ in range(5_000))
        ratio = counts["heavy"] / counts["light"]
        # expected ratio is 4.0; allow ±25% sampling slack
        assert 3.0 < ratio < 5.5, counts

    def test_round_robin_visits_each_task(self):
        sched = TaskScheduler(tasks=["a", "b", "c"])
        seq = [sched.next_task() for _ in range(6)]
        assert seq == ["a", "b", "c", "a", "b", "c"]

    def test_adaptive_focuses_on_high_loss_task(self):
        sched = TaskScheduler(
            tasks=["easy", "hard"],
            config=TaskSchedulerConfig(strategy="adaptive", seed=7),
        )
        # Inject losses many times so the EMA actually moves.
        for _ in range(100):
            sched.update_losses({"easy": 0.05, "hard": 5.0})
        counts = Counter(sched.next_task() for _ in range(2_000))
        assert counts["hard"] > counts["easy"], counts


# =========================================================
# LAYER 2 — DATA-LEVEL CLASS-BALANCE ANALYSIS
# =========================================================

class TestClassBalance:
    def test_imbalance_detected_on_skewed_classification(self):
        df = pd.DataFrame({"y": [0] * 95 + [1] * 5})
        report = analyze_classification(df, "y")
        assert report.imbalance_detected is True
        assert report.weights is not None
        # Minority class should weigh more than majority class.
        assert report.weights[1] > report.weights[0]

    def test_no_imbalance_on_balanced_classification(self):
        df = pd.DataFrame({"y": [0] * 50 + [1] * 50})
        report = analyze_classification(df, "y")
        assert report.imbalance_detected is False

    def test_multilabel_per_column_weight(self):
        # Columns: A is balanced, B is heavily skewed positive.
        df = pd.DataFrame({"A": [1, 0] * 50, "B": [1] * 95 + [0] * 5})
        report = analyze_multilabel(df, ["A", "B"])
        assert report.weights is not None
        # neg/pos for B is small (≈5/95), for A it's 1.0 — sanity check direction.
        assert report.weights["A"] > report.weights["B"]


# =========================================================
# LAYER 3 — LOSS-LEVEL BALANCING
# =========================================================

class TestComputeClassWeights:
    def test_weights_are_higher_for_rare_classes(self):
        labels = [0] * 90 + [1] * 10
        w = compute_class_weights(labels, num_classes=2)
        assert w[1] > w[0]

    def test_normalized_weights_sum_to_num_classes(self):
        labels = [0] * 50 + [1] * 30 + [2] * 20
        w = compute_class_weights(labels, num_classes=3)
        assert pytest.approx(float(w.sum()), rel=1e-5) == 3.0

    def test_balanced_input_yields_uniform_weights(self):
        labels = [0] * 10 + [1] * 10 + [2] * 10
        w = compute_class_weights(labels, num_classes=3)
        assert torch.allclose(w, torch.ones(3), atol=1e-3)

    def test_unseen_class_does_not_explode(self):
        # Class 2 never appears — smoothing must keep weight finite.
        labels = [0, 0, 1]
        w = compute_class_weights(labels, num_classes=3)
        assert torch.isfinite(w).all()


class TestComputePosWeight:
    def test_rare_positive_yields_large_pos_weight(self):
        labels = torch.tensor([[1.0]] * 5 + [[0.0]] * 95)
        pw = compute_pos_weight(labels, smoothing=0.0)
        # neg/pos = 95/5 = 19
        assert float(pw.item()) == pytest.approx(19.0, rel=1e-3)

    def test_pos_weight_per_column(self):
        # col0: 50/50 balanced. col1: 10 positives / 90 negatives (rare).
        col0 = [1.0] * 50 + [0.0] * 50
        col1 = [1.0] * 10 + [0.0] * 90
        labels = torch.tensor(list(zip(col0, col1)), dtype=torch.float32)
        pw = compute_pos_weight(labels, smoothing=0.0)
        assert pw[0].item() == pytest.approx(1.0, rel=1e-3)
        # neg/pos = 90/10 = 9.0
        assert pw[1].item() == pytest.approx(9.0, rel=1e-3)


class TestFocalLoss:
    def test_focal_with_gamma_zero_matches_weighted_ce(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 3)
        targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        weight = torch.tensor([1.0, 2.0, 0.5])

        focal = FocalLoss(gamma=0.0, weight=weight)(logits, targets)
        ce = torch.nn.CrossEntropyLoss(weight=weight)(logits, targets)
        assert torch.allclose(focal, ce, atol=1e-6)

    def test_focal_downweights_easy_examples(self):
        # One trivially-correct example + one hard example.
        logits = torch.tensor(
            [
                [10.0, 0.0],   # easy: model is very confident in correct class
                [0.1, 0.0],    # hard: model barely above chance for correct class
            ]
        )
        targets = torch.tensor([0, 0])

        ce = torch.nn.CrossEntropyLoss(reduction="none")(logits, targets)
        # Per-sample focal contribution before mean reduction.
        focal_module = FocalLoss(gamma=2.0, reduction="none")
        focal = focal_module(logits, targets)

        # Easy sample should be far more attenuated than the hard sample.
        attenuation_easy = (focal[0] / ce[0]).item()
        attenuation_hard = (focal[1] / ce[1]).item()
        assert attenuation_easy < attenuation_hard
        assert attenuation_easy < 0.01

    def test_ignore_index_is_respected(self):
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, -100, 2, -100])
        loss = FocalLoss(ignore_index=-100)(logits, targets)
        assert torch.isfinite(loss)


# =========================================================
# LOSS BALANCER PLANNER
# =========================================================

class TestLossBalancerPlanner:
    def test_balanced_multiclass_no_weighting(self):
        labels = [0, 1, 2] * 100
        plan = plan_for_labels(labels, task_type="multiclass", num_classes=3)
        assert plan.class_weights is None
        assert plan.use_focal is False

    def test_imbalanced_multiclass_enables_class_weights(self):
        labels = [0] * 80 + [1] * 15 + [2] * 5
        plan = plan_for_labels(labels, task_type="multiclass", num_classes=3)
        assert plan.class_weights is not None
        assert plan.use_focal is False  # 0.8 < focal_threshold default 0.9

    def test_extreme_imbalance_enables_focal(self):
        labels = [0] * 95 + [1] * 5
        plan = plan_for_labels(labels, task_type="multiclass", num_classes=2)
        assert plan.class_weights is not None
        assert plan.use_focal is True
        assert plan.focal_gamma == 2.0

    def test_multilabel_drops_degenerate_columns(self):
        # Col 0 is all zero, col 1 is balanced, col 2 is all one.
        df = pd.DataFrame({
            "A": [0] * 100,
            "B": [0, 1] * 50,
            "C": [1] * 100,
        })
        plan = plan_for_dataframe(
            df,
            label_columns=["A", "B", "C"],
            task_type="multilabel",
        )
        assert plan.valid_label_indices == [1]
        assert sorted(plan.dropped_label_indices) == [0, 2]
        assert plan.pos_weight is not None
        assert plan.pos_weight.shape == (1,)

    def test_plan_for_dataframe_handles_missing_columns(self):
        df = pd.DataFrame({"y": [0, 1, 0, 1]})
        plan = plan_for_dataframe(
            df,
            label_columns=["does_not_exist"],
            task_type="multiclass",
            num_classes=2,
        )
        assert "label_columns_missing" in plan.notes


# =========================================================
# END-TO-END: LossEngine consumes the plan
# =========================================================

class TestLossEngineWithBalancing:
    def _logits_and_labels(self, *, num_classes, batch_size=16, skew=0.9):
        torch.manual_seed(0)
        n_majority = int(batch_size * skew)
        labels = torch.tensor(
            [0] * n_majority + list(range(1, num_classes)) * (
                (batch_size - n_majority) // max(num_classes - 1, 1)
            )
        )
        labels = labels[:batch_size]
        logits = torch.randn(batch_size, num_classes, requires_grad=True)
        return logits, labels

    def test_class_weights_propagate_to_cross_entropy_module(self):
        cw = compute_class_weights([0] * 90 + [1] * 10, num_classes=2)
        engine = LossEngine(
            LossEngineConfig(
                task_types={"bias": "multiclass"},
                class_weights={"bias": cw},
            )
        )
        loss_fn = engine.loss_module.router.loss_functions["bias"]
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
        assert loss_fn.weight is not None
        assert torch.allclose(loss_fn.weight, cw)

    def test_focal_loss_engaged_for_extreme_imbalance(self):
        cw = compute_class_weights([0] * 95 + [1] * 5, num_classes=2)
        engine = LossEngine(
            LossEngineConfig(
                task_types={"bias": "multiclass"},
                class_weights={"bias": cw},
                use_focal={"bias": True},
                focal_gamma={"bias": 2.0},
            )
        )
        loss_fn = engine.loss_module.router.loss_functions["bias"]
        assert isinstance(loss_fn, FocalLoss)
        assert loss_fn.gamma == 2.0

    def test_pos_weight_propagates_for_multilabel(self):
        pw = torch.tensor([1.0, 9.0, 19.0])
        engine = LossEngine(
            LossEngineConfig(
                task_types={"emotion": "multilabel"},
                pos_weights={"emotion": pw},
            )
        )
        loss_fn = engine.loss_module.router.loss_functions["emotion"]
        assert isinstance(loss_fn, torch.nn.BCEWithLogitsLoss)
        assert torch.allclose(loss_fn.pos_weight, pw)

    def test_loss_is_finite_and_backprop_works(self):
        cw = compute_class_weights([0] * 90 + [1] * 10, num_classes=2)
        engine = LossEngine(
            LossEngineConfig(
                task_types={"bias": "multiclass"},
                class_weights={"bias": cw},
                use_focal={"bias": True},
            )
        )
        torch.manual_seed(0)
        logits = torch.randn(32, 2, requires_grad=True)
        labels = torch.tensor([0] * 28 + [1] * 4)
        outputs = {"task_logits": {"bias": logits}}
        batch = {"labels": {"bias": labels}}

        total_loss, task_losses = engine.compute(outputs, batch)
        assert torch.isfinite(total_loss)
        assert "bias" in task_losses

        total_loss.backward()
        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_balanced_dataset_keeps_unweighted_baseline(self):
        # No plan inputs → vanilla CrossEntropyLoss with weight=None.
        engine = LossEngine(
            LossEngineConfig(task_types={"bias": "multiclass"})
        )
        loss_fn = engine.loss_module.router.loss_functions["bias"]
        assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
        assert loss_fn.weight is None
