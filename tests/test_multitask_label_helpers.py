from __future__ import annotations

import pytest
import torch

from src.models.multitask.multitask_truthlens_model import MultiTaskTruthLensModel


def test_prepare_single_label_targets_accepts_indices() -> None:
    labels = torch.tensor([0, 2, 1], dtype=torch.int64)

    targets = MultiTaskTruthLensModel._prepare_single_label_targets(
        labels,
        num_classes=3,
        task_name="ideology",
    )

    assert targets.dtype == torch.int64
    assert targets.tolist() == [0, 2, 1]


def test_prepare_single_label_targets_converts_one_hot() -> None:
    labels = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    targets = MultiTaskTruthLensModel._prepare_single_label_targets(
        labels,
        num_classes=3,
        task_name="ideology",
    )

    assert targets.tolist() == [0, 2]


def test_prepare_multi_label_targets_rejects_wrong_shape() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0])

    with pytest.raises(ValueError):
        MultiTaskTruthLensModel._prepare_multi_label_targets(
            labels,
            num_classes=3,
            task_name="narrative",
        )
