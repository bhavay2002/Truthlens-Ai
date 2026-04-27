#src\models\training\evaluation_engine.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class EvaluationConfig:
    task_types: Dict[str, str]
    device: Optional[str] = None
    ignore_index: int = -100
    threshold: float = 0.5


# =========================================================
# STREAMING METRICS  (PERF-1: keep accumulators on-device)
# =========================================================
#
# Original implementation called ``.sum().item()`` on every batch which
# forces a host-device sync per metric per batch (3-6× val-loop slowdown
# on GPU). The new metrics:
#   * lazily allocate accumulator tensors on the FIRST batch's device
#   * never sync inside ``update`` — only on ``compute``
#   * expose ``sync_distributed`` so DDP can SUM-reduce raw counters
#     (PERF-2: correct DDP merging requires (sum, count) reductions, not
#     pre-divided averages)
# =========================================================


def _all_reduce_sum(*tensors: torch.Tensor) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for t in tensors:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)


class StreamingAccuracy:
    def __init__(self):
        self.correct: Optional[torch.Tensor] = None  # device tensor, lazy
        self.total: int = 0                          # host int (numel is metadata, no sync)

    def update(self, preds, labels):
        if self.correct is None:
            self.correct = torch.zeros((), device=preds.device, dtype=torch.float64)
        self.correct = self.correct + (preds == labels).sum().to(self.correct.dtype)
        self.total += int(labels.numel())

    def sync_distributed(self):
        if self.correct is None:
            return
        total_t = torch.tensor(
            float(self.total),
            device=self.correct.device,
            dtype=self.correct.dtype,
        )
        _all_reduce_sum(self.correct, total_t)
        self.total = int(total_t.item())

    def compute(self):
        if self.correct is None:
            return 0.0
        return float(self.correct.item()) / max(self.total, 1)

    def reset(self):
        self.correct = None
        self.total = 0


class StreamingF1:
    def __init__(self):
        self.tp: Optional[torch.Tensor] = None
        self.fp: Optional[torch.Tensor] = None
        self.fn: Optional[torch.Tensor] = None

    def _ensure(self, device):
        if self.tp is None:
            self.tp = torch.zeros((), device=device, dtype=torch.float64)
            self.fp = torch.zeros((), device=device, dtype=torch.float64)
            self.fn = torch.zeros((), device=device, dtype=torch.float64)

    def update(self, preds, labels):
        self._ensure(preds.device)
        preds = preds.view(-1).to(torch.int64)
        labels = labels.view(-1).to(torch.int64)
        self.tp += torch.logical_and(preds == 1, labels == 1).sum().to(self.tp.dtype)
        self.fp += torch.logical_and(preds == 1, labels == 0).sum().to(self.fp.dtype)
        self.fn += torch.logical_and(preds == 0, labels == 1).sum().to(self.fn.dtype)

    def sync_distributed(self):
        if self.tp is None:
            return
        _all_reduce_sum(self.tp, self.fp, self.fn)

    def compute(self):
        if self.tp is None:
            return 0.0
        tp = float(self.tp.item())
        fp = float(self.fp.item())
        fn = float(self.fn.item())
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        return 2 * precision * recall / (precision + recall + 1e-12)

    def reset(self):
        self.tp = self.fp = self.fn = None


class StreamingMSE:
    def __init__(self):
        self.sum_sq: Optional[torch.Tensor] = None
        self.count: int = 0

    def update(self, preds, targets):
        if self.sum_sq is None:
            self.sum_sq = torch.zeros((), device=preds.device, dtype=torch.float64)
        diff = preds.float() - targets.float()
        self.sum_sq = self.sum_sq + (diff ** 2).sum().to(self.sum_sq.dtype)
        self.count += int(targets.numel())

    def sync_distributed(self):
        if self.sum_sq is None:
            return
        count_t = torch.tensor(
            float(self.count),
            device=self.sum_sq.device,
            dtype=self.sum_sq.dtype,
        )
        _all_reduce_sum(self.sum_sq, count_t)
        self.count = int(count_t.item())

    def compute(self):
        if self.sum_sq is None:
            return 0.0
        return float(self.sum_sq.item()) / max(self.count, 1)

    def reset(self):
        self.sum_sq = None
        self.count = 0


# =========================================================
# EVALUATION ENGINE
# =========================================================

class EvaluationEngine:

    def __init__(self, config: EvaluationConfig):

        self.config = config

        self.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("EvaluationEngine initialized | device=%s", self.device)

    # =====================================================
    # MAIN
    # =====================================================

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, dataloader) -> Dict[str, Any]:

        model = model.to(self.device)
        model.eval()

        metrics = self._init_metrics()

        for i, batch in enumerate(dataloader):

            batch = self._move_batch(batch)

            outputs = model(**batch)

            self._update_metrics(metrics, outputs, batch)

        return self._compute_metrics(metrics)

    # =====================================================
    # METRICS INIT
    # =====================================================

    def _init_metrics(self):

        metrics = {}

        for task, ttype in self.config.task_types.items():

            if ttype == "multiclass":
                metrics[task] = StreamingAccuracy()

            elif ttype == "multilabel":
                metrics[task] = StreamingF1()

            elif ttype == "regression":
                metrics[task] = StreamingMSE()

        return metrics

    # =====================================================
    # UPDATE
    # =====================================================

    def _update_metrics(self, metrics, outputs, batch):

        task_logits = outputs.get("task_logits")
        if task_logits is None:
            return

        for task, logits in task_logits.items():

            if task not in batch["labels"]:
                continue

            labels = batch["labels"][task].to(logits.device)
            ttype = self.config.task_types.get(task)

            if ttype == "multiclass":

                preds = torch.argmax(logits, dim=-1)

                if labels.dim() == 2:
                    labels = labels.argmax(dim=-1)

                mask = labels != self.config.ignore_index
                preds = preds[mask]
                labels = labels[mask]

            elif ttype == "multilabel":

                preds = (torch.sigmoid(logits) > self.config.threshold).int()

                mask = labels != self.config.ignore_index
                preds = preds[mask]
                labels = labels[mask]

            elif ttype == "regression":

                preds = logits

            else:
                continue

            # PERF-1: keep tensors on device — Streaming* metrics now hold
            # accumulators on the GPU and only sync at compute() time.
            metrics[task].update(preds.detach(), labels.detach())

    # =====================================================
    # COMPUTE
    # =====================================================

    def _compute_metrics(self, metrics):

        results = {}

        for task, metric in metrics.items():

            # PERF-2: Reduce raw (numerator, denominator) accumulators across
            # ranks BEFORE dividing — averaging post-divided rank-local scores
            # is mathematically wrong when shards have different sample
            # counts (drop_last=False). Each Streaming* metric implements
            # `sync_distributed` which all_reduces only its raw counters.
            if hasattr(metric, "sync_distributed"):
                metric.sync_distributed()

            results[f"{task}_score"] = metric.compute()

        return results

    # =====================================================
    # DEVICE
    # =====================================================

    def _move_batch(self, batch):

        return {
            k: v.to(self.device, non_blocking=True)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }