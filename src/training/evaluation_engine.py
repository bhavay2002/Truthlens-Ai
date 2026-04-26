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
# STREAMING METRICS
# =========================================================

class StreamingAccuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, preds, labels):
        self.correct += (preds == labels).sum().item()
        self.total += labels.numel()

    def compute(self):
        return self.correct / (self.total + 1e-12)

    def reset(self):
        self.correct = 0
        self.total = 0


class StreamingF1:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, preds, labels):
        preds = preds.view(-1).int()
        labels = labels.view(-1).int()

        self.tp += torch.logical_and(preds == 1, labels == 1).sum().item()
        self.fp += torch.logical_and(preds == 1, labels == 0).sum().item()
        self.fn += torch.logical_and(preds == 0, labels == 1).sum().item()

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-12)
        recall = self.tp / (self.tp + self.fn + 1e-12)
        return 2 * precision * recall / (precision + recall + 1e-12)

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0


class StreamingMSE:
    def __init__(self):
        self.sum_sq = 0.0
        self.count = 0

    def update(self, preds, targets):
        diff = preds - targets
        self.sum_sq += (diff ** 2).sum().item()
        self.count += targets.numel()

    def compute(self):
        return self.sum_sq / (self.count + 1e-12)

    def reset(self):
        self.sum_sq = 0.0
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

            preds = preds.detach().cpu()
            labels = labels.detach().cpu()

            metrics[task].update(preds, labels)

    # =====================================================
    # COMPUTE
    # =====================================================

    def _compute_metrics(self, metrics):

        results = {}

        for task, metric in metrics.items():

            value = metric.compute()

            # DDP sync
            value = self._sync_scalar(value)

            results[f"{task}_score"] = value

        return results

    # =====================================================
    # DDP SYNC
    # =====================================================

    def _sync_scalar(self, value: float) -> float:

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tensor = torch.tensor(value, device=self.device)
            torch.distributed.all_reduce(tensor)
            tensor /= torch.distributed.get_world_size()
            return tensor.item()

        return value

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