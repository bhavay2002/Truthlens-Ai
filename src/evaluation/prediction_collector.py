from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from src.config.task_config import TASK_CONFIG, get_task_type
from src.utils.device_utils import autocast_context, move_batch

logger = logging.getLogger(__name__)


# =========================================================
# DEVICE
# =========================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# TOKENIZATION
# =========================================================

def _tokenize(tokenizer: AutoTokenizer, texts: List[str], max_length: int):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# =========================================================
# POSTPROCESS
# =========================================================

def _postprocess(logits: np.ndarray, task_type: str, *, threshold: float = 0.5):
    """Convert raw logits into ``(preds, probs)`` for a given task type.

    Notably handles the binary case where the head emits 2 logits (softmax over
    {0, 1}) versus a single logit (sigmoid).
    """
    arr = np.asarray(logits)
    logits_t = torch.tensor(arr, dtype=torch.float32)

    if task_type == "multiclass":
        probs = torch.softmax(logits_t, dim=-1).numpy()
        preds = np.argmax(probs, axis=1).astype(int)

    elif task_type == "binary":
        if logits_t.ndim == 2 and logits_t.shape[-1] == 2:
            probs_full = torch.softmax(logits_t, dim=-1).numpy()
            probs = probs_full[:, 1]
        else:
            probs = torch.sigmoid(logits_t).numpy().reshape(-1)
        preds = (probs >= threshold).astype(int)

    elif task_type == "multilabel":
        probs = torch.sigmoid(logits_t).numpy()
        preds = (probs >= threshold).astype(int)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return preds, probs


# =========================================================
# COLLECTOR — class wrapper used by Evaluator
# =========================================================

class PredictionCollector:
    """Light wrapper that bundles raw model output into a uniform record.

    The class is intentionally stateless; downstream consumers (ErrorAnalyzer,
    ThresholdOptimizer) only need the dictionary it returns.
    """

    @staticmethod
    def collect(
        *,
        y_true: Optional[Iterable] = None,
        y_pred: Optional[Iterable] = None,
        y_proba: Optional[Iterable] = None,
        logits: Optional[Iterable] = None,
        task: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {"task": task, "task_type": task_type}

        if y_true is not None:
            record["y_true"] = np.asarray(y_true)
        if y_pred is not None:
            record["y_pred"] = np.asarray(y_pred)
        if y_proba is not None:
            record["y_proba"] = np.asarray(y_proba)
        if logits is not None:
            record["logits"] = np.asarray(logits)

        return record


# =========================================================
# SINGLE-TASK COLLECTION FROM TEXTS
# =========================================================

def collect_predictions(
    model,
    texts: List[str],
    task: str,
    tokenizer: AutoTokenizer,
    *,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}")

    device = device or get_device()
    task_type = TASK_CONFIG[task]["type"]

    model.to(device)
    model.eval()

    all_logits: List[np.ndarray] = []

    logger.info("[COLLECT] task=%s samples=%d", task, len(texts))

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            encoded = _tokenize(tokenizer, batch_texts, max_length)
            encoded = move_batch(encoded, device)

            with autocast_context():
                out = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    task=task,
                )

            logits = out["logits"].detach().cpu().numpy()
            all_logits.append(logits)

    logits_arr = np.vstack(all_logits) if all_logits else np.empty((0,))
    preds, probs = _postprocess(logits_arr, task_type, threshold=threshold)

    return {
        "task": task,
        "task_type": task_type,
        "logits": logits_arr,
        "probabilities": probs,
        "predictions": preds,
    }


# =========================================================
# MULTI-TASK COLLECTION FROM TEXTS
# =========================================================

def collect_all_tasks(
    model,
    texts: List[str],
    tokenizer: AutoTokenizer,
    *,
    tasks: Optional[List[str]] = None,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, Any]]:
    selected_tasks = tasks or list(TASK_CONFIG.keys())
    device = device or get_device()

    results: Dict[str, Dict[str, Any]] = {}

    logger.info("[COLLECT] multi-task start (%d tasks)", len(selected_tasks))

    for task in selected_tasks:
        results[task] = collect_predictions(
            model=model,
            texts=texts,
            task=task,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

    logger.info("[COLLECT] multi-task done")
    return results


# =========================================================
# DATALOADER PATH (used by EvaluationEngine)
# =========================================================

def _normalize_label_dict(batch_labels: Any, tasks: List[str]) -> Dict[str, torch.Tensor]:
    """Pull a ``{task: tensor}`` dict out of a batch's label payload."""
    if isinstance(batch_labels, dict):
        return {t: batch_labels[t] for t in tasks if t in batch_labels}
    raise TypeError(
        "DataLoader batches must yield a dict-like ``labels`` field for "
        "multi-task evaluation"
    )


def collect_all_tasks_from_loader(
    model,
    dataloader,
    *,
    tasks: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Run inference over a DataLoader, returning per-task arrays + ground truth."""
    selected_tasks = tasks or list(TASK_CONFIG.keys())
    device = device or get_device()

    model.to(device)
    model.eval()

    logits_buf: Dict[str, List[np.ndarray]] = {t: [] for t in selected_tasks}
    labels_buf: Dict[str, List[np.ndarray]] = {t: [] for t in selected_tasks}

    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = move_batch(batch, device)
            inputs = {
                k: v
                for k, v in batch_on_device.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }

            label_dict = _normalize_label_dict(
                batch_on_device.get("labels", batch_on_device),
                selected_tasks,
            )

            for task in selected_tasks:
                with autocast_context():
                    out = model(task=task, **inputs)

                logits_buf[task].append(out["logits"].detach().cpu().numpy())

                if task in label_dict:
                    labels_buf[task].append(
                        label_dict[task].detach().cpu().numpy()
                    )

    results: Dict[str, Dict[str, Any]] = {}
    for task in selected_tasks:
        if not logits_buf[task]:
            continue

        task_type = get_task_type(task)
        logits_arr = np.vstack(logits_buf[task])
        preds, probs = _postprocess(logits_arr, task_type, threshold=threshold)

        record: Dict[str, Any] = {
            "task": task,
            "task_type": task_type,
            "logits": logits_arr,
            "probabilities": probs,
            "predictions": preds,
            "y_pred": preds,
            "y_proba": probs,
        }
        if labels_buf[task]:
            y_true = np.concatenate(labels_buf[task], axis=0)
            record["y_true"] = y_true
            record["labels"] = y_true

        results[task] = record

    return results


# =========================================================
# STREAMING (LARGE DATASETS)
# =========================================================

def stream_logits(
    model,
    texts: List[str],
    task: str,
    tokenizer: AutoTokenizer,
    *,
    batch_size: int = 32,
    max_length: int = 512,
):
    device = get_device()
    model.to(device)
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        encoded = _tokenize(tokenizer, batch_texts, max_length)
        encoded = move_batch(encoded, device)

        with torch.no_grad(), autocast_context():
            out = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                task=task,
            )

        yield out["logits"].detach().cpu().numpy()


__all__ = [
    "PredictionCollector",
    "collect_all_tasks",
    "collect_all_tasks_from_loader",
    "collect_predictions",
    "get_device",
    "stream_logits",
]
