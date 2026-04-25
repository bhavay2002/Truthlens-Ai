from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from src.config.task_config import TASK_CONFIG
from src.utils.device_utils import move_batch, autocast_context

logger = logging.getLogger(__name__)


# =========================================================
# DEVICE
# =========================================================

def get_device():
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
# POSTPROCESS (STRICT)
# =========================================================

def _postprocess(logits: np.ndarray, task_type: str):
    logits_t = torch.tensor(logits)

    if task_type == "multiclass":
        probs = torch.softmax(logits_t, dim=-1).numpy()
        preds = np.argmax(probs, axis=1)

    elif task_type == "binary":
        probs = torch.sigmoid(logits_t).numpy().reshape(-1)
        preds = (probs >= 0.5).astype(int)

    elif task_type == "multilabel":
        probs = torch.sigmoid(logits_t).numpy()
        preds = (probs >= 0.5).astype(int)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return preds, probs


# =========================================================
# SINGLE TASK COLLECTION
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
) -> Dict[str, Any]:

    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}")

    device = device or get_device()
    task_type = TASK_CONFIG[task]["type"]

    model.to(device)
    model.eval()

    all_logits = []

    logger.info(f"[COLLECT] task={task} samples={len(texts)}")

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):

            batch_texts = texts[i:i + batch_size]

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

    logits = np.vstack(all_logits)

    preds, probs = _postprocess(logits, task_type)

    return {
        "task": task,
        "task_type": task_type,
        "logits": logits,
        "probabilities": probs,
        "predictions": preds,
    }


# =========================================================
# MULTI-TASK COLLECTION
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

    tasks = tasks or list(TASK_CONFIG.keys())
    device = device or get_device()

    results = {}

    logger.info("[COLLECT] multi-task start")

    for task in tasks:
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
# STREAM MODE (LARGE DATASETS)
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

        batch_texts = texts[i:i + batch_size]

        encoded = _tokenize(tokenizer, batch_texts, max_length)
        encoded = move_batch(encoded, device)

        with torch.no_grad(), autocast_context():
            out = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                task=task,
            )

        yield out["logits"].detach().cpu().numpy()