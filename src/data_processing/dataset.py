"""
TruthLens datasets (pre-tokenized, contract-driven).

Design:
- All tokenization happens ONCE in __init__ (no per-sample tokenizer calls).
- Texts/labels are stored as numpy arrays / lists for O(1) __getitem__ access.
- Optionally returns offset_mapping for downstream token-alignment / explainability.
- Label column names come from the data_contracts module (single source of truth).
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# =========================================================
# BASE DATASET
# =========================================================

class BaseTextDataset(Dataset):
    """
    Base dataset: pre-tokenizes the entire text column up-front.

    Args:
        df: dataframe (must contain `text_col`)
        tokenizer: HuggingFace tokenizer (fast tokenizer required if
            ``return_offsets_mapping=True``)
        text_col: text column name
        max_length: max tokens per sample (truncation only — padding done in
            the collate step)
        return_offsets_mapping: if True, store per-sample offset_mapping for
            char-level alignment in explainability layers
        log_truncation: if True, log how many samples were truncated
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        *,
        text_col: str = "text",
        max_length: int = 512,
        return_offsets_mapping: bool = False,
        log_truncation: bool = True,
    ):
        if text_col not in df.columns:
            raise ValueError(
                f"Missing text column '{text_col}' (have: {list(df.columns)})"
            )

        self.tokenizer = tokenizer
        self.text_col = text_col
        self.max_length = max_length
        self.return_offsets_mapping = return_offsets_mapping
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

        # ----- pre-tokenize whole column -----
        texts = df[text_col].astype(str).tolist()

        if return_offsets_mapping and not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "return_offsets_mapping=True requires a fast tokenizer "
                "(PreTrainedTokenizerFast)."
            )

        enc = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_attention_mask=True,
            return_offsets_mapping=return_offsets_mapping,
            return_length=True,
        )

        ids_lists: List[List[int]] = enc["input_ids"]
        attn_lists: List[List[int]] = enc["attention_mask"]
        om_lists: Optional[List[List[List[int]]]] = (
            enc.get("offset_mapping") if return_offsets_mapping else None
        )

        # =====================================================
        # FLATTEN STORAGE (PERF-D2)
        #
        # ~25M Python ints for a 100k × 256 corpus = ~200 MB of pure
        # CPython object overhead, plus full GC scans, plus a copy
        # storm whenever a DataLoader worker forks. Storing one
        # ``int32`` ids array + one ``int64`` offsets array is
        # ~3-5× lower RSS, ~30% faster ``__getitem__``, and the
        # arrays are shared by reference across worker forks.
        # =====================================================
        n = len(ids_lists)
        lengths = np.fromiter((len(x) for x in ids_lists), dtype=np.int64, count=n)
        self._offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._offsets[1:])
        total = int(self._offsets[-1])

        self._ids_flat = np.empty(total, dtype=np.int32)
        self._attn_flat = np.empty(total, dtype=np.int8)
        cursor = 0
        for ids, attn in zip(ids_lists, attn_lists):
            k = len(ids)
            self._ids_flat[cursor:cursor + k] = ids
            self._attn_flat[cursor:cursor + k] = attn
            cursor += k

        if om_lists is not None:
            self._om_flat: Optional[np.ndarray] = np.empty((total, 2), dtype=np.int64)
            cursor = 0
            for om in om_lists:
                k = len(om)
                self._om_flat[cursor:cursor + k] = om
                cursor += k
        else:
            self._om_flat = None

        # truncation diagnostics — use the canonical HuggingFace signal
        # ``encodings[i].overflowing`` when available (TOK-D2). The old
        # ``L >= max_length`` heuristic over-counted samples that fit
        # exactly. Fall back to the heuristic for slow tokenizers.
        if log_truncation:
            n_truncated = 0
            encodings = getattr(enc, "encodings", None)
            if encodings is not None:
                n_truncated = sum(1 for e in encodings if getattr(e, "overflowing", None))
            else:
                fallback_lengths = enc.get("length") or [len(x) for x in ids_lists]
                n_truncated = sum(1 for L in fallback_lengths if L >= max_length)
            if n_truncated > 0:
                logger.warning(
                    "Tokenizer truncation | samples=%d | truncated=%d (%.1f%%) | max_length=%d",
                    len(texts),
                    n_truncated,
                    100.0 * n_truncated / max(len(texts), 1),
                    max_length,
                )

        self._n = n

    def __len__(self) -> int:
        return self._n

    # subclasses override __getitem__ — base helper returns the encoded inputs
    def _encoded_inputs(self, idx: int) -> Dict[str, torch.Tensor]:
        s = int(self._offsets[idx])
        e = int(self._offsets[idx + 1])
        item: Dict[str, torch.Tensor] = {
            # .astype(int64) returns a fresh array; from_numpy then takes
            # ownership and yields a tensor without a second copy.
            "input_ids": torch.from_numpy(self._ids_flat[s:e].astype(np.int64, copy=True)),
            "attention_mask": torch.from_numpy(self._attn_flat[s:e].astype(np.int64, copy=True)),
        }
        if self._om_flat is not None:
            item["offset_mapping"] = torch.from_numpy(
                self._om_flat[s:e].astype(np.int64, copy=True)
            )
        return item


# =========================================================
# CLASSIFICATION DATASET (bias, ideology, propaganda)
# =========================================================

class ClassificationDataset(BaseTextDataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        *,
        label_col: str,
        num_classes: int,
        task_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(df, tokenizer, **kwargs)

        if label_col not in df.columns:
            raise ValueError(
                f"Missing label column '{label_col}' (have: {list(df.columns)})"
            )

        self.label_col = label_col
        self.num_classes = num_classes
        self.task_name = task_name or label_col

        # vectorize labels once
        labels = df[label_col].to_numpy()
        if pd.isna(labels).any():
            raise ValueError(
                f"NaN labels in column '{label_col}' — clean/validate first."
            )
        self._labels = labels.astype(np.int64)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._encoded_inputs(idx)
        item["labels"] = torch.as_tensor(self._labels[idx], dtype=torch.long)
        item["task"] = self.task_name
        return item


# =========================================================
# MULTILABEL DATASET (frame, narrative, emotion)
# =========================================================

class MultiLabelDataset(BaseTextDataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        *,
        label_cols: List[str],
        task_name: str,
        **kwargs,
    ):
        super().__init__(df, tokenizer, **kwargs)

        missing = [c for c in label_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing multilabel columns {missing} (have: {list(df.columns)})"
            )

        self.label_cols = list(label_cols)
        self.task_name = task_name

        matrix = df[self.label_cols].to_numpy(dtype=np.float32)
        if np.isnan(matrix).any():
            raise ValueError(
                f"NaN values in multilabel columns {self.label_cols} — clean first."
            )
        self._label_matrix = matrix

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._encoded_inputs(idx)
        item["labels"] = torch.as_tensor(self._label_matrix[idx], dtype=torch.float)
        item["task"] = self.task_name
        return item
