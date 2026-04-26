from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


# =========================================================
# CLASSIFICATION SAMPLER
# =========================================================

def build_classification_sampler(
    labels: List[int],
    *,
    normalize: bool = True,
) -> WeightedRandomSampler:
    """
    Build WeightedRandomSampler for classification tasks.

    Args:
        labels: list/array of class labels
    """

    labels = np.asarray(labels)

    class_counts = np.bincount(labels)
    total = class_counts.sum()

    # inverse frequency
    class_weights = total / np.maximum(class_counts, 1)

    if normalize:
        class_weights = class_weights / class_weights.sum()

    sample_weights = class_weights[labels]

    logger.info(
        "Sampler | classification | classes=%d",
        len(class_counts),
    )

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# =========================================================
# MULTILABEL SAMPLER
# =========================================================

def build_multilabel_sampler(
    label_matrix: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> WeightedRandomSampler:
    """
    Build sampler for multilabel tasks.

    Strategy:
        - weight samples by inverse frequency of positive labels
    """

    label_matrix = np.asarray(label_matrix)

    # count positives per label
    pos_counts = label_matrix.sum(axis=0)

    # avoid division by zero
    pos_counts = np.maximum(pos_counts, epsilon)

    label_weights = 1.0 / pos_counts

    # sample weight = sum of label weights
    sample_weights = (label_matrix * label_weights).sum(axis=1)

    # fallback if all zeros
    sample_weights = np.where(sample_weights == 0, 1.0, sample_weights)

    logger.info(
        "Sampler | multilabel | labels=%d",
        label_matrix.shape[1],
    )

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


# =========================================================
# LENGTH-BASED BUCKET SAMPLER (OPTIONAL)
# =========================================================

class BucketSampler(Sampler):
    """
    Groups sequences of similar lengths to reduce padding.

    NOTE: use with caution in multi-task setup.
    """

    def __init__(
        self,
        lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
    ):
        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # sort indices by length
        self.indices = np.argsort(self.lengths)

    def __iter__(self):

        if self.shuffle:
            # shuffle buckets
            buckets = [
                self.indices[i:i + self.batch_size]
                for i in range(0, len(self.indices), self.batch_size)
            ]
            np.random.shuffle(buckets)
            indices = np.concatenate(buckets)
        else:
            indices = self.indices

        return iter(indices.tolist())

    def __len__(self):
        return len(self.lengths)


# =========================================================
# FACTORY (CRITICAL)
# =========================================================

def build_sampler(
    *,
    task: str,
    df,
    use_weighted: bool = True,
):
    """
    Factory to build sampler per task.
    """

    if not use_weighted:
        return None

    if task in ("bias", "ideology", "propaganda"):
        labels = df[task].values
        return build_classification_sampler(labels)

    elif task == "frame":
        labels = df[["CO", "EC", "HI", "MO", "RE"]].values
        return build_multilabel_sampler(labels)

    elif task == "narrative":
        labels = df[["hero", "villain", "victim"]].values
        return build_multilabel_sampler(labels)

    elif task == "emotion":
        cols = [f"emotion_{i}" for i in range(20)]
        labels = df[cols].values
        return build_multilabel_sampler(labels)

    else:
        raise ValueError(f"Unknown task: {task}")