"""
File Name: emotion_target.py
Module: Emotion Analysis - Target Encoding

Description:
    Emotion label encoder used for training and inference within the TruthLens AI emotion subsystem.

"""

import logging
from typing import Dict, List, Iterable, Optional

import numpy as np
import torch


logger = logging.getLogger(__name__)


class EmotionTargetEncoder:

    DEFAULT_EMOTIONS = [
        "anger",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "disgust",
        "trust",
        "anticipation",
    ]

    def __init__(
        self,
        emotion_labels: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):

        labels = emotion_labels if emotion_labels else self.DEFAULT_EMOTIONS

        if not isinstance(labels, list) or not labels:
            raise ValueError("emotion_labels must be a non-empty list")

        self.labels = [label.lower() for label in labels]

        self.label_to_index = {
            label: idx for idx, label in enumerate(self.labels)
        }

        self.index_to_label = {
            idx: label for label, idx in self.label_to_index.items()
        }

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("EmotionTargetEncoder initialized | labels=%d", len(self.labels))

    # -----------------------------------------------------
    # Encoding
    # -----------------------------------------------------

    def encode(self, labels: List[str]) -> np.ndarray:

        if not isinstance(labels, list):
            raise ValueError("labels must be a list")

        vector = np.zeros(len(self.labels), dtype=np.float32)

        for label in labels:

            if not isinstance(label, str):
                continue

            label = label.lower().strip()

            if label in self.label_to_index:
                vector[self.label_to_index[label]] = 1.0

        return vector

    # -----------------------------------------------------

    def encode_batch(self, label_lists: Iterable[List[str]]) -> np.ndarray:

        vectors = [self.encode(labels) for labels in label_lists]

        return np.vstack(vectors)

    # -----------------------------------------------------

    def encode_tensor(self, labels: List[str]) -> torch.Tensor:

        vector = self.encode(labels)

        return torch.tensor(vector, dtype=torch.float32, device=self.device)

    # -----------------------------------------------------
    # Decoding
    # -----------------------------------------------------

    def decode(
        self,
        vector: np.ndarray,
        threshold: float = 0.5,
    ) -> List[str]:

        if vector.shape[0] != len(self.labels):
            raise ValueError("vector size mismatch")

        labels = []

        for idx, value in enumerate(vector):

            if value >= threshold:
                labels.append(self.index_to_label[idx])

        return labels

    # -----------------------------------------------------

    def decode_topk(
        self,
        vector: np.ndarray,
        k: int = 1,
    ) -> List[str]:

        indices = np.argsort(vector)[::-1][:k]

        return [self.index_to_label[i] for i in indices]

    # -----------------------------------------------------

    def decode_tensor(
        self,
        tensor: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[str]:

        vector = tensor.detach().cpu().numpy()

        return self.decode(vector, threshold)

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    def get_label_mapping(self) -> Dict[str, int]:

        return dict(self.label_to_index)

    def get_labels(self) -> List[str]:

        return list(self.labels)