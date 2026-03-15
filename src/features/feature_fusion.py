"""
File: feature_fusion.py

Purpose
-------
Feature fusion utilities for TruthLens AI.

This module combines transformer embeddings with symbolic
features such as bias and emotion scores.

Input
-----
cls_embedding : torch.Tensor
bias_score : torch.Tensor
emotion_score : torch.Tensor

Output
------
torch.Tensor
    fused feature vector
"""

import logging
import torch

# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Feature Fusion Function
# ---------------------------------------------------------


def fuse_features(
    cls_embedding: torch.Tensor,
    bias_score: torch.Tensor,
    emotion_score: torch.Tensor,
) -> torch.Tensor:
    """
    Fuse RoBERTa embeddings with bias and emotion features.

    Parameters
    ----------
    cls_embedding : torch.Tensor
        CLS token embedding from RoBERTa
        Shape: (batch_size, hidden_size)

    bias_score : torch.Tensor
        Bias detection score
        Shape: (batch_size,)

    emotion_score : torch.Tensor
        Emotion manipulation score
        Shape: (batch_size,)

    Returns
    -------
    torch.Tensor
        Fused feature vector
        Shape: (batch_size, hidden_size + 2)
    """

    try:

        if bias_score.dim() == 1:
            bias_score = bias_score.unsqueeze(1)

        if emotion_score.dim() == 1:
            emotion_score = emotion_score.unsqueeze(1)

        fused_features = torch.cat(
            [cls_embedding, bias_score, emotion_score],
            dim=1,
        )

        logger.debug("Feature fusion successful")

        return fused_features

    except Exception:

        logger.exception("Feature fusion failed")

        raise
