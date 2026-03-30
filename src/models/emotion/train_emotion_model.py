"""
File Name: train_emotion_model.py
Module: Training Pipeline - Emotion Model Training
Description:
    Implements the training pipeline for the emotion classification model in the
    TruthLens AI system. The module loads configuration parameters, prepares
    datasets and dataloaders, initializes the model, optimizer, and scheduler,
    and runs the training loop with logging, checkpointing, and evaluation.

Dependencies:
    logging
    random
    typing
    yaml
    torch
    torch.nn
    torch.optim
    torch.utils.data
    numpy
    emotion_classifier (local module)

Inputs:
    YAML configuration file containing training parameters

Outputs:
    Trained emotion model and training logs
"""

import logging
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.features.emotion.emotion_classifier import EmotionClassifier


logger = logging.getLogger(__name__)


class EmotionTrainer:
    """
    Handles training workflow for the emotion classification model.
    """

    def __init__(
        self,
        config_path: str,
        train_dataset,
        val_dataset,
    ) -> None:
        """Initialize training pipeline."""

        if not isinstance(config_path, str) or not config_path:
            raise ValueError("config_path must be a valid string")

        self.config = self._load_config(config_path)

        self._set_seed(self.config.get("seed", 42))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
        )

        self.model = EmotionClassifier(
            model_name=self.config["model"]["encoder_model"],
            num_emotions=self.config["model"]["num_labels"],
            multi_label=bool(self.config["model"].get("multi_label", False)),
            dropout=self.config["model"].get("dropout", 0.1),
            device=str(self.device),
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
        )

        self.loss_fn = (
            nn.BCEWithLogitsLoss()
            if bool(self.config["model"].get("multi_label", False))
            else nn.CrossEntropyLoss()
        )

        logger.info("EmotionTrainer initialized")

    def train(self) -> None:
        """Run the full training loop."""

        epochs = self.config["training"]["epochs"]

        for epoch in range(epochs):

            train_loss = self._train_epoch()

            val_loss = self._validate_epoch()

            logger.info(
                "Epoch %d | Train Loss: %.4f | Val Loss: %.4f",
                epoch + 1,
                train_loss,
                val_loss,
            )

    def _train_epoch(self) -> float:
        """Train model for one epoch."""

        self.model.train()

        total_loss = 0.0
        batch_count = 0

        for batch in self.train_loader:

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs["loss"]

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            batch_count += 1

        return total_loss / max(batch_count, 1)

    def _validate_epoch(self) -> float:
        """Run validation loop."""

        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():

            for batch in self.val_loader:

                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs["loss"]

                total_loss += loss.item()

                batch_count += 1

        return total_loss / max(batch_count, 1)

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration."""

        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as exc:
            logger.exception("Failed to load configuration")
            raise RuntimeError("Config loading failed") from exc

    def _set_seed(self, seed: int) -> None:
        """Ensure reproducibility."""

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
