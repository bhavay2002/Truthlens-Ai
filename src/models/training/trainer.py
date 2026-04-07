"""
File Name: trainer.py
Module: models.training
Description:
    Implements the training engine for TruthLens models. This module provides a
    reusable Trainer abstraction responsible for coordinating the full training
    lifecycle including forward passes, backpropagation, gradient accumulation,
    optimizer steps, scheduler updates, checkpointing hooks, and metric logging.

    The trainer is framework-agnostic with respect to the model architecture and
    supports both single-task and multi-task models that return either dictionaries
    or structured outputs.

    Designed for research reproducibility and production ML pipelines.

Dependencies:
    logging
    typing
    dataclasses
    torch
    torch.nn
    torch.optim
Inputs:
    Model
    Training DataLoader
    Validation DataLoader
Outputs:
    Training history and trained model parameters
"""

from __future__ import annotations

import inspect
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..checkpointing.checkpoint_manager import CheckpointManager
from src.training.checkpointing import (
    list_checkpoints as list_training_checkpoints,
    resume_training as resume_training_checkpoint,
    save_checkpoint as save_training_checkpoint,
)
from src.utils import create_folder, get_device, move_to_device
from ..metadata.model_card import (
    DatasetInfo,
    EthicalConsiderations,
    EvaluationResults,
    ModelArtifacts as ModelCardArtifacts,
    ModelCard,
    ModelDetails,
    TrainingConfig as CardTrainingConfig,
)
from ..config import ModelConfigLoader, MultiTaskModelConfig
from ..metadata.model_metadata import (
    ArtifactPaths,
    ModelIdentity,
    ModelMetadata,
    RuntimeEnvironment,
    TrainingProvenance,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """
    Configuration for the training process.
    """

    epochs: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: Optional[str] = None
    log_every_steps: int = 50
    checkpoint_dir: Optional[str] = None
    drive_checkpoint_dir: Optional[str] = None
    checkpoint_every_steps: int = 0
    max_checkpoints: int = 3
    model_name: str = "truthlens_model"
    model_version: str = "1.0.0"
    architecture: str = "transformer"
    dataset_name: str = "unknown"
    framework: str = "pytorch"
    author: str = "TruthLens"


class Trainer:
    """
    Generic trainer for TruthLens models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: TrainerConfig,
    ) -> None:

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch optimizer")

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        self.device = torch.device(config.device) if config.device else get_device(prefer_gpu=True)

        self.model.to(self.device)
        self.global_step = 0

        self.checkpoint_manager: Optional[CheckpointManager] = None
        if config.checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(Path(config.checkpoint_dir))
            self._attempt_resume()

        logger.info("Trainer initialized on device %s", self.device)

    def _attempt_resume(self) -> None:
        if not self.config.checkpoint_dir:
            return

        checkpoint_root = Path(self.config.checkpoint_dir)
        available = list_training_checkpoints(checkpoint_root)
        if not available:
            return

        latest = available[-1]
        try:
            state = resume_training_checkpoint(
                self.model,
                checkpoint_dir=latest,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )
            self.global_step = int(state.get("start_step", 0) or 0)
            logger.info(
                "Resumed trainer state from %s | step=%s",
                latest,
                self.global_step,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Checkpoint resume skipped: %s", exc)

    def _save_training_checkpoint(
        self,
        *,
        epoch: Optional[int],
        step: int,
        metadata: Dict[str, Any],
    ) -> None:
        if not self.config.checkpoint_dir:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir) / f"step_{step}"
        try:
            save_training_checkpoint(
                self.model,
                checkpoint_dir=checkpoint_dir,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                step=step,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unified checkpoint save skipped: %s", exc)
            return

        self._sync_checkpoint_to_drive(checkpoint_dir)

    def _sync_checkpoint_to_drive(self, source_dir: Path) -> None:
        """
        Mirror a local checkpoint directory into Google Drive.

        If ``config.drive_checkpoint_dir`` is not set, or if Drive is not
        mounted, this is a no-op and a warning is logged rather than raising.

        Parameters
        ----------
        source_dir : Path
            Local directory that was just written by the checkpoint manager or
            ``save_training_checkpoint``.
        """

        if not self.config.drive_checkpoint_dir:
            return

        drive_root = Path(self.config.drive_checkpoint_dir)

        if not drive_root.exists():
            try:
                drive_root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "Google Drive checkpoint sync skipped — could not create "
                    "drive directory %s: %s",
                    drive_root,
                    exc,
                )
                return

        dest = drive_root / source_dir.name
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(source_dir, dest)
            logger.info(
                "Checkpoint synced to Google Drive: %s → %s", source_dir, dest
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Google Drive checkpoint sync failed for %s: %s", source_dir, exc
            )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Execute full training loop.
        """

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(self.config.epochs):

            logger.info("Starting epoch %d/%d", epoch + 1, self.config.epochs)

            train_loss = self._train_epoch(train_loader, epoch)

            history["train_loss"].append(train_loss)

            logger.info("Epoch %d training loss: %.6f", epoch + 1, train_loss)

            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_checkpoint(
                    step=self.global_step,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler is not None else None
                    ),
                    metadata={
                        "epoch": epoch + 1,
                        "train_loss": float(train_loss),
                    },
                )
                self._sync_checkpoint_to_drive(
                    Path(self.config.checkpoint_dir) / f"checkpoint-{self.global_step}"
                )
                self.checkpoint_manager.cleanup_old_checkpoints(
                    max_checkpoints=self.config.max_checkpoints
                )
                self._save_training_checkpoint(
                    epoch=epoch + 1,
                    step=self.global_step,
                    metadata={
                        "epoch": epoch + 1,
                        "train_loss": float(train_loss),
                    },
                )

            if val_loader is not None:

                val_loss = self._validate_epoch(val_loader)

                history["val_loss"].append(val_loss)

                logger.info("Epoch %d validation loss: %.6f", epoch + 1, val_loss)

        return history

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """
        Train model for a single epoch.
        """

        self.model.train()

        total_loss = 0.0
        step_count = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(dataloader):

            batch = self._move_batch_to_device(batch)

            outputs = self.model(**self._prepare_model_inputs(batch))

            loss = self._extract_loss(outputs)

            loss = loss / self.config.gradient_accumulation_steps

            loss.backward()

            total_loss += loss.item()
            step_count += 1
            self.global_step += 1

            if (step + 1) % self.config.gradient_accumulation_steps == 0:

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()

            if (
                self.checkpoint_manager is not None
                and self.config.checkpoint_every_steps > 0
                and self.global_step % self.config.checkpoint_every_steps == 0
            ):
                self.checkpoint_manager.save_checkpoint(
                    step=self.global_step,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=(
                        self.scheduler.state_dict() if self.scheduler is not None else None
                    ),
                    metadata={
                        "epoch": epoch + 1,
                        "step_loss": float(loss.item()),
                    },
                )
                self._sync_checkpoint_to_drive(
                    Path(self.config.checkpoint_dir) / f"checkpoint-{self.global_step}"
                )
                self.checkpoint_manager.cleanup_old_checkpoints(
                    max_checkpoints=self.config.max_checkpoints
                )
                self._save_training_checkpoint(
                    epoch=epoch + 1,
                    step=self.global_step,
                    metadata={
                        "epoch": epoch + 1,
                        "step_loss": float(loss.item()),
                    },
                )

            if (step + 1) % self.config.log_every_steps == 0:
                logger.info(
                    "Training step %d | loss %.6f",
                    step + 1,
                    loss.item(),
                )

        avg_loss = total_loss / max(step_count, 1)

        return avg_loss

    def _validate_epoch(self, dataloader: DataLoader) -> float:
        """
        Run validation loop.
        """

        self.model.eval()

        total_loss = 0.0
        step_count = 0

        with torch.no_grad():

            for batch in dataloader:

                batch = self._move_batch_to_device(batch)

                outputs = self.model(**self._prepare_model_inputs(batch))

                loss = self._extract_loss(outputs)

                total_loss += loss.item()
                step_count += 1

        avg_loss = total_loss / max(step_count, 1)

        return avg_loss

    def _extract_loss(self, outputs: Any) -> torch.Tensor:
        """
        Extract loss tensor from model output.
        """

        if isinstance(outputs, dict):

            if "loss" not in outputs:
                raise RuntimeError("Model output dictionary must contain 'loss'")

            return outputs["loss"]

        if hasattr(outputs, "loss"):
            return outputs.loss

        raise RuntimeError("Unable to extract loss from model output")

    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Move batch tensors to the configured device.
        """

        if not isinstance(batch, dict):
            raise TypeError("Batch must be a dictionary")

        return move_to_device(batch, self.device)

    def _prepare_model_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a flat batch dict to the model's ``forward()`` signature.

        Keys that appear in the ``forward()`` signature are passed directly.
        All remaining keys (e.g. task-specific label fields like
        ``hero_entities``, ``bias_label``) are collected into a ``labels``
        dictionary, which is then forwarded under the ``labels`` kwarg if the
        model accepts it.  If the model does not accept ``labels`` at all the
        extra keys are silently dropped so that the call never raises a
        ``TypeError`` for unexpected keyword arguments.

        Parameters
        ----------
        batch : dict
            Device-resident batch produced by ``_move_batch_to_device``.

        Returns
        -------
        dict
            Keyword-argument dict ready for ``self.model(**...)``.
        """

        try:
            sig = inspect.signature(self.model.forward)
            accepted = set(sig.parameters.keys()) - {"self"}
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
        except (ValueError, TypeError):
            return batch

        # If the model accepts **kwargs it will handle anything — pass as-is.
        if has_var_keyword:
            return batch

        forward_kwargs: Dict[str, Any] = {}
        label_dict: Dict[str, Any] = {}

        for key, value in batch.items():
            if key in accepted:
                forward_kwargs[key] = value
            else:
                label_dict[key] = value

        # Bundle extra keys into `labels` when the signature supports it.
        if label_dict and "labels" in accepted:
            existing = forward_kwargs.get("labels")
            if isinstance(existing, dict):
                existing.update(label_dict)
            else:
                forward_kwargs["labels"] = label_dict

        return forward_kwargs

    @classmethod
    def from_model_config(
        cls,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        model_config: MultiTaskModelConfig,
        scheduler: Optional[Any] = None,
        overrides: Optional["TrainerConfig"] = None,
    ) -> "Trainer":
        """
        Build a ``Trainer`` pre-populated with settings derived from a
        ``MultiTaskModelConfig``.

        The ``model_name`` field of ``TrainerConfig`` is set to
        ``model_config.encoder.model_name`` and ``architecture`` is set to the
        class name of ``model``.  Any field in ``overrides`` that differs from
        the default ``TrainerConfig`` takes precedence.

        Parameters
        ----------
        model:
            Instantiated PyTorch model.
        optimizer:
            Configured optimizer.
        model_config:
            Central model configuration loaded via
            ``ModelConfigLoader.load_multitask_config()``.
        scheduler:
            Optional learning-rate scheduler.
        overrides:
            Optional ``TrainerConfig`` whose non-default fields override the
            values derived from ``model_config``.

        Returns
        -------
        Trainer
        """
        from dataclasses import replace as _replace

        base = overrides if overrides is not None else TrainerConfig()
        effective = _replace(
            base,
            model_name=model_config.encoder.model_name,
            architecture=type(model).__name__,
        )
        logger.info(
            "Trainer.from_model_config | model=%s architecture=%s",
            effective.model_name,
            effective.architecture,
        )
        return cls(model=model, optimizer=optimizer, scheduler=scheduler, config=effective)

    @classmethod
    def from_yaml_config(
        cls,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        yaml_path: str | Path,
        scheduler: Optional[Any] = None,
        overrides: Optional["TrainerConfig"] = None,
    ) -> "Trainer":
        model_config = ModelConfigLoader.load_multitask_config(yaml_path)
        return cls.from_model_config(
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            scheduler=scheduler,
            overrides=overrides,
        )

    def save_model_config(
        self,
        output_dir: str | Path,
        model_config: MultiTaskModelConfig,
    ) -> Path:
        """
        Serialise a ``MultiTaskModelConfig`` to ``config.yaml`` alongside
        trained artifacts.

        Parameters
        ----------
        output_dir:
            Directory where ``config.yaml`` will be written.
        model_config:
            Structured model configuration to persist.

        Returns
        -------
        Path
            Path to the saved ``config.yaml`` file.
        """
        import yaml

        output_path = Path(output_dir)
        create_folder(output_path)

        out_dict: Dict[str, Any] = {
            "encoder": {
                "model_name": model_config.encoder.model_name,
                "pooling": model_config.encoder.pooling,
                "device": model_config.encoder.device,
            },
            "tasks": {
                name: {
                    "num_labels": tc.num_labels,
                    "task_type": tc.task_type,
                }
                for name, tc in model_config.tasks.items()
            },
            "dropout": model_config.dropout,
            "metadata": model_config.metadata,
        }

        out_path = output_path / "config.yaml"
        with open(out_path, "w", encoding="utf-8") as _f:
            yaml.safe_dump(out_dict, _f, default_flow_style=False, allow_unicode=True)

        logger.info("MultiTaskModelConfig saved: %s", out_path)
        return out_path

    def save_model_metadata(
        self,
        output_dir: str | Path,
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_dir: Optional[str] = None,
    ) -> Path:
        """
        Build and persist a ModelMetadata file alongside trained artifacts.

        Parameters
        ----------
        output_dir : str | Path
            Directory where metadata.json will be written.
        metrics : dict, optional
            Evaluation metrics to embed in the metadata.
        checkpoint_dir : str, optional
            Override for checkpoint directory in artifact paths.

        Returns
        -------
        Path
            Path to the saved metadata.json file.
        """

        output_path = create_folder(output_dir)

        identity = ModelIdentity(
            model_name=self.config.model_name,
            version=self.config.model_version,
            architecture=self.config.architecture,
        )

        provenance = TrainingProvenance(
            dataset_name=self.config.dataset_name,
            dataset_version=None,
            experiment_name=None,
            run_id=None,
            framework=self.config.framework,
            seed=None,
        )

        artifacts = ArtifactPaths(
            model_weights=str(output_path / "model.pt"),
            config_file=str(output_path / "config.json"),
            tokenizer_path=str(output_path / "tokenizer"),
            training_logs=None,
            checkpoint_directory=checkpoint_dir or self.config.checkpoint_dir,
        )

        runtime = RuntimeEnvironment(
            python_version=sys.version.split()[0],
            framework_version=torch.__version__,
            cuda_version=torch.version.cuda,
            hardware=str(self.device),
            device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )

        metadata = ModelMetadata(
            identity=identity,
            provenance=provenance,
            artifacts=artifacts,
            runtime=runtime,
            metrics=metrics,
        )

        save_path = metadata.save_json(output_path / "metadata.json")
        logger.info("ModelMetadata saved: %s", save_path)
        return save_path

    def save_model_card(
        self,
        output_dir: str | Path,
        metrics: Optional[Dict[str, float]] = None,
        dataset_source: Optional[str] = None,
    ) -> Path:
        """
        Build and persist a ModelCard (JSON + Markdown) alongside trained artifacts.

        Parameters
        ----------
        output_dir : str | Path
            Directory where model_card.json and model_card.md will be written.
        metrics : dict, optional
            Evaluation metrics to embed in the card.
        dataset_source : str, optional
            Human-readable source description for the training dataset.

        Returns
        -------
        Path
            Path to the saved model_card.json file.
        """

        output_path = create_folder(output_dir)

        details = ModelDetails(
            name=self.config.model_name,
            version=self.config.model_version,
            architecture=self.config.architecture,
            description=f"TruthLens {self.config.architecture} model for misinformation detection.",
            author=self.config.author,
        )

        dataset_info = DatasetInfo(
            name=self.config.dataset_name,
            source=dataset_source,
        )

        training_cfg = CardTrainingConfig(
            framework=self.config.framework,
            epochs=self.config.epochs,
            batch_size=1,
            optimizer="adam",
            learning_rate=1e-5,
            hardware=str(self.device),
        )

        eval_metrics: Dict[str, float] = metrics if metrics else {"placeholder": 0.0}
        evaluation = EvaluationResults(metrics=eval_metrics)

        ethics = EthicalConsiderations(
            intended_use="Misinformation and propaganda detection in news text.",
            out_of_scope_use="Not intended for legal decisions or high-stakes autonomous actions.",
            limitations="Performance may degrade on domains not covered in training data.",
            bias_risks="Potential bias from training data distribution.",
        )

        card_artifacts = ModelCardArtifacts(
            model_weights=str(output_path / "model.pt"),
            tokenizer=str(output_path / "tokenizer"),
            config_file=str(output_path / "config.json"),
            checkpoint_dir=self.config.checkpoint_dir,
        )

        card = ModelCard(
            model_details=details,
            datasets=[dataset_info],
            training=training_cfg,
            evaluation=evaluation,
            ethics=ethics,
            artifacts=card_artifacts,
        )

        json_path = card.save_json(output_path / "model_card.json")
        card.save_markdown(output_path / "model_card.md")
        logger.info("ModelCard saved: %s", json_path)
        return json_path