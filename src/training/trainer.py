from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.models.config.model_config import ModelConfigLoader

from src.training.training_setup import (
    TrainingSetupConfig,
    setup_runtime,
    optimize_model,
    run_sanity_check,
)

from .training_step import TrainingStep
from .evaluation_engine import EvaluationEngine
from ..models.checkpointing.checkpoint_manager import CheckpointEngine
from .distributed_engine import DistributedEngine
from .experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


# =========================================================
# TRAINER (PRODUCTION-GRADE)
# =========================================================

class Trainer:

    def __init__(
        self,
        config_path: str,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        training_step: TrainingStep,
        evaluator: EvaluationEngine,
        checkpoint: Optional[CheckpointEngine] = None,
        distributed: Optional[DistributedEngine] = None,
        tracker: Optional[ExperimentTracker] = None,
        monitor_metric: str = "val_loss",
        maximize_metric: bool = False,
        params_override: Optional[Dict[str, Any]] = None,
        setup_config: Optional[TrainingSetupConfig] = None,
        log_every_steps: Optional[int] = None,
        checkpoint_every_steps: Optional[int] = None,
    ):

        # -------------------------------------------------
        # CONFIG
        # -------------------------------------------------
        self.cfg = ModelConfigLoader.load_multitask_config(config_path)

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.training_step = training_step
        self.evaluator = evaluator
        self.checkpoint = checkpoint
        self.distributed = distributed
        self.tracker = tracker

        self.monitor_metric = monitor_metric
        self.maximize_metric = maximize_metric

        # -------------------------------------------------
        # TRAINING SETUP
        #
        # CFG-4: ``TrainingSetupConfig`` is ``frozen=True`` (immutability is
        # the right default — callers can't accidentally mutate runtime
        # precision flags mid-training). Previously the Trainer always
        # constructed a default instance, so callers could not disable e.g.
        # ``run_sanity_check`` for fast Optuna trials. Accept an explicit
        # override here as the documented escape hatch.
        # -------------------------------------------------
        self.setup_cfg = setup_config or TrainingSetupConfig()

        self.device = setup_runtime(self.setup_cfg)

        self.model = optimize_model(self.model)

        # GPU-1: the model is moved to its final device EXACTLY ONCE in
        # ``create_trainer_fn`` BEFORE ``build_optimizer`` runs, so that
        # the optimizer captures parameter references already living on
        # the target device. The previous in-place ``self.model.to(self.device)``
        # here was the first of three redundant moves (TrainingStep also
        # did one, DistributedEngine.wrap_model another) and silently
        # broke optimizer state on AMP/CUDA when the model was created on
        # CPU. We now validate device match instead of re-moving — if the
        # caller forgot to move the model first, surface a loud warning
        # rather than papering over a stale-optimizer-state bug.
        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = self.device

        if model_device != self.device:
            logger.warning(
                "GPU-1: Trainer received model on %s but expected %s; "
                "in-place moving (optimizer parameter refs may be stale). "
                "Move the model BEFORE build_optimizer in create_trainer_fn.",
                model_device,
                self.device,
            )
            self.model.to(self.device)

        # -------------------------------------------------
        # TRAINING PARAMS  (BUG-9: honour params["epochs"] from
        # Optuna / hyperparameter tuning if supplied; fall back to
        # the YAML config otherwise.)
        # -------------------------------------------------
        params_override = params_override or {}
        self.epochs = int(
            params_override.get("epochs", self.cfg.training.num_epochs)
        )
        self.early_patience = int(
            params_override.get(
                "early_stopping_patience",
                self.cfg.training.early_stopping_patience,
            )
        )

        self.global_step = 0
        self._epoch = 0
        self.best_metric = None
        self.no_improve_epochs = 0

        # -------------------------------------------------
        # LOGGING / CHECKPOINT CADENCE
        #
        # CFG-3: The previous implementation hardcoded 50 (log) and 500
        # (checkpoint) inside ``_train_epoch``. Both are now driven by
        # ctor args (with the same defaults) so:
        #   * Optuna / fast smoke tests can log every step
        #     (``log_every_steps=1``), and
        #   * long production runs can dial checkpoint cadence up
        #     (e.g. every 5000 steps) without code edits.
        # ``params_override`` also accepts the same keys so a YAML /
        # tuning config can drive both without touching the Trainer call
        # site.
        # -------------------------------------------------
        self.log_every_steps = int(
            log_every_steps
            if log_every_steps is not None
            else params_override.get("log_every_steps", 50)
        )
        self.checkpoint_every_steps = int(
            checkpoint_every_steps
            if checkpoint_every_steps is not None
            else params_override.get("checkpoint_every_steps", 500)
        )

        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be > 0")
        if self.checkpoint_every_steps <= 0:
            raise ValueError("checkpoint_every_steps must be > 0")

        # -------------------------------------------------
        # DISTRIBUTED
        # -------------------------------------------------
        if self.distributed:
            self.distributed.initialize()
            self.model = self.distributed.wrap_model(self.model)

        # -------------------------------------------------
        # LOG CONFIG
        # -------------------------------------------------
        if self.tracker and self._is_main():
            self.tracker.log_params(asdict(self.cfg))

        logger.info("Trainer initialized (PRODUCTION-GRADE)")

    # =====================================================
    # TRAIN ENTRY
    # =====================================================

    def train(self):

        # 🔥 SANITY CHECK (CRITICAL)
        if self.setup_cfg.run_sanity_check:
            self._run_sanity_check()

        for epoch in range(self.epochs):

            self._epoch = epoch

            if self._is_main():
                logger.info("Epoch %d/%d", epoch + 1, self.epochs)

            # DDP sampler sync
            if self.distributed and self.distributed.initialized:
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch()

            # -------------------------
            # VALIDATION
            # -------------------------
            if self.val_loader:

                if self.distributed and self.distributed.initialized:
                    self.distributed.barrier()

                val_metrics = self.evaluate()

                metric_value = val_metrics.get(self.monitor_metric)

                if metric_value is not None:
                    self._update_early_stopping(metric_value)

                # LOGGING
                if self.tracker and self._is_main():
                    self.tracker.log_metrics(val_metrics, step=self.global_step)

                # CHECKPOINT
                if self.checkpoint and self._is_main():
                    self._save_checkpoint(val_metrics)

                # EARLY STOP
                if self.no_improve_epochs >= self.early_patience:
                    if self._is_main():
                        logger.warning("Early stopping triggered")
                    break

        # -------------------------------------------------
        # CLEANUP
        # -------------------------------------------------
        if self.tracker and self._is_main():
            self.tracker.finish()

        if self.distributed:
            self.distributed.cleanup()

    # =====================================================
    # TRAIN EPOCH
    # =====================================================

    def _train_epoch(self):

        for batch in self.train_loader:

            outputs = self.training_step.run(batch, self.global_step)

            self.global_step += 1

            # Skip failed step
            if outputs.get("skipped"):
                continue

            # -------------------------
            # LOGGING
            # -------------------------
            if self.global_step % self.log_every_steps == 0 and self._is_main():

                log_data = {
                    "train/loss": float(outputs.get("raw_loss", 0.0)),
                    "train/grad_norm": outputs.get("grad_norm"),
                    "train/throughput": outputs.get("throughput"),
                }

                logger.info("Step %d | %s", self.global_step, log_data)

                if self.tracker:
                    self.tracker.log_metrics(log_data, step=self.global_step)

            # -------------------------
            # CHECKPOINT  (BUG-4: persist optimizer/scheduler/scaler
            # state so resume restores momentum, LR-step counter and
            # AMP loss-scale.)
            # -------------------------
            if (
                self.checkpoint
                and self.global_step % self.checkpoint_every_steps == 0
                and self._is_main()
            ):
                self.checkpoint.save(
                    step=self.global_step,
                    epoch=self._epoch,
                    model=self._unwrap_model(),
                    optimizer=getattr(self.training_step, "optimizer", None),
                    scheduler=getattr(self.training_step, "scheduler", None),
                    scaler=getattr(self.training_step, "scaler", None),
                )

    # =====================================================
    # SANITY CHECK
    # =====================================================

    def _run_sanity_check(self):

        if not self.train_loader:
            return

        logger.info("Running sanity check...")

        batch = next(iter(self.train_loader))

        run_sanity_check(
            model=self._unwrap_model(),
            batch=batch,
            training_step=self.training_step,
            device=self.device,
        )

        logger.info("Sanity check passed")

    # =====================================================
    # EARLY STOPPING
    # =====================================================

    def _update_early_stopping(self, metric_value: float):

        improved = False

        if self.best_metric is None:
            improved = True
        elif self.maximize_metric:
            improved = metric_value > self.best_metric
        else:
            improved = metric_value < self.best_metric

        if improved:
            self.best_metric = metric_value
            self.no_improve_epochs = 0
        else:
            self.no_improve_epochs += 1

    # =====================================================
    # CHECKPOINT LOGIC
    # =====================================================

    def _save_checkpoint(self, metrics: Dict[str, Any]):

        # BUG-4: persist optimizer / scheduler / scaler state so resume
        # restores momentum, LR-step counter, and AMP loss-scale.
        common_kwargs = dict(
            model=self._unwrap_model(),
            optimizer=getattr(self.training_step, "optimizer", None),
            scheduler=getattr(self.training_step, "scheduler", None),
            scaler=getattr(self.training_step, "scaler", None),
            epoch=self._epoch,
            metrics=metrics,
        )

        # epoch checkpoint
        self.checkpoint.save(
            step=self.global_step,
            **common_kwargs,
        )

        metric_value = metrics.get(self.monitor_metric)

        if metric_value is None:
            return

        # best checkpoint — uses CheckpointEngine's save_best mechanism
        if (
            self.best_metric is not None
            and metric_value == self.best_metric
        ):
            self.checkpoint.save(
                step=self.global_step,
                save_best=True,
                metric_name=self.monitor_metric,
                mode="max" if self.maximize_metric else "min",
                **common_kwargs,
            )

    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(self) -> Dict[str, Any]:

        if not self.val_loader:
            return {}

        model = self._unwrap_model()
        results = self.evaluator.evaluate(model, self.val_loader)

        if self._is_main():
            logger.info("Validation: %s", results)

        return results

    # =====================================================
    # HELPERS
    # =====================================================

    def _unwrap_model(self):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model

    def _is_main(self):
        return not self.distributed or self.distributed.is_main_process()

# =========================================================
# COMPAT: lightweight TrainerConfig dataclass
# =========================================================

from dataclasses import dataclass as _dataclass, field as _field
from typing import Any as _Any, Dict as _Dict, Optional as _Optional


@_dataclass
class TrainerConfig:
    config_path: str = ""
    monitor_metric: str = "val_loss"
    maximize_metric: bool = False

    # CFG-3: explicit cadence knobs (mirror the Trainer.__init__ kwargs).
    log_every_steps: int = 50
    checkpoint_every_steps: int = 500

    # CFG-4: explicit setup-config override (mirror Trainer.__init__).
    setup_config: _Optional["TrainingSetupConfig"] = None

    extras: _Dict[str, _Any] = _field(default_factory=dict)
