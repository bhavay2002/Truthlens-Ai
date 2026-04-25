from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentTrackerConfig:
    backend: str = "none"
    project_name: str = "truthlens"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


class ExperimentTracker:

    def __init__(self, config: Optional[ExperimentTrackerConfig] = None):
        self.config = config or ExperimentTrackerConfig()
        self.backend = self.config.backend.lower()
        self._step = 0

        self._init_backend()

        logger.info("ExperimentTracker initialized | backend=%s", self.backend)

    # =====================================================
    # UTILS
    # =====================================================

    def _is_main(self):
        try:
            import torch.distributed as dist
            return not dist.is_initialized() or dist.get_rank() == 0
        except Exception:
            return True

    def _safe(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.warning("Tracker error: %s", e)

    # =====================================================
    # INIT
    # =====================================================

    def _init_backend(self):

        if self.backend == "mlflow":
            import mlflow

            if self.config.tracking_uri:
                mlflow.set_tracking_uri(self.config.tracking_uri)

            mlflow.set_experiment(self.config.project_name)

            mlflow.start_run(run_name=self.config.run_name)

            if self.config.tags:
                mlflow.set_tags(self.config.tags)

        elif self.backend == "wandb":
            import wandb

            wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                tags=list(self.config.tags.keys()),
                config={}
            )

        elif self.backend == "none":
            return

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    # =====================================================
    # LOGGING
    # =====================================================

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):

        if not self._is_main():
            return

        step = step if step is not None else self._step
        self._step += 1

        if self.backend == "mlflow":
            import mlflow
            for k, v in metrics.items():
                self._safe(mlflow.log_metric, k, float(v), step=step)

        elif self.backend == "wandb":
            import wandb
            self._safe(wandb.log, metrics, step=step)

    def log_params(self, params: Dict[str, Any]):

        if not self._is_main():
            return

        if self.backend == "mlflow":
            import mlflow
            self._safe(mlflow.log_params, params)

        elif self.backend == "wandb":
            import wandb
            self._safe(wandb.config.update, params, allow_val_change=True)

    def log_artifact(self, path: str):

        if not self._is_main():
            return

        if self.backend == "mlflow":
            import mlflow
            self._safe(mlflow.log_artifact, path)

        elif self.backend == "wandb":
            import wandb
            self._safe(wandb.save, path)

    # =====================================================
    # EXTRA
    # =====================================================

    def watch_model(self, model):
        if self.backend == "wandb":
            import wandb
            self._safe(wandb.watch, model)

    # =====================================================
    # FINALIZE
    # =====================================================

    def finish(self):

        if not self._is_main():
            return

        if self.backend == "mlflow":
            import mlflow
            self._safe(mlflow.end_run)

        elif self.backend == "wandb":
            import wandb
            self._safe(wandb.finish)

    # =====================================================
    # CONTEXT MANAGER
    # =====================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish()