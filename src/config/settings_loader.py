from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =========================================================
# PATH SETTINGS
# =========================================================

@dataclass(frozen=True)
class PathSettings:
    project_root: Path
    data_dir: Path
    artifacts_dir: Path
    models_dir: Path
    logs_dir: Path
    checkpoints_dir: Path

    training_log_path: Path
    evaluation_results_path: Path

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)


# =========================================================
# RUNTIME FLAGS (ENV-DRIVEN)
# =========================================================

@dataclass(frozen=True)
class RuntimeFlags:
    require_gpu: bool = False
    debug: bool = False


# =========================================================
# GLOBAL SETTINGS
# =========================================================

@dataclass(frozen=True)
class Settings:
    paths: PathSettings
    runtime: RuntimeFlags = field(default_factory=RuntimeFlags)


# =========================================================
# HELPERS
# =========================================================

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _env_path(key: str, default: Path) -> Path:
    val = os.environ.get(key)
    return Path(val).expanduser().resolve() if val else default.resolve()


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y"}


# =========================================================
# MAIN
# =========================================================

def load_settings() -> Settings:
    root = _project_root()

    artifacts_dir = _env_path("TRUTHLENS_ARTIFACTS_DIR", root / "artifacts")
    data_dir = _env_path("TRUTHLENS_DATA_DIR", root / "data")
    models_dir = _env_path("TRUTHLENS_MODELS_DIR", artifacts_dir / "models")
    logs_dir = _env_path("TRUTHLENS_LOGS_DIR", artifacts_dir / "logs")
    checkpoints_dir = _env_path("TRUTHLENS_CKPT_DIR", artifacts_dir / "checkpoints")

    paths = PathSettings(
        project_root=root,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        models_dir=models_dir,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        training_log_path=logs_dir / "training.log",
        evaluation_results_path=models_dir / "evaluation.json",
    )

    paths.ensure_dirs()

    runtime = RuntimeFlags(
        require_gpu=_env_bool("TRUTHLENS_REQUIRE_GPU", False),
        debug=_env_bool("TRUTHLENS_DEBUG", False),
    )

    return Settings(paths=paths, runtime=runtime)