from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer

from src.config.settings_loader import load_settings
from src.config.config_loader import load_config

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"

from src.data_processing.data_pipeline import run_data_pipeline, DataPipelineConfig
from src.data_processing.dataloader_factory import DataLoaderConfig
from src.training.create_trainer_fn import create_trainer_fn
from src.pipelines.truthlens_pipeline import TruthLensPipeline
from src.utils.logging_utils import configure_logging
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="truthlens",
        description=(
            "TruthLens entry point. Use --mode infer to run the "
            "analysis pipeline on a few sample texts without loading "
            "any training data; --mode train runs the data + training "
            "stages only; --mode both does both."
        ),
    )
    parser.add_argument("--mode", choices=("train", "infer", "both"), default="infer")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--enable-explainability", action="store_true")
    parser.add_argument("--enable-evaluation", action="store_true")
    parser.add_argument("--no-parallel-stages", action="store_true")
    return parser.parse_args()

def main():
    args = _parse_args()
    try:
        settings = load_settings(validate_data=args.mode in ("train", "both"))
        config = load_config(CONFIG_PATH)
        configure_logging()
        set_seed(config.project.seed)
        logger.info(" TruthLens System Started | mode=%s", args.mode)
        tokenizer = AutoTokenizer.from_pretrained(config.model.encoder)
        if args.mode in ("train", "both"):
            data_config: Dict = {
                "bias": {"train": settings.data.get("bias", "train"), "val": settings.data.get("bias", "val"), "test": settings.data.get("bias", "test")},
                "ideology": {"train": settings.data.get("ideology", "train"), "val": settings.data.get("ideology", "val"), "test": settings.data.get("ideology", "test")},
                "propaganda": {"train": settings.data.get("propaganda", "train"), "val": settings.data.get("propaganda", "val"), "test": settings.data.get("propaganda", "test")},
                "narrative": {"train": settings.data.get("narrative", "train"), "val": settings.data.get("narrative", "val"), "test": settings.data.get("narrative", "test")},
                "narrative_frame": {"train": settings.data.get("narrative_frame", "train"), "val": settings.data.get("narrative_frame", "val"), "test": settings.data.get("narrative_frame", "test")},
                "emotion": {"train": settings.data.get("emotion", "train"), "val": settings.data.get("emotion", "val"), "test": settings.data.get("emotion", "test")},
            }
            loader_cfg = DataLoaderConfig.from_yaml_data(config.data)
            datasets = run_data_pipeline(
                data_config=data_config,
                tokenizer=tokenizer,
                build_dataloaders=False,
                config=DataPipelineConfig(enable_cache=True, dataloader_config=loader_cfg),
            )
            logger.info("✅ Data pipeline completed")

            # ``narrative_frame`` only exists inside the multitask spec
            # (a 5-label head); the single-task model factory has no
            # mapping for it, so skip it here with a clear note.
            SINGLE_TASK_SUPPORTED = {
                "bias", "ideology", "propaganda", "narrative", "emotion",
            }
            unsupported = [t for t in datasets if t not in SINGLE_TASK_SUPPORTED]
            if unsupported:
                logger.info(
                    "Skipping tasks not wired into single-task training: %s "
                    "(only the multitask path covers these heads)",
                    unsupported,
                )

            trainers = {}
            for task in datasets:
                if task not in SINGLE_TASK_SUPPORTED:
                    continue
                logger.info("🧠 Creating trainer | task=%s", task)
                trainer = create_trainer_fn(
                    task=task,
                    train_df=datasets[task]["train"],
                    val_df=datasets[task]["val"],
                    params={
                        # YAML scalars like ``3e-5`` (no decimal point) are
                        # parsed as ``str`` by PyYAML — coerce to ``float``
                        # before handing off to torch.optim, which checks
                        # ``0.0 <= lr`` and crashes on a string compare.
                        "lr": float(config.optimizer.lr),
                        "batch_size": int(loader_cfg.batch_size),
                        "weight_decay": float(config.optimizer.weight_decay),
                        # Cap epochs at 1 for the CPU smoke run — 4 epochs ×
                        # 5 roberta-base finetunes is unworkably slow on CPU
                        # and the goal here is "produce a checkpoint", not
                        # "fully train". Override via env var if needed.
                        "epochs": int(os.environ.get(
                            "TRUTHLENS_TRAIN_EPOCHS", "1"
                        )),
                        # Force ``num_workers=0`` for the smoke run.
                        # Each per-task DataLoader otherwise forks
                        # ``num_workers`` × 2 (train + val) child
                        # processes that fork the *current* RAM image
                        # (which by then holds previously-trained
                        # roberta-base weights). Across 5 sequential
                        # tasks these forks pile up, sometimes causing
                        # the parent to be silently reaped between
                        # tasks. Single-process loading is fast enough
                        # for the 10-row demo data and removes the
                        # cross-task multiprocessing fragility.
                        "num_workers": 0,
                        "pin_memory": bool(loader_cfg.pin_memory),
                        "tokenizer": tokenizer,
                        "model_name": config.model.encoder,
                        "dropout": float(config.model.dropout),
                    },
                )
                trainers[task] = trainer

            # Save the checkpoint INCREMENTALLY after each task so a
            # crash midway through (e.g. a CPU OOM on the 4th roberta-
            # base finetune) still leaves a usable checkpoint behind
            # for the inference path. The unified shape is the
            # ``{"model": nn.Module}`` dict that ``main.py --mode infer``
            # expects via ``state.get("model")`` — re-saving simply
            # overwrites with the most recently trained head.
            import gc
            save_dir = Path("saved_models")
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / "checkpoint.pt"
            last_trained_task = None
            for task, trainer in trainers.items():
                logger.info("🔥 Training | task=%s", task)
                trainer.train()
                last_trained_task = task
                torch.save(
                    {
                        "model": trainer.model.cpu().eval(),
                        "task": task,
                        "encoder": config.model.encoder,
                    },
                    ckpt_path,
                )
                logger.info(
                    "📦 Saved checkpoint → %s (task=%s)", ckpt_path, task,
                )
                # Drop references and force GC before the next per-task
                # roberta-base finetune so 5 sequential trainers don't
                # accumulate ~700MB each in resident memory.
                trainer.model = None
                trainers[task] = None
                del trainer
                gc.collect()
        if args.mode in ("infer", "both"):
            logger.info("🧪 Running FULL TruthLens pipeline")
            model_version = getattr(getattr(config, "model", object()), "version", config.model.encoder)
            predictor = None
            try:
                from src.models.inference.predictor import Predictor
                from src.utils import settings as _runtime_settings
                runtime = _runtime_settings.load_settings()
                model_dir = Path(getattr(runtime.model, "path", "saved_models")).resolve()
                checkpoint_file = model_dir / "checkpoint.pt"
                if checkpoint_file.is_file():
                    logger.info(" Found checkpoint at %s — loading Predictor", checkpoint_file)
                    # ``weights_only=False`` is required because the
                    # checkpoint contains a *pickled* ``nn.Module``
                    # (``state["model"]``), not just a state-dict. Newer
                    # PyTorch defaults to ``weights_only=True`` which
                    # rejects arbitrary pickled objects. We control both
                    # the writer (``main.py --mode train``) and the
                    # reader, so opting out of the safe-load is fine.
                    state = torch.load(
                        checkpoint_file,
                        map_location="cpu",
                        weights_only=False,
                    )
                    multitask_model = state.get("model") if isinstance(state, dict) else None
                    if not isinstance(multitask_model, torch.nn.Module):
                        logger.warning(" Checkpoint at %s does not contain a serialised nn.Module under key 'model' — running without a predictor. Re-export the checkpoint with the training pipeline to enable prediction.", checkpoint_file)
                    else:
                        predictor = Predictor(model=multitask_model)
                        logger.info(" Predictor attached")
                else:
                    logger.warning(" No checkpoint at %s — running the analysis / features / aggregation stack only. Run `python main.py --mode train` (after placing the dataset CSVs under data/{train,val,test}/) to produce a checkpoint and unlock prediction.", checkpoint_file)
            except Exception:
                logger.exception(" Predictor load failed — continuing without prediction")
                predictor = None
            pipeline = TruthLensPipeline(predictor=predictor, tokenizer=tokenizer, model_version=model_version, enable_explainability=args.enable_explainability, enable_evaluation=args.enable_evaluation, parallel_stages=not args.no_parallel_stages)
            sample_texts = ["The government clearly failed the people.", "This is a neutral statement.", "The heroic leader saved the nation."][: max(1, args.num_samples)]
            batch_result = pipeline.analyze_batch(sample_texts)
            logger.info(" BATCH SUMMARY: n=%d total_time=%.3fs model_version=%s", batch_result["batch_metadata"]["n_articles"], batch_result["batch_metadata"]["total_time"], batch_result["batch_metadata"]["model_version"])
            for i, result in enumerate(batch_result["articles"]):
                logger.info(" RESULT %d:", i + 1)
                logger.info("  scores: %s", result.get("scores"))
                logger.info("  predictions keys: %s", list(result.get("predictions", {}).keys()))
                if result.get("errors"):
                    logger.warning("  stage errors: %s", result["errors"])
            pipeline.close()
        logger.info(" SYSTEM COMPLETED SUCCESSFULLY")
    except Exception as e:
        logger.error(" SYSTEM FAILED: %s", str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
