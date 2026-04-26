from __future__ import annotations

import logging
import sys
from typing import Dict

from transformers import AutoTokenizer

# =========================
# CONFIG
# =========================

from src.config.settings_loader import load_settings
from src.config.config_loader import load_config

# =========================
# DATA PIPELINE
# =========================

from src.data.data_pipeline import run_data_pipeline, DataPipelineConfig

# =========================
# TRAINING
# =========================

from src.training.create_trainer_fn import create_trainer_fn

# =========================
# FULL PIPELINE ( CORE)
# =========================

from src.pipelines.truthlens_pipeline import TruthLensPipeline

# =========================
# UTILS
# =========================

from src.utils.logging_utils import configure_logging
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


# =========================================================
# MAIN
# =========================================================

def main():

    try:

        # -------------------------------------------------
        # 1. INIT
        # -------------------------------------------------

        settings = load_settings()
        config = load_config()

        configure_logging()
        set_seed(settings.project.seed)

        logger.info(" TruthLens System Started")

        # -------------------------------------------------
        # 2. DATA CONFIG
        # -------------------------------------------------

        data_config: Dict = {
            "bias": {
                "train": settings.data.bias_train,
                "val": settings.data.bias_val,
                "test": settings.data.bias_test,
            },
            "ideology": {
                "train": settings.data.ideology_train,
                "val": settings.data.ideology_val,
                "test": settings.data.ideology_test,
            },
            "propaganda": {
                "train": settings.data.propaganda_train,
                "val": settings.data.propaganda_val,
                "test": settings.data.propaganda_test,
            },
            "narrative": {
                "train": settings.data.narrative_train,
                "val": settings.data.narrative_val,
                "test": settings.data.narrative_test,
            },
            "narrative_frame": {
                "train": settings.data.frame_train,
                "val": settings.data.frame_val,
                "test": settings.data.frame_test,
            },
            "emotion": {
                "train": settings.data.emotion_train,
                "val": settings.data.emotion_val,
                "test": settings.data.emotion_test,
            },
        }

        # -------------------------------------------------
        # 3. TOKENIZER
        # -------------------------------------------------

        tokenizer = AutoTokenizer.from_pretrained(config.model.encoder)

        # -------------------------------------------------
        # 4. DATA PIPELINE
        # -------------------------------------------------

        datasets = run_data_pipeline(
            data_config=data_config,
            tokenizer=tokenizer,
            build_dataloaders=False,
            config=DataPipelineConfig(enable_cache=True),
        )

        logger.info("✅ Data pipeline completed")

        # -------------------------------------------------
        # 5. TRAINING
        # -------------------------------------------------

        trainers = {}

        for task in datasets:

            logger.info("🧠 Creating trainer | task=%s", task)

            trainer = create_trainer_fn(
                task=task,
                train_df=datasets[task]["train"],
                val_df=datasets[task]["val"],
                params={
                    "lr": config.optimizer.lr,
                    "batch_size": config.data.batch_size,
                    "weight_decay": config.optimizer.weight_decay,
                    "epochs": config.training.epochs,
                },
            )

            trainers[task] = trainer

        for task, trainer in trainers.items():

            logger.info("🔥 Training | task=%s", task)
            trainer.train()

        # -------------------------------------------------
        # 6.  FULL PIPELINE (ALL SYSTEMS USED)
        # -------------------------------------------------

        logger.info("🧪 Running FULL TruthLens pipeline")

        pipeline = TruthLensPipeline(
            enable_explainability=True,
            enable_evaluation=True,
        )

        sample_texts = [
            "The government clearly failed the people.",
            "This is a neutral statement.",
            "The heroic leader saved the nation.",
        ]

        for text in sample_texts:

            result = pipeline.analyze(text)

            logger.info(" RESULT SUMMARY:")
            logger.info("Scores: %s", result.get("scores"))
            logger.info("Predictions: %s", result.get("predictions"))

        logger.info(" SYSTEM COMPLETED SUCCESSFULLY")

    except Exception as e:

        logger.error(" SYSTEM FAILED: %s", str(e), exc_info=True)
        sys.exit(1)


# =========================================================
# ENTRYPOINT
# =========================================================

if __name__ == "__main__":
    main()