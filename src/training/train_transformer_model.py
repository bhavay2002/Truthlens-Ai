"""
File Name: train_truthlens_model.py
Module: training
Description:
    Production training pipeline for the TruthLens transformer-based
    misinformation detection system.

    This module provides a unified training interface for transformer
    architectures including RoBERTa, DeBERTa, and Longformer. It handles
    dataset validation, train/validation/test splitting, tokenizer
    preparation, model initialization, training orchestration using the
    HuggingFace Trainer API, evaluation metrics, checkpoint management,
    and artifact persistence.

    The implementation follows research-grade ML engineering standards
    and is designed to support future model upgrades without changing
    the training pipeline.

Dependencies:
    logging
    math
    pathlib
    typing
    numpy
    pandas
    torch
    datasets
    sklearn
    transformers
    src.utils.input_validation
    src.utils.settings
Inputs:
    df : pandas.DataFrame
        Must contain:
            text : str
            label : int or categorical

    params : dict
        Optional hyperparameters

Outputs:
    transformers.Trainer
    datasets.Dataset (test)
"""
from __future__ import annotations

import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint as hf_get_last_checkpoint
from src.models.multitask.multitask_model import TruthLensMultiTaskModel

from src.utils.input_validation import ensure_dataframe, ensure_non_empty_text_column
from src.utils.helper_functions import create_folder
from src.utils.seed_utils import set_seed
from src.utils.settings import load_settings
from src.models.training.training_utils import TrainingMetrics, get_device
from src.features.dataset_feature_generator import DatasetFeatureGenerator
from src.features.feature_schema_validator import FeatureSchemaValidator
from src.features.feature_statistics import FeatureStatistics
from src.models.export import (
    ONNXExportConfig,
    ONNXExporter,
    QuantizationConfig,
    QuantizationEngine,
    TorchScriptExportConfig, 
    TorchScriptExporter,
)
from src.models.checkpointing.artifact_manager import ArtifactManager
from src.models.checkpointing.checkpoint_manager import CheckpointManager
from src.training.checkpointing import (
    save_checkpoint as save_training_checkpoint,
)
from src.models.metadata.model_card import (
    DatasetInfo,
    EthicalConsiderations,
    EvaluationResults,
    ModelArtifacts as ModelCardArtifacts,
    ModelCard,
    ModelDetails,
    TrainingConfig as CardTrainingConfig,
)
from src.models.metadata.model_metadata import (
    ArtifactPaths,
    ModelIdentity,
    ModelMetadata,
    RuntimeEnvironment,
    TrainingProvenance,
)
from src.models.metadata.model_versioning import ModelVersionInfo, ModelVersionRegistry

logger = logging.getLogger(__name__)

SETTINGS = load_settings()

MODEL_NAME = SETTINGS.model.name
MAX_LENGTH = SETTINGS.model.max_length

SEED = SETTINGS.training.seed
DEFAULT_EPOCHS = SETTINGS.training.epochs
DEFAULT_BATCH_SIZE = SETTINGS.training.batch_size
DEFAULT_LEARNING_RATE = SETTINGS.training.learning_rate
DEFAULT_RESUME_FROM_CHECKPOINT = SETTINGS.training.resume_from_checkpoint

DEFAULT_VALIDATION_SIZE = SETTINGS.training.validation_size
DEFAULT_TEST_SIZE = SETTINGS.training.test_size

MODELS_DIR = Path("/content/drive/MyDrive/truthlens-models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_save_steps = 1000
FREE_COLAB_NUM_PROC = 2
TOKENIZED_DATASET_CACHE_DIR = Path("/content/tokenized_dataset")
LOGS_DIR = Path("/content/drive/MyDrive/truthlens-logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
GOOGLE_DRIVE_REPORTS_DIR = Path("/content/drive/MyDrive/truthlens-reports")
GOOGLE_DRIVE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
GOOGLE_DRIVE_CHECKPOINTS_DIR = Path("/content/drive/MyDrive/truthlens-checkpoints")
GOOGLE_DRIVE_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(SETTINGS.model.path)
TEST_SET_PATH = Path(SETTINGS.data.test_set_path)


class _HFExportWrapper(nn.Module):
    """
    Adapter to expose HuggingFace classification model with tensor-only forward.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids)
        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, dict):
            bias_outputs = outputs.get("bias")
            if isinstance(bias_outputs, dict):
                logits = bias_outputs.get("logits")
        if logits is None:
            raise RuntimeError("Expected HuggingFace model output to include logits.")
        return logits


def _entity_to_binary(entity_field) -> int:
    if entity_field is None:
        return 0
    if isinstance(entity_field, str) and entity_field.strip() == "":
        return 0
    return 1


def _quant_backend() -> str:
    supported = list(torch.backends.quantized.supported_engines)
    if "fbgemm" in supported:
        return "fbgemm"
    if "qnnpack" in supported:
        return "qnnpack"
    return supported[0] if supported else "qnnpack"


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    unique_labels = np.unique(labels)
    average = "binary" if len(unique_labels) <= 2 else "macro"

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average=average,
        zero_division=0,
    )

    acc = accuracy_score(labels, preds)

    try:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        if probs.shape[1] == 2:
            roc_auc = roc_auc_score(labels, probs[:, 1])
        else:
            roc_auc = roc_auc_score(labels, probs, multi_class="ovr")
    except Exception:
        roc_auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def tokenize_function(example: dict, tokenizer, text_column: str):
    return tokenizer(
        example[text_column],
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
    )


def get_last_checkpoint(directory: Path) -> str | None:
    if not directory.exists():
        return None
    try:
        return hf_get_last_checkpoint(str(directory))
    except Exception:
        return None


def _split_train_val_test(
    df: pd.DataFrame,
    *,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    holdout_size = DEFAULT_VALIDATION_SIZE + DEFAULT_TEST_SIZE

    if not (0.0 < holdout_size < 1.0):
        raise ValueError("validation_size + test_size must be between 0 and 1")

    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=SEED,
        stratify=df[label_column],
    )

    val_fraction = DEFAULT_VALIDATION_SIZE / holdout_size

    val_df, test_df = train_test_split(
        holdout_df,
        test_size=(1.0 - val_fraction),
        random_state=SEED,
        stratify=holdout_df[label_column],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _validate_split_df(
    df: pd.DataFrame,
    name: str,
    text_column: str,
    label_column: str = "label",
):
    ensure_dataframe(df, name=name, required_columns=[text_column, label_column], min_rows=2)
    ensure_non_empty_text_column(df, text_column, name=name)


def _to_hf_dataset(df: pd.DataFrame) -> Dataset:
    dataset = Dataset.from_pandas(df.reset_index(drop=True))

    if "__index_level_0__" in dataset.column_names:
        dataset = dataset.remove_columns(["__index_level_0__"])

    return dataset


def _compute_checkpoint_save_steps(
    *,
    train_examples: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
) -> int:

    forward_steps_per_epoch = math.ceil(train_examples / batch_size)
    optimizer_steps_per_epoch = math.ceil(forward_steps_per_epoch / gradient_accumulation_steps)
    total_optimizer_steps = max(1, optimizer_steps_per_epoch * epochs)

    return max(1, math.ceil(total_optimizer_steps * 0.10))


def _save_reports_to_google_drive(report_source_dir: Path, report_files: list[str]) -> None:
    """
    Copy generated report artifacts to the Google Drive reports directory.
    """
    for report_file in report_files:
        source_path = report_source_dir / report_file
        if not source_path.exists():
            logger.warning("Expected report file not found for Drive sync: %s", source_path)
            continue

        destination_path = GOOGLE_DRIVE_REPORTS_DIR / report_file
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)


def _save_checkpoints_to_google_drive(source_dir: Path, label: str = "") -> None:
    """
    Mirror a local checkpoint directory tree to Google Drive.

    The function copies ``source_dir`` into ``GOOGLE_DRIVE_CHECKPOINTS_DIR``
    preserving the sub-directory name so resuming from Drive works identically
    to resuming from local storage.

    Parameters
    ----------
    source_dir : Path
        Local directory produced by the checkpoint manager or
        ``save_training_checkpoint`` (e.g. ``MODEL_PATH/checkpoint_bundle``).
    label : str
        Optional tag used in log messages for easier debugging.
    """

    if not source_dir.exists():
        logger.warning(
            "Checkpoint Drive sync skipped — source not found%s: %s",
            f" [{label}]" if label else "",
            source_dir,
        )
        return

    GOOGLE_DRIVE_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = GOOGLE_DRIVE_CHECKPOINTS_DIR / source_dir.name

    try:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source_dir, dest)
        logger.info(
            "Checkpoint synced to Google Drive%s: %s → %s",
            f" [{label}]" if label else "",
            source_dir,
            dest,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Google Drive checkpoint sync failed%s for %s: %s",
            f" [{label}]" if label else "",
            source_dir,
            exc,
        )


def train_model(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    text_column: str = "text",
    label_column: str = "label",
    validation_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
):
    """
    Train TruthLens transformer classifier.

    Returns
    -------
    Trainer
    Dataset (test)
    """

    try:

        logger.info("Starting TruthLens transformer training pipeline")

        _validate_split_df(df, "df", text_column, label_column)

        if validation_df is None or test_df is None:
            train_df, val_df, resolved_test_df = _split_train_val_test(
                df,
                label_column=label_column,
            )
        else:
            train_df = df
            val_df = validation_df
            resolved_test_df = test_df

        set_seed(SEED)

        _log_feature_diagnostics(
            train_df[text_column].dropna().tolist(),
            label="training set",
        )

        device = get_device()
        logger.info("Training device: %s", device)

        params = params or {}

        learning_rate = float(params.get("learning_rate", DEFAULT_LEARNING_RATE))
        batch_size = int(params.get("batch_size", DEFAULT_BATCH_SIZE))
        epochs = int(params.get("epochs", DEFAULT_EPOCHS))
        gradient_accumulation_steps = 2

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_dataset = _to_hf_dataset(train_df)
        val_dataset = _to_hf_dataset(val_df)
        test_dataset = _to_hf_dataset(resolved_test_df)

        def combine_text(example):
            title = example["title"] if example["title"] else ""
            text = example["text"] if example["text"] else ""
            example["input_text"] = title + " " + text
            return example

        def build_multitask_labels(example):
            hero_entity = _entity_to_binary(example.get("hero_entities"))
            villain_entity = _entity_to_binary(example.get("villain_entities"))
            victim_entity = _entity_to_binary(example.get("victim_entities"))

            try:
                hero_original = int(example.get("hero", 0) or 0)
            except (TypeError, ValueError):
                hero_original = 0
            try:
                villain_original = int(example.get("villain", 0) or 0)
            except (TypeError, ValueError):
                villain_original = 0
            try:
                victim_original = int(example.get("victim", 0) or 0)
            except (TypeError, ValueError):
                victim_original = 0

            example["hero"] = max(hero_original, hero_entity)
            example["villain"] = max(villain_original, villain_entity)
            example["victim"] = max(victim_original, victim_entity)

            example["emotion_labels"] = [
                example[f"emotion_{i}"] for i in range(20)
            ]

            example["frame_labels"] = [
                example["CO"],
                example["EC"],
                example["HI"],
                example["MO"],
                example["RE"],
            ]

            return example

        def tokenize(example):
            return tokenizer(
                example["input_text"],
                truncation=True,
                max_length=512,
                padding=False,
            )

        def compute_length(example):
            return {"length": len(example["input_ids"])}

        cache_train_dir = TOKENIZED_DATASET_CACHE_DIR / "train"
        cache_val_dir = TOKENIZED_DATASET_CACHE_DIR / "validation"
        cache_test_dir = TOKENIZED_DATASET_CACHE_DIR / "test"

        if cache_train_dir.exists() and cache_val_dir.exists() and cache_test_dir.exists():
            logger.info("Loading tokenized datasets from cache at %s", TOKENIZED_DATASET_CACHE_DIR)
            train_dataset = load_from_disk(str(cache_train_dir))
            val_dataset = load_from_disk(str(cache_val_dir))
            test_dataset = load_from_disk(str(cache_test_dir))
        else:
            logger.info("Tokenized cache not found. Building tokenized datasets with num_proc=%s", FREE_COLAB_NUM_PROC)

            train_dataset = train_dataset.map(combine_text, num_proc=FREE_COLAB_NUM_PROC)
            val_dataset = val_dataset.map(combine_text, num_proc=FREE_COLAB_NUM_PROC)
            test_dataset = test_dataset.map(combine_text, num_proc=FREE_COLAB_NUM_PROC)

            train_dataset = train_dataset.map(build_multitask_labels, num_proc=FREE_COLAB_NUM_PROC)
            val_dataset = val_dataset.map(build_multitask_labels, num_proc=FREE_COLAB_NUM_PROC)
            test_dataset = test_dataset.map(build_multitask_labels, num_proc=FREE_COLAB_NUM_PROC)

            train_dataset = train_dataset.map(tokenize, batched=True, num_proc=FREE_COLAB_NUM_PROC)
            val_dataset = val_dataset.map(tokenize, batched=True, num_proc=FREE_COLAB_NUM_PROC)
            test_dataset = test_dataset.map(tokenize, batched=True, num_proc=FREE_COLAB_NUM_PROC)

            train_dataset = train_dataset.map(compute_length, num_proc=FREE_COLAB_NUM_PROC)
            val_dataset = val_dataset.map(compute_length, num_proc=FREE_COLAB_NUM_PROC)
            test_dataset = test_dataset.map(compute_length, num_proc=FREE_COLAB_NUM_PROC)

            cache_train_dir.parent.mkdir(parents=True, exist_ok=True)
            train_dataset.save_to_disk(str(cache_train_dir))
            val_dataset.save_to_disk(str(cache_val_dir))
            test_dataset.save_to_disk(str(cache_test_dir))
            logger.info("Saved tokenized datasets to cache at %s", TOKENIZED_DATASET_CACHE_DIR)

        train_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "bias_label",
                "ideology_label",
                "propaganda_label",
                "emotion_labels",
                "frame_labels",
                "hero",
                "villain",
                "victim",
            ],
        )

        val_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "bias_label",
                "ideology_label",
                "propaganda_label",
                "emotion_labels",
                "frame_labels",
                "hero",
                "villain",
                "victim",
            ],
        )

        test_dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "bias_label",
                "ideology_label",
                "propaganda_label",
                "emotion_labels",
                "frame_labels",
                "hero",
                "villain",
                "victim",
            ],
        )

        def multitask_data_collator(features):
            batch = base_data_collator(
                [
                    {
                        "input_ids": feature["input_ids"],
                        "attention_mask": feature["attention_mask"],
                    }
                    for feature in features
                ]
            )

            batch["labels"] = {
                "bias": torch.stack([feature["bias_label"].long() for feature in features]),
                "ideology": torch.stack([feature["ideology_label"].long() for feature in features]),
                "propaganda": torch.stack([feature["propaganda_label"].long() for feature in features]),
                "narrative": torch.stack(
                    [
                        torch.tensor(
                            [feature["hero"], feature["villain"], feature["victim"]],
                            dtype=torch.float,
                        )
                        for feature in features
                    ]
                ),
                "narrative_frame": torch.stack(
                    [feature["frame_labels"].float() for feature in features]
                ),
                "emotion": torch.stack(
                    [feature["emotion_labels"].float() for feature in features]
                ),
            }

            return batch

        model = TruthLensMultiTaskModel("roberta-base")

        model.to(device)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        if torch.__version__.startswith("2"):
            try:
                model = torch.compile(model)
            except Exception:
                pass

        training_args = TrainingArguments(
            
            output_dir=str(MODELS_DIR),

            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,

            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,

            num_train_epochs=epochs,

            logging_dir=str(LOGS_DIR),
            logging_steps=200,

            save_strategy="steps",
            save_steps=checkpoint_save_steps,
            save_total_limit=3,

            evaluation_strategy="steps",
            eval_steps=checkpoint_save_steps,

            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            fp16=torch.cuda.is_available(),
            gradient_checkpointing=False,

            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            group_by_length=True,
            length_column_name="length",

            seed=SEED,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=multitask_data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        last_checkpoint = None
        if DEFAULT_RESUME_FROM_CHECKPOINT:
            last_checkpoint = get_last_checkpoint(MODELS_DIR)

        trainer.train(resume_from_checkpoint=last_checkpoint)

        trainer.save_model(str(MODEL_PATH))
        tokenizer.save_pretrained(str(MODEL_PATH))

        # ----------------------------------------------------------
        # Persist ModelMetadata, ModelCard, and register version
        # ----------------------------------------------------------
        try:
            training_metrics = TrainingMetrics()
            if trainer.state and trainer.state.log_history:
                for entry in reversed(trainer.state.log_history):
                    for key in ("eval_accuracy", "eval_f1", "eval_precision", "eval_recall", "eval_roc_auc"):
                        if key in entry and key not in training_metrics.losses:
                            training_metrics.update(key.replace("eval_", ""), float(entry[key]))
            final_metrics: dict = training_metrics.to_dict()

            identity = ModelIdentity(
                model_name=MODEL_NAME,
                version="1.0.0",
                architecture=MODEL_NAME,
            )
            provenance = TrainingProvenance(
                dataset_name="truthlens_dataset",
                dataset_version=None,
                experiment_name=None,
                run_id=None,
                framework="pytorch",
                seed=SEED,
            )
            artifact_paths = ArtifactPaths(
                model_weights=str(MODEL_PATH / "pytorch_model.bin"),
                config_file=str(MODEL_PATH / "config.json"),
                tokenizer_path=str(MODEL_PATH),
                training_logs=str(LOGS_DIR),
                checkpoint_directory=str(MODELS_DIR),
            )
            runtime_env = RuntimeEnvironment(
                python_version=sys.version.split()[0],
                framework_version=torch.__version__,
                cuda_version=torch.version.cuda,
                hardware="cuda" if torch.cuda.is_available() else "cpu",
                device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            )
            model_metadata = ModelMetadata(
                identity=identity,
                provenance=provenance,
                artifacts=artifact_paths,
                runtime=runtime_env,
                metrics=final_metrics or None,
            )
            model_metadata.save_json(MODEL_PATH / "metadata.json")

            model_details = ModelDetails(
                name=MODEL_NAME,
                version="1.0.0",
                architecture=MODEL_NAME,
                description="TruthLens transformer model for misinformation detection.",
                author="TruthLens",
            )
            dataset_info = DatasetInfo(name="truthlens_dataset")
            card_training = CardTrainingConfig(
                framework="pytorch",
                epochs=epochs,
                batch_size=batch_size,
                optimizer="adamw",
                learning_rate=learning_rate,
                hardware="cuda" if torch.cuda.is_available() else "cpu",
                seed=SEED,
            )
            eval_results = EvaluationResults(
                metrics=final_metrics if final_metrics else {"placeholder": 0.0},
                validation_dataset="validation_split",
            )
            ethics = EthicalConsiderations(
                intended_use="Misinformation and propaganda detection in news text.",
                out_of_scope_use="Not intended for legal decisions or high-stakes autonomous actions.",
                limitations="Performance may degrade on domains outside training distribution.",
                bias_risks="Potential bias from training data distribution.",
            )
            card_artifact_paths = ModelCardArtifacts(
                model_weights=str(MODEL_PATH / "pytorch_model.bin"),
                tokenizer=str(MODEL_PATH),
                config_file=str(MODEL_PATH / "config.json"),
                training_logs=str(LOGS_DIR),
            )
            model_card = ModelCard(
                model_details=model_details,
                datasets=[dataset_info],
                training=card_training,
                evaluation=eval_results,
                ethics=ethics,
                artifacts=card_artifact_paths,
            )
            model_card.save_json(MODEL_PATH / "model_card.json")
            model_card.save_markdown(MODEL_PATH / "model_card.md")

            _save_reports_to_google_drive(
                report_source_dir=MODEL_PATH,
                report_files=["metadata.json", "model_card.json", "model_card.md"],
            )

            version_registry = ModelVersionRegistry(MODELS_DIR)
            version_info = ModelVersionInfo(
                model_name=MODEL_NAME,
                version="1.0.0",
                description="TruthLens transformer classifier",
                metrics=final_metrics or None,
                artifact_path=str(MODEL_PATH),
            )
            version_registry.register_version(version_info)

            logger.info("Model metadata, card, and version registration complete")

        except Exception as _meta_exc:
            logger.warning("Metadata/card/versioning step failed (non-fatal): %s", _meta_exc)

        export_formats = params.get("export_formats", [])
        if isinstance(export_formats, str):
            export_formats = [export_formats]

        if export_formats:
            _export_trained_artifacts(
                model=trainer.model,
                train_dataset=train_dataset,
                export_formats=[str(x) for x in export_formats],
                export_root=MODEL_PATH / "exports",
            )

        example_input = _build_export_example_input(train_dataset)
        checkpoint_bundle_dir = MODEL_PATH / "checkpoint_bundle"
        checkpoint_manager = CheckpointManager(checkpoint_bundle_dir)
        artifact_manager = ArtifactManager(checkpoint_bundle_dir)

        checkpoint_metadata = {
            "model_name": MODEL_NAME,
            "num_labels": int(TruthLensMultiTaskModel.NUM_BIAS),
            "task_label_dims": {
                "bias": int(TruthLensMultiTaskModel.NUM_BIAS),
                "ideology": int(TruthLensMultiTaskModel.NUM_IDEOLOGY),
                "propaganda": int(TruthLensMultiTaskModel.NUM_PROPAGANDA),
                "narrative": int(TruthLensMultiTaskModel.NUM_NARRATIVE),
                "frame": int(TruthLensMultiTaskModel.NUM_NARRATIVE_FRAMES),
                "emotion": int(TruthLensMultiTaskModel.NUM_EMOTIONS),
            },
            "export_formats": export_formats,
            "epoch": epochs,
        }

        checkpoint_manager.save_checkpoint(
            step=int(getattr(trainer.state, "global_step", 0)),
            model_state_dict=trainer.model.state_dict(),
            metadata=checkpoint_metadata,
        )
        artifact_manager.save_metadata(checkpoint_metadata)

        wrapped_export_model = _HFExportWrapper(trainer.model).cpu().eval()
        save_training_checkpoint(
            trainer.model,
            checkpoint_dir=checkpoint_bundle_dir / "training",
            optimizer=getattr(trainer, "optimizer", None),
            scheduler=getattr(trainer, "lr_scheduler", None),
            epoch=epochs,
            step=int(getattr(trainer.state, "global_step", 0)),
            metadata=checkpoint_metadata,
            export_formats=[str(x) for x in export_formats],
            export_model=wrapped_export_model,
            export_example_input=example_input,
        )

        if "torchscript" in {fmt.strip().lower() for fmt in export_formats} and example_input is not None:
            artifact_manager.export_torchscript(
                model=wrapped_export_model,
                example_input=example_input.detach().cpu(),
            )
        if "onnx" in {fmt.strip().lower() for fmt in export_formats} and example_input is not None:
            artifact_manager.export_onnx(
                model=wrapped_export_model,
                example_input=example_input.detach().cpu(),
            )
        if "quantized" in {fmt.strip().lower() for fmt in export_formats}:
            artifact_manager.export_quantized_model(model=wrapped_export_model)

        _save_checkpoints_to_google_drive(checkpoint_bundle_dir, label="final_bundle")
        _save_checkpoints_to_google_drive(
            checkpoint_bundle_dir / "training", label="training_checkpoint"
        )

        resolved_test_df.to_csv(TEST_SET_PATH, index=False)

        logger.info("Training completed successfully")

        return trainer, test_dataset

    except Exception:
        logger.exception("Training pipeline failed")
        raise


def _log_feature_diagnostics(texts: list[str], label: str = "") -> None:
    """
    Generate feature statistics and schema diagnostics for a text corpus.

    Uses DatasetFeatureGenerator to extract a feature matrix, then
    FeatureStatistics to report the dataset summary and detect any
    constant (zero-variance) features. FeatureSchemaValidator confirms
    that all extracted feature vectors conform to the inferred schema.

    This is a non-blocking diagnostic step — any failure is logged as a
    warning and does not interrupt the training pipeline.

    Parameters
    ----------
    texts : list[str]
        Raw article texts (e.g. the training split).
    label : str
        Descriptive label used in log messages (e.g. "training set").
    """
    try:
        from src.features.pipelines.feature_pipeline import FeaturePipeline
        from src.features.pipelines.batch_feature_pipeline import BatchFeaturePipeline

        tag = f" [{label}]" if label else ""
        logger.info("Running feature diagnostics%s | samples=%d", tag, len(texts))

        batch_pipeline = BatchFeaturePipeline(pipeline=FeaturePipeline())
        generator = DatasetFeatureGenerator(pipeline=batch_pipeline)
        matrix, feature_names = generator.generate(texts)

        stats = FeatureStatistics()

        contexts = generator._build_contexts(texts)
        feature_dicts = batch_pipeline._sequential_extract(contexts)

        summary = stats.dataset_summary(feature_dicts)
        logger.info(
            "Feature dataset summary%s | samples=%d features=%d "
            "mean_variance=%.6f",
            tag,
            int(summary["num_samples"]),
            int(summary["num_features"]),
            summary["mean_variance"],
        )

        constant = stats.detect_constant_features(feature_dicts)
        if constant:
            logger.warning(
                "Detected %d constant (zero-variance) feature(s)%s: %s",
                len(constant),
                tag,
                constant[:10],
            )

        validator = FeatureSchemaValidator(
            expected_features=feature_names,
            strict=False,
            allow_missing=True,
            allow_extra=True,
        )
        validator.validate_batch(feature_dicts[:min(5, len(feature_dicts))])
        schema_info = validator.schema_summary()
        logger.info(
            "Feature schema validated%s | schema_features=%d",
            tag,
            schema_info["num_features"],
        )

    except Exception as _diag_exc:
        logger.warning("Feature diagnostics skipped (non-fatal): %s", _diag_exc)


def _build_export_example_input(train_dataset: Dataset) -> torch.Tensor | None:
    try:
        sample = train_dataset[0]
        input_ids = sample.get("input_ids")
        if isinstance(input_ids, torch.Tensor):
            return input_ids.unsqueeze(0)
        if isinstance(input_ids, list):
            return torch.tensor([input_ids], dtype=torch.long)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build export example input: %s", exc)
    return None


def _export_trained_artifacts(
    *,
    model: nn.Module,
    train_dataset: Dataset,
    export_formats: list[str],
    export_root: Path,
) -> None:
    create_folder(export_root)
    wrapped_model = _HFExportWrapper(model).cpu().eval()
    example_input = _build_export_example_input(train_dataset)

    if example_input is None:
        logger.warning("Skipping model exports: no example input available.")
        return

    requested = {fmt.strip().lower() for fmt in export_formats}

    if "torchscript" in requested:
        try:
            TorchScriptExporter(
                TorchScriptExportConfig(device="cpu", verify_export=False)
            ).export(
                model=wrapped_model,
                example_input=example_input.detach().cpu(),
                output_path=export_root / "model.ts.pt",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("TorchScript export skipped due to error: %s", exc)

    if "onnx" in requested:
        try:
            ONNXExporter(
                ONNXExportConfig(device="cpu", verify_export=False)
            ).export(
                model=wrapped_model,
                example_input=example_input.detach().cpu(),
                output_path=export_root / "model.onnx",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ONNX export skipped due to error: %s", exc)

    if "quantized" in requested:
        try:
            quantized_model = QuantizationEngine(
                QuantizationConfig(
                    method="dynamic",
                    device="cpu",
                    backend=_quant_backend(),
                )
            ).apply(wrapped_model)
            torch.save(
                quantized_model.state_dict(),
                export_root / "model.quantized.pt",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Quantized export skipped due to error: %s", exc)
