"""End-to-end test runner for TruthLens AI.

Loads all 6 test CSVs (bias, ideology, propaganda, narrative_frame,
narrative, emotion), wires up the full TruthLensPipeline
(Inference → Analysis → Aggregation → Evaluation → Explainability),
and prints a structured summary of every stage.

Usage:
    uv run python run_test_pipeline.py
    uv run python run_test_pipeline.py --samples 5
    uv run python run_test_pipeline.py --samples 5 --explainability
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src.config.config_loader import load_config
from src.utils.logging_utils import configure_logging
from src.utils.seed_utils import set_seed
from src.pipelines.truthlens_pipeline import TruthLensPipeline
from src.evaluation.evaluation_pipeline import run_evaluation_pipeline

CONFIG_PATH = Path("config/config.yaml")
DATA_DIR = Path("data/test")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_test_pipeline")


# =========================================================
# DATASET LOADERS
# =========================================================

def load_bias(n: int) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(DATA_DIR / "bias.csv").dropna(subset=["text", "bias_label"]).head(n)
    return df["text"].tolist(), df["bias_label"].astype(int).values


def load_ideology(n: int) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(DATA_DIR / "ideology.csv").dropna(subset=["text", "ideology_label"]).head(n)
    return df["text"].tolist(), df["ideology_label"].astype(int).values


def load_propaganda(n: int) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(DATA_DIR / "propaganda.csv").dropna(subset=["text", "propaganda_label"]).head(n)
    return df["text"].tolist(), df["propaganda_label"].astype(int).values


def load_narrative_frame(n: int) -> Tuple[List[str], np.ndarray]:
    """frame.csv: columns CO, EC, HI, MO, RE → multilabel (5 cols)."""
    df = pd.read_csv(DATA_DIR / "frame.csv").dropna(subset=["text"]).head(n)
    label_cols = ["CO", "EC", "HI", "MO", "RE"]
    for c in label_cols:
        if c not in df.columns:
            df[c] = 0
    labels = df[label_cols].fillna(0).astype(int).values
    return df["text"].tolist(), labels


def load_narrative(n: int) -> Tuple[List[str], np.ndarray]:
    """narrative.csv: columns hero, villain, victim → multilabel (3 cols)."""
    df = pd.read_csv(DATA_DIR / "narrative.csv").dropna(subset=["text"]).head(n)
    label_cols = ["hero", "villain", "victim"]
    for c in label_cols:
        if c not in df.columns:
            df[c] = 0
    labels = df[label_cols].fillna(0).astype(int).values
    return df["text"].tolist(), labels


def load_emotion(n: int) -> Tuple[List[str], np.ndarray]:
    """emotion.csv: columns emotion_0..emotion_10 → multilabel (11 cols)."""
    df = pd.read_csv(DATA_DIR / "emotion.csv").dropna(subset=["text"]).head(n)
    label_cols = [f"emotion_{i}" for i in range(11)]
    for c in label_cols:
        if c not in df.columns:
            df[c] = 0
    labels = df[label_cols].fillna(0).astype(int).values
    return df["text"].tolist(), labels


DATASET_LOADERS = {
    "bias": load_bias,
    "ideology": load_ideology,
    "propaganda": load_propaganda,
    "narrative_frame": load_narrative_frame,
    "narrative": load_narrative,
    "emotion": load_emotion,
}


# =========================================================
# LABEL FORMATTERS
# =========================================================

def labels_to_eval_fmt(task: str, labels: np.ndarray) -> Any:
    """Convert raw numpy label arrays into the format expected by
    run_evaluation_pipeline (lists of int for multiclass / lists of
    binary lists for multilabel)."""
    if labels.ndim == 1:
        return labels.tolist()
    return labels.tolist()


# =========================================================
# REPORTING HELPERS
# =========================================================

_SEP = "─" * 70


def _hdr(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _print_article(idx: int, text: str, result: Dict[str, Any]) -> None:
    print(f"\n  [Article {idx + 1}]  {text[:80]}{'...' if len(text) > 80 else ''}")
    errors = result.get("errors") or {}
    if errors:
        for stage, msg in errors.items():
            print(f"    ⚠  {stage}: {msg[:120]}")

    scores = result.get("scores") or {}
    if scores:
        score_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items()) if isinstance(v, float))
        print(f"    scores      : {score_str or '(none)'}")

    preds = result.get("predictions") or {}
    if preds:
        def _fmt_val(v: Any) -> str:
            if isinstance(v, (list, np.ndarray)):
                return str(v)
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        pred_str = "  ".join(f"{k}={_fmt_val(v)}" for k, v in sorted(preds.items()))
        print(f"    predictions : {pred_str[:120]}")

    stages = (result.get("metadata") or {}).get("stages") or {}
    if stages:
        timing = "  ".join(f"{k}={v*1000:.0f}ms" for k, v in stages.items())
        print(f"    timing      : {timing}")

    agg = result.get("aggregation") or {}
    credibility = (agg.get("scores") or agg.get("raw_scores") or {}).get("credibility_score")
    if credibility is not None:
        print(f"    credibility : {credibility:.3f}")

    expl = result.get("explainability")
    if expl:
        top_tokens = (expl.get("tokens") or [])[:5]
        if top_tokens:
            print(f"    top tokens  : {top_tokens}")


def _print_eval_report(task: str, task_report: Dict[str, Any]) -> None:
    metrics = task_report.get("metrics") or {}
    if not metrics:
        print(f"    {task}: no metrics computed (model not yet trained)")
        return
    parts = []
    for k in ("accuracy", "f1", "precision", "recall", "roc_auc"):
        v = metrics.get(k)
        if v is not None:
            parts.append(f"{k}={v:.3f}")
    print(f"    {task}: {' | '.join(parts) if parts else str(metrics)[:120]}")


# =========================================================
# MAIN
# =========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TruthLens end-to-end test pipeline")
    p.add_argument("--samples", type=int, default=3,
                   help="Number of samples to pull from each test dataset (default 3)")
    p.add_argument("--explainability", action="store_true",
                   help="Enable explainability stage (requires trained model)")
    p.add_argument("--no-parallel", action="store_true",
                   help="Disable parallel analysis/graph stages")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    config = load_config(CONFIG_PATH)
    set_seed(config.project.seed)

    logger.info("=" * 60)
    logger.info("TruthLens  End-to-End Test Pipeline")
    logger.info("datasets  : %s", list(DATASET_LOADERS))
    logger.info("samples   : %d per dataset", args.samples)
    logger.info("explainability: %s", args.enable_explainability if hasattr(args, 'enable_explainability') else args.explainability)
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # 1. LOAD DATASETS
    # ----------------------------------------------------------
    _hdr("1 / 6  LOADING DATASETS")
    all_texts: Dict[str, List[str]] = {}
    all_labels: Dict[str, Any] = {}
    missing: List[str] = []

    for task, loader in DATASET_LOADERS.items():
        csv_path = DATA_DIR / f"{task if task != 'narrative_frame' else 'frame'}.csv"
        if not csv_path.exists():
            logger.warning("  ✗  %s — file not found: %s", task, csv_path)
            missing.append(task)
            continue
        try:
            texts, labels = loader(args.samples)
            all_texts[task] = texts
            all_labels[task] = labels_to_eval_fmt(task, labels)
            shape = labels.shape if hasattr(labels, "shape") else len(labels)
            logger.info("  ✓  %-16s %d samples  labels shape=%s", task, len(texts), shape)
        except Exception as exc:
            logger.error("  ✗  %s — load failed: %s", task, exc, exc_info=True)
            missing.append(task)

    if missing:
        logger.warning("Skipped datasets: %s", missing)
    if not all_texts:
        logger.error("No datasets loaded — aborting.")
        sys.exit(1)

    # ----------------------------------------------------------
    # 2. BUILD PIPELINE
    # ----------------------------------------------------------
    _hdr("2 / 6  BUILDING PIPELINE")

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder)
    logger.info("Tokenizer loaded: %s", config.model.encoder)

    predictor: Optional[Any] = None
    ckpt_path = Path("saved_models/checkpoint.pt")
    if ckpt_path.is_file():
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            from src.models.inference.predictor import Predictor
            model_obj = state.get("model") if isinstance(state, dict) else None
            if isinstance(model_obj, torch.nn.Module):
                predictor = Predictor(model=model_obj)
                logger.info("✓  Predictor loaded from %s", ckpt_path)
            else:
                logger.warning("Checkpoint found but contains no nn.Module — prediction disabled")
        except Exception as exc:
            logger.warning("Checkpoint load failed (%s) — prediction disabled", exc)
    else:
        logger.warning(
            "No checkpoint at %s — analysis/aggregation/graph will run; "
            "prediction/evaluation/explainability require a trained model. "
            "Run `uv run python main.py --mode train` first.", ckpt_path
        )

    pipeline = TruthLensPipeline(
        predictor=predictor,
        tokenizer=tokenizer,
        model_version=config.model.encoder,
        enable_explainability=args.explainability,
        enable_evaluation=predictor is not None,
        parallel_stages=not args.no_parallel,
    )
    logger.info("Pipeline ready  (explainability=%s  evaluation=%s  parallel=%s)",
                args.explainability, predictor is not None, not args.no_parallel)

    # ----------------------------------------------------------
    # 3. RUN INFERENCE + ANALYSIS + AGGREGATION per dataset
    # ----------------------------------------------------------
    _hdr("3 / 6  INFERENCE · ANALYSIS · AGGREGATION")

    dataset_results: Dict[str, Dict[str, Any]] = {}
    total_articles = 0
    t_pipeline_start = time.time()

    for task, texts in all_texts.items():
        logger.info("\n  ── Dataset: %s (%d texts) ──", task, len(texts))
        labels_for_eval = {task: all_labels[task]} if predictor is not None else None

        try:
            batch = pipeline.analyze_batch(texts, labels=labels_for_eval)
        except Exception as exc:
            logger.error("  analyze_batch failed for %s: %s", task, exc, exc_info=True)
            continue

        dataset_results[task] = batch
        total_articles += len(texts)

        for i, (text, result) in enumerate(zip(texts, batch["articles"])):
            _print_article(i, text, result)

        meta = batch.get("batch_metadata", {})
        logger.info("  ✓  %s complete | %d articles | %.2fs total",
                    task, meta.get("n_articles", 0), meta.get("total_time", 0))

    pipeline_elapsed = time.time() - t_pipeline_start
    logger.info("\n  Pipeline total: %d articles in %.2fs (%.0f ms/article)",
                total_articles, pipeline_elapsed,
                pipeline_elapsed / max(1, total_articles) * 1000)

    # ----------------------------------------------------------
    # 4. EVALUATION  (dataset-level, requires trained model)
    # ----------------------------------------------------------
    _hdr("4 / 6  EVALUATION")

    if predictor is None:
        logger.warning(
            "Evaluation skipped — no trained model loaded.\n"
            "  Train first:  uv run python main.py --mode train\n"
            "  Then re-run:  uv run python run_test_pipeline.py"
        )
    else:
        # Merge all texts + labels across every loaded task so the
        # evaluation pipeline sees the full multi-task label matrix.
        merged_texts: List[str] = []
        merged_labels: Dict[str, Any] = {}

        for task, texts in all_texts.items():
            start_idx = len(merged_texts)
            for _ in texts:
                merged_texts.append("")
            merged_labels[task] = all_labels[task]

        # Use a representative common text list (first available dataset)
        common_texts = next(iter(all_texts.values()))

        try:
            t_eval = time.time()
            eval_report = run_evaluation_pipeline(
                model=getattr(predictor, "model", None),
                tokenizer=tokenizer,
                texts=common_texts,
                labels={t: all_labels[t] for t in all_texts
                        if t in all_labels and len(all_labels[t]) == len(common_texts)},
                enable_calibration=True,
                enable_threshold_opt=True,
                enable_uncertainty=True,
                enable_error_analysis=True,
                enable_correlation=True,
            )
            logger.info("Evaluation complete in %.2fs", time.time() - t_eval)

            tasks_report = eval_report.get("tasks") or {}
            for task_name, task_report in tasks_report.items():
                _print_eval_report(task_name, task_report)
        except Exception as exc:
            logger.error("Evaluation pipeline failed: %s", exc, exc_info=True)

    # ----------------------------------------------------------
    # 5. EXPLAINABILITY SUMMARY
    # ----------------------------------------------------------
    _hdr("5 / 6  EXPLAINABILITY")

    expl_count = 0
    for task, batch in dataset_results.items():
        for article in batch.get("articles", []):
            expl = article.get("explainability")
            if expl:
                expl_count += 1
                method_scores = expl.get("method_scores") or {}
                logger.info("  %s | methods=%s", task, list(method_scores))

    if expl_count == 0:
        reason = "no trained model" if predictor is None else "explainability flag not set"
        logger.info("  No explanations generated (%s).", reason)
        if not args.explainability:
            logger.info("  Re-run with --explainability to enable SHAP/LIME/attention explanations.")

    # ----------------------------------------------------------
    # 6. AGGREGATION SCORE SUMMARY
    # ----------------------------------------------------------
    _hdr("6 / 6  AGGREGATION SCORE SUMMARY")

    score_rows: List[Dict[str, Any]] = []
    for task, batch in dataset_results.items():
        for i, article in enumerate(batch.get("articles", [])):
            agg = article.get("aggregation") or {}
            raw_scores = agg.get("scores") or agg.get("raw_scores") or {}
            scores_flat = article.get("scores") or {}
            merged = {**raw_scores, **scores_flat}
            credibility = merged.get("credibility_score") or merged.get("credibility")
            errors = list((article.get("errors") or {}).keys())
            score_rows.append({
                "dataset": task,
                "sample": i + 1,
                "credibility": round(float(credibility), 3) if credibility is not None else None,
                "n_errors": len(errors),
                "error_stages": errors if errors else [],
            })

    if score_rows:
        df = pd.DataFrame(score_rows)
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(df.to_string(index=False))

        by_dataset = df.groupby("dataset")["credibility"].agg(["mean", "min", "max"]).round(3)
        print("\n  Credibility by dataset:")
        print(by_dataset.to_string())

        total_errors = df["n_errors"].sum()
        print(f"\n  Total stage errors across all articles: {total_errors}")
        if total_errors == 0:
            print("  ✅ All pipeline stages completed without errors.")
        else:
            err_df = df[df["n_errors"] > 0][["dataset", "sample", "error_stages"]]
            print(err_df.to_string(index=False))

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------
    _hdr("PIPELINE COMPLETE")
    logger.info("Datasets processed : %d / %d", len(dataset_results), len(DATASET_LOADERS))
    logger.info("Articles analysed  : %d", total_articles)
    logger.info("Wall time          : %.2fs", pipeline_elapsed)
    logger.info("Model checkpoint   : %s", "loaded" if predictor else "not found (train first)")
    logger.info("Evaluation         : %s", "ran" if predictor else "skipped")
    logger.info("Explainability     : %s", "ran" if (args.explainability and predictor) else "skipped")

    if predictor is None:
        logger.info("\n  Next step → train the model:")
        logger.info("    uv run python main.py --mode train")
        logger.info("  Then re-run with full evaluation + explainability:")
        logger.info("    uv run python run_test_pipeline.py --explainability")


if __name__ == "__main__":
    main()
