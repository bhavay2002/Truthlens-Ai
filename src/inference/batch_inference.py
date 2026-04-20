"""
File Name: batch_inference.py
Module: Batch Inference Engine
Description:
    Executes large-scale inference over datasets containing thousands or
    millions of articles. The module orchestrates loading input datasets,
    running the TruthLens prediction pipeline, generating reports, and
    exporting structured outputs.

    Typical usage scenarios include:
        • research experiments
        • evaluation pipelines
        • dataset labeling
        • large-scale monitoring systems
        • offline analytics

    The engine processes data in batches to ensure memory efficiency and
    GPU utilization.

Dependencies:
    logging
    typing
    dataclasses
    pathlib
    argparse
    pandas
    tqdm
    json

Inputs:
    CSV dataset containing article text and optional metadata.

Outputs:
    JSON / CSV files containing predictions and generated reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm
import torch

from src.inference.model_loader import ModelLoader
from src.inference.feature_preparer import (
    FeaturePreparer,
    FeaturePreparationConfig,
)
from src.inference.prediction_pipeline import (
    PredictionPipeline,
    PredictionPipelineConfig,
)
from src.inference.report_generator import ReportGenerator
from src.inference.result_formatter import ResultFormatter
from src.analysis.integration_runner import AnalysisIntegrationRunner
from src.graph.graph_pipeline import GraphPipeline

logger = logging.getLogger(__name__)

_worker_runner: Optional[AnalysisIntegrationRunner] = None
_worker_graph: Optional[GraphPipeline] = None


@dataclass
class BatchInferenceConfig:
    """
    Configuration for batch inference.
    """
    dataset_path: str
    text_column: str = "text"
    output_path: str = "batch_predictions.json"
    batch_size: int = 32
    models_dir: str = "models"
    num_workers: int = 0


def _init_worker() -> None:
    global _worker_runner, _worker_graph
    _worker_runner = AnalysisIntegrationRunner()
    _worker_graph = GraphPipeline()


def _analyze_text_worker(text: str) -> Dict[str, Any]:
    if _worker_runner is None:
        raise RuntimeError("Worker runner not initialized")
    return _worker_runner.analyze_text(text)


def _graph_run_worker(text: str) -> Dict[str, Any]:
    if _worker_graph is None:
        raise RuntimeError("Worker graph not initialized")
    return _worker_graph.run(text)


class BatchInferenceEngine:
    """
    Runs large-scale inference across datasets.
    """

    def __init__(
        self,
        config: BatchInferenceConfig,
        feature_preparer: Optional[FeaturePreparer] = None,
    ) -> None:

        self.config = config

        self.model_loader = ModelLoader(config.models_dir)
        self.artifacts = self.model_loader.load_all()

        if feature_preparer is not None:
            self.feature_preparer = feature_preparer
        else:
            schema = self.artifacts.feature_schema
            if isinstance(schema, dict):
                feature_schema = [str(k) for k in schema.keys()]
            elif isinstance(schema, list):
                feature_schema = [str(k) for k in schema]
            else:
                feature_schema = ["text_length"]

            prep_config = FeaturePreparationConfig(
                feature_schema=feature_schema,
                return_tensor=True,
            )
            self.feature_preparer = FeaturePreparer(
                prep_config,
                scaler=self.artifacts.feature_scaler,
                selector=self.artifacts.feature_selector,
            )

        self.prediction_pipeline = PredictionPipeline(
            config=PredictionPipelineConfig(
                device=str(self.model_loader.device),
                return_probabilities=False,
            ),
            bias_model=self.artifacts.bias_model,
            ideology_model=self.artifacts.ideology_model,
            propaganda_model=None,
            emotion_model=self.artifacts.emotion_model,
        )

        if torch.cuda.is_available():
            for model in [
                self.artifacts.bias_model,
                self.artifacts.ideology_model,
                self.artifacts.emotion_model,
            ]:
                if model is not None:
                    model.half()

        compile_models = getattr(self.prediction_pipeline, "compile_models", None)
        if callable(compile_models):
            compile_models()

        self.report_generator = ReportGenerator()
        self.formatter = ResultFormatter()
        self.analysis_runner = AnalysisIntegrationRunner()
        self.graph_pipeline = GraphPipeline()

        logger.info("BatchInferenceEngine initialized")

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load dataset from disk.
        """

        path = Path(self.config.dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)

        if self.config.text_column not in df.columns:
            raise ValueError(
                f"Text column '{self.config.text_column}' not found in dataset"
            )

        logger.info("Loaded dataset with %d rows", len(df))

        return df

    def _process_batch(
        self,
        texts: List[str],
        metadata_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run inference for a batch of articles.
        """

        if self.feature_preparer is None:
            raise RuntimeError("FeaturePreparer is required for batch inference")

        clean_texts: List[str] = []
        for t in texts:
            if t is None:
                clean_texts.append("")
            else:
                clean_texts.append(str(t))
        features_list = [{"text": t, "text_length": len(t)} for t in clean_texts]
        prepared = self.feature_preparer.prepare_batch(features_list)

        prepared = torch.as_tensor(prepared, dtype=torch.float32)

        # Pin only when CUDA transfer can benefit from it.
        if prepared.device.type == "cpu" and torch.cuda.is_available():
            prepared = prepared.pin_memory()

        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            predictions = self.prediction_pipeline.predict(prepared)

        results: List[Dict[str, Any]] = []

        bias_values = predictions.get("bias")
        emotion_values = predictions.get("emotion")
        credibility_values = predictions.get("credibility_score")
        ideology_values = predictions.get("ideology")
        propaganda_values = predictions.get("propaganda_probability")

        if self.config.num_workers > 0:
            ctx = mp.get_context("spawn")
            with ctx.Pool(self.config.num_workers, initializer=_init_worker) as pool:
                analysis_results = pool.map(_analyze_text_worker, texts)
                graph_results = pool.map(_graph_run_worker, texts)
        else:
            analysis_results = [self.analysis_runner.analyze_text(text) for text in texts]
            graph_results = [self.graph_pipeline.run(text) for text in texts]

        for i, text in enumerate(texts):
            metadata = metadata_list[i]
            analysis_modules = analysis_results[i]
            graph_outputs = graph_results[i]

            def _value_at(value: Any) -> Any:
                if isinstance(value, list):
                    return value[i]
                if torch.is_tensor(value):
                    return value[i].item() if value.ndim > 0 else value.item()
                return value

            report = self.report_generator.generate_report(
                article_text=text,
                title=metadata.get("title"),
                source=metadata.get("source"),
                bias_analysis={"bias": _value_at(bias_values)},
                emotion_analysis={"emotion": _value_at(emotion_values)},
                narrative_structure=analysis_modules.get("narrative_propagation", {}),
                entity_graph=graph_outputs,
                credibility_score=_value_at(credibility_values),
            )

            api_output = self.formatter.format_api_response(
                {
                    "bias": _value_at(bias_values),
                    "ideology": _value_at(ideology_values),
                    "propaganda_probability": _value_at(propaganda_values),
                    "emotion": _value_at(emotion_values),
                    "credibility_score": _value_at(credibility_values),
                    "credibility_explanation": _value_at(
                        predictions.get("credibility_explanation")
                    ),
                }
            )

            results.append(
                {
                    "prediction": api_output,
                    "report": report,
                    "analysis_modules": analysis_modules,
                }
            )

        return results

    def run(self) -> List[Dict[str, Any]]:
        """
        Execute batch inference.
        """

        df = self._load_dataset()

        results: List[Dict[str, Any]] = []

        batch_size = self.config.batch_size

        for start in tqdm(range(0, len(df), batch_size), desc="Batch inference"):
            batch_df = df.iloc[start:start + batch_size]
            texts = [str(t) if pd.notna(t) else "" for t in batch_df[self.config.text_column].tolist()]
            texts = [t for t in texts if t.strip()]

            # Metadata columns are optional by module contract.
            has_title = "title" in batch_df.columns
            has_source = "source" in batch_df.columns
            metadata_list = []
            for _, row in batch_df.iterrows():
                metadata_list.append(
                    {
                        "title": row["title"] if has_title else None,
                        "source": row["source"] if has_source else None,
                    }
                )

            try:
                batch_results = self._process_batch(texts, metadata_list)
                results.extend(batch_results)
            except Exception as exc:
                logger.exception("Batch failed")
                results.extend([{"error": str(exc)}] * len(texts))

        logger.info("Batch inference completed")

        return results

    def export_onnx(self, path: str) -> None:
        if self.artifacts.bias_model is None:
            raise RuntimeError("Bias model unavailable for ONNX export")

        dummy = torch.randn(1, 10, device=self.model_loader.device)

        torch.onnx.export(
            self.artifacts.bias_model,
            dummy,
            path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}},
            opset_version=17,
        )

        logger.info("ONNX export completed: %s", path)

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Save results to disk.
        """

        output_path = Path(self.config.output_path)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)

        logger.info("Saved results to %s", output_path)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """

    parser = argparse.ArgumentParser(description="Run TruthLens batch inference")

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset CSV",
    )

    parser.add_argument(
        "--text-column",
        default="text",
        help="Column containing article text",
    )

    parser.add_argument(
        "--output",
        default="batch_predictions.json",
        help="Output predictions file",
    )

    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing trained models",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entrypoint.
    """

    args = parse_args()

    config = BatchInferenceConfig(
        dataset_path=args.dataset,
        text_column=args.text_column,
        output_path=args.output,
        batch_size=args.batch_size,
        models_dir=args.models_dir,
    )

    engine = BatchInferenceEngine(config)

    results = engine.run()

    engine.save_results(results)


if __name__ == "__main__":
    main()
