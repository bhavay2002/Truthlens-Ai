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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm

from src.inference.model_loader import ModelLoader
from src.inference.feature_preparer import FeaturePreparer
from src.inference.prediction_pipeline import PredictionPipeline
from src.inference.report_generator import ReportGenerator
from src.inference.result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


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

        self.feature_preparer = feature_preparer

        self.prediction_pipeline = PredictionPipeline(
            config=None,
            bias_model=self.artifacts.bias_model,
            ideology_model=self.artifacts.ideology_model,
            propaganda_model=None,
            emotion_model=self.artifacts.emotion_model,
        )

        self.report_generator = ReportGenerator()
        self.formatter = ResultFormatter()

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

    def _process_article(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run inference for a single article.
        """

        if self.feature_preparer is None:
            raise RuntimeError("FeaturePreparer is required for batch inference")

        features_dict = {"text_length": len(text)}

        prepared_features = self.feature_preparer.prepare_single(features_dict)

        prediction = self.prediction_pipeline.predict(prepared_features)

        report = self.report_generator.generate_report(
            article_text=text,
            title=metadata.get("title") if metadata else None,
            source=metadata.get("source") if metadata else None,
            bias_analysis={"bias": prediction.get("bias")},
            emotion_analysis={"emotion": prediction.get("emotion")},
            narrative_structure={},
            entity_graph={},
            credibility_score=prediction.get("credibility_score"),
        )

        api_output = self.formatter.format_api_response(prediction)

        result = {
            "prediction": api_output,
            "report": report,
        }

        return result

    def run(self) -> List[Dict[str, Any]]:
        """
        Execute batch inference.
        """

        df = self._load_dataset()

        results: List[Dict[str, Any]] = []

        iterator = tqdm(df.iterrows(), total=len(df), desc="Running inference")

        for _, row in iterator:

            text = row[self.config.text_column]

            metadata = {
                "title": row.get("title"),
                "source": row.get("source"),
            }

            try:
                result = self._process_article(text, metadata)
                results.append(result)

            except Exception as exc:
                logger.exception("Inference failed for article")
                results.append({"error": str(exc)})

        logger.info("Batch inference completed")

        return results

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