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
import numpy as np

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


# =========================================================
# HELPER
# =========================================================

def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class BatchInferenceConfig:
    dataset_path: str
    text_column: str = "text"
    output_path: str = "batch_predictions.json"
    batch_size: int = 32
    models_dir: str = "models"
    num_workers: int = 0


# =========================================================
# ENGINE
# =========================================================

class BatchInferenceEngine:

    def __init__(self, config: BatchInferenceConfig):

        self.config = config
        self.model_loader = ModelLoader(config.models_dir)
        self.artifacts = self.model_loader.load_all()

        # ---------------- FEATURE PREPARER ----------------
        schema = self.artifacts.feature_schema
        feature_schema = (
            list(schema.keys()) if isinstance(schema, dict)
            else schema if isinstance(schema, list)
            else ["text_length"]
        )

        self.feature_preparer = FeaturePreparer(
            FeaturePreparationConfig(
                feature_schema=feature_schema,
                return_tensor=True,
            ),
            scaler=self.artifacts.feature_scaler,
            selector=self.artifacts.feature_selector,
        )

        # ---------------- MODEL PIPELINE ----------------
        self.prediction_pipeline = PredictionPipeline(
            config=PredictionPipelineConfig(
                device=str(self.model_loader.device),
                return_probabilities=True,  # 🔥 IMPORTANT
                return_logits=True,         # 🔥 NEW
            ),
            bias_model=self.artifacts.bias_model,
            ideology_model=self.artifacts.ideology_model,
            emotion_model=self.artifacts.emotion_model,
        )

        self.report_generator = ReportGenerator()
        self.formatter = ResultFormatter()
        self.analysis_runner = AnalysisIntegrationRunner()
        self.graph_pipeline = GraphPipeline()

    # =====================================================
    # DATA
    # =====================================================

    def _load_dataset(self):
        df = pd.read_csv(self.config.dataset_path)
        if self.config.text_column not in df.columns:
            raise ValueError("Text column missing")
        return df

    # =====================================================
    # BATCH PROCESSING (UPDATED 🔥)
    # =====================================================

    def _process_batch(self, texts: List[str]):

        features = [{"text": t, "text_length": len(t)} for t in texts]
        prepared = self.feature_preparer.prepare_batch(features)
        prepared = torch.tensor(prepared, dtype=torch.float32)

        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available(),
        ):
            output = self.prediction_pipeline.predict(prepared)

        results = []

        for i, text in enumerate(texts):

            # ---------------- EXTRACT PER SAMPLE ----------------
            logits = {
                k: _to_numpy(v[i]) if v is not None else None
                for k, v in output.get("logits", {}).items()
            }

            probs = {
                k: _to_numpy(v[i]) if v is not None else None
                for k, v in output.get("probabilities", {}).items()
            }

            preds = {
                k: _to_numpy(v[i]) if v is not None else None
                for k, v in output.get("predictions", {}).items()
            }

            # ---------------- REPORT ----------------
            report = self.report_generator.generate_report(
                article_text=text,
                bias_analysis={"bias": preds.get("bias")},
                emotion_analysis={"emotion": preds.get("emotion")},
                credibility_score=preds.get("credibility_score"),
            )

            results.append(
                {
                    "text": text,

                    # 🔥 EVALUATION READY
                    "predictions": preds,
                    "probabilities": probs,
                    "logits": logits,

                    # 🔥 EXISTING OUTPUT
                    "report": report,
                }
            )

        return results

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        df = self._load_dataset()
        results = []

        for i in tqdm(range(0, len(df), self.config.batch_size)):

            batch = df.iloc[i:i + self.config.batch_size]
            texts = batch[self.config.text_column].fillna("").tolist()

            batch_results = self._process_batch(texts)
            results.extend(batch_results)

        return results

    # =====================================================
    # SAVE
    # =====================================================

    def save_results(self, results):
        with open(self.config.output_path, "w") as f:
            json.dump(results, f, indent=4)


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="predictions.json")

    args = parser.parse_args()

    engine = BatchInferenceEngine(
        BatchInferenceConfig(
            dataset_path=args.dataset,
            output_path=args.output,
        )
    )

    results = engine.run()
    engine.save_results(results)


if __name__ == "__main__":
    main()