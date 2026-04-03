"""
File Name: run_inference.py
Module: inference
Description:
    Command-line interface (CLI) for running inference with the TruthLens AI
    system.

    This script loads a trained model using the PredictionPipeline and allows
    users to run predictions directly from the terminal.

    Example usage:
        python run_inference.py --model_dir models/truthlens --article "Some text"

    The script supports both single-article inference and batch inference
    through text files.

Author: ML Engineering System
Date: 2026-04-03
Dependencies:
    argparse
    json
    logging
    pathlib
    typing
    sys
    src.inference.run_inference (PredictionPipeline)
Inputs:
    --model_dir : Path to trained model directory
    --article : Single article text
    --input_file : File containing articles (one per line)
Outputs:
    Printed JSON prediction results
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
import sys
from pathlib import Path
from typing import List

from src.inference.inference_engine import InferenceConfig, InferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run inference using a trained TruthLens model."
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the trained model directory.",
    )

    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help="Single article text for prediction.",
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Text file containing articles (one per line).",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length for the tokenizer.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cpu or cuda).",
    )

    return parser.parse_args()


def load_text_file(path: str | Path) -> List[str]:
    """
    Load input texts from a file.

    Each line in the file represents a single article.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    texts: List[str] = []

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                texts.append(text)

    if not texts:
        raise ValueError("Input file contains no valid texts.")

    return texts


def run_single(engine: InferenceEngine, text: str) -> None:
    """
    Run inference for a single article.
    """

    logger.info("Running single-article inference.")

    result = asdict(engine.predict_single(text))

    print(json.dumps(result, indent=2))


def run_batch(engine: InferenceEngine, texts: List[str]) -> None:
    """
    Run inference for multiple articles.
    """

    logger.info("Running batch inference for %d texts.", len(texts))

    results = [asdict(item) for item in engine.predict(texts)]

    print(json.dumps(results, indent=2))


def main() -> None:
    """
    Main CLI entrypoint.
    """

    args = parse_args()

    if args.article is None and args.input_file is None:
        logger.error("You must provide either --article or --input_file.")
        sys.exit(1)

    try:
        engine = InferenceEngine(
            InferenceConfig(
                model_path=args.model_dir,
                tokenizer_path=None,
                max_length=args.max_length,
                device=args.device or "auto",
            )
        )

        if args.article:
            run_single(engine, args.article)

        if args.input_file:
            texts = load_text_file(args.input_file)
            run_batch(engine, texts)

    except Exception:
        logger.exception("Inference failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
