from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

HF_REPO_ID = "bhavaygupta2002/truthlens2"
MAX_LENGTH = 512

# ---------------------------------------------------------------------------
# Label helpers — adapted directly from test.py
# ---------------------------------------------------------------------------

def _build_idx_to_label(model) -> dict[int, str]:
    """Use config.id2label first; fallback to label2id if needed."""
    idx_to_label: dict[int, str] = {}

    id2label = getattr(model.config, "id2label", None) or {}
    for idx, label in id2label.items():
        idx_to_label[int(idx)] = str(label).strip().upper()

    if idx_to_label:
        return idx_to_label

    label2id = getattr(model.config, "label2id", None) or {}
    for label, idx in label2id.items():
        idx_to_label[int(idx)] = str(label).strip().upper()

    if not idx_to_label:
        idx_to_label = {0: "REAL", 1: "FAKE"}

    return idx_to_label


def _get_label_index(idx_to_label: dict[int, str], target: str) -> int | None:
    target = target.strip().upper()
    for idx, label in idx_to_label.items():
        if label == target:
            return idx
    return None


# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_model: Optional[AutoModelForSequenceClassification] = None
_tokenizer: Optional[AutoTokenizer] = None
_idx_to_label: Optional[dict[int, str]] = None
_device: Optional[torch.device] = None


def _load_model() -> tuple[
    AutoModelForSequenceClassification,
    AutoTokenizer,
    dict[int, str],
    torch.device,
]:
    """Load the HuggingFace model and tokenizer once, then cache them."""
    global _model, _tokenizer, _idx_to_label, _device

    if _model is not None:
        return _model, _tokenizer, _idx_to_label, _device

    logger.info("Loading model from HuggingFace: %s", HF_REPO_ID)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID)
    model.to(device)
    model.eval()

    idx_to_label = _build_idx_to_label(model)

    logger.info(
        "Model loaded on %s | labels: %s",
        device,
        idx_to_label,
    )

    _model = model
    _tokenizer = tokenizer
    _idx_to_label = idx_to_label
    _device = device

    return _model, _tokenizer, _idx_to_label, _device


# ---------------------------------------------------------------------------
# Core inference — mirrors test.py logic
# ---------------------------------------------------------------------------

def _predict_single(text: str) -> dict[str, Any]:
    """Run inference on a single text string.

    Returns a dict with:
      - prediction        : "REAL" or "FAKE"
      - fake_probability  : float 0-1
      - real_probability  : float 0-1
      - confidence        : probability of the predicted class
      - class_probabilities: {label: probability} for every class
    """
    model, tokenizer, idx_to_label, device = _load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_label.get(pred_idx, f"CLASS_{pred_idx}")

    real_idx = _get_label_index(idx_to_label, "REAL")
    fake_idx = _get_label_index(idx_to_label, "FAKE")

    fake_prob = float(probs[fake_idx].item()) if fake_idx is not None else 0.0
    real_prob = float(probs[real_idx].item()) if real_idx is not None else 0.0
    confidence = float(probs[pred_idx].item())

    class_probabilities = {
        idx_to_label[i]: round(float(probs[i].item()), 6)
        for i in sorted(idx_to_label.keys())
    }

    return {
        "prediction": pred_label,
        "fake_probability": round(fake_prob, 6),
        "real_probability": round(real_prob, 6),
        "confidence": round(confidence, 6),
        "class_probabilities": class_probabilities,
    }


def _predict_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Run inference on a batch of texts in a single forward pass."""
    model, tokenizer, idx_to_label, device = _load_model()

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    all_probs = F.softmax(outputs.logits, dim=1)

    real_idx = _get_label_index(idx_to_label, "REAL")
    fake_idx = _get_label_index(idx_to_label, "FAKE")

    results = []
    for i, probs in enumerate(all_probs):
        pred_idx = int(torch.argmax(probs).item())
        pred_label = idx_to_label.get(pred_idx, f"CLASS_{pred_idx}")
        fake_prob = float(probs[fake_idx].item()) if fake_idx is not None else 0.0
        real_prob = float(probs[real_idx].item()) if real_idx is not None else 0.0
        confidence = float(probs[pred_idx].item())
        class_probabilities = {
            idx_to_label[j]: round(float(probs[j].item()), 6)
            for j in sorted(idx_to_label.keys())
        }
        results.append(
            {
                "prediction": pred_label,
                "fake_probability": round(fake_prob, 6),
                "real_probability": round(real_prob, 6),
                "confidence": round(confidence, 6),
                "class_probabilities": class_probabilities,
            }
        )
    return results


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TruthLens2 Fake-News Detection API",
    description=(
        "Fake / real news classifier powered by "
        f"[{HF_REPO_ID}](https://huggingface.co/{HF_REPO_ID}) "
        "(RoBERTa fine-tuned on multi-source news datasets)."
    ),
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": (
                    "Mixed messages from Trump leave more questions than answers "
                    "over war's end"
                )
            }
        }
    )
    text: str = Field(
        ...,
        min_length=10,
        max_length=10_000,
        description="News article text (or title + body) to classify.",
    )


class PredictResponse(BaseModel):
    text_preview: str = Field(..., description="First 200 characters of the input.")
    prediction: str = Field(..., description='"REAL" or "FAKE".')
    fake_probability: float = Field(..., ge=0, le=1)
    real_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted class.")
    class_probabilities: dict[str, float] = Field(
        ..., description="Per-class softmax probabilities."
    )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "Scientists confirm climate change is accelerating based on new data.",
                    "Government hiding truth about vaccines, insider reveals shocking secret.",
                ]
            }
        }
    )
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of news texts to classify (max 50).",
    )


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int


class HealthResponse(BaseModel):
    status: str
    model_repo: str
    model_loaded: bool
    device: Optional[str]
    labels: Optional[dict[str, Any]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Home")
def home():
    return {
        "message": "TruthLens2 Fake-News Detection API",
        "model": HF_REPO_ID,
        "status": "online",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "health": "/health",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health():
    """Returns model load status and runtime device."""
    loaded = _model is not None
    return HealthResponse(
        status="healthy" if loaded else "model_not_loaded",
        model_repo=HF_REPO_ID,
        model_loaded=loaded,
        device=str(_device) if _device is not None else None,
        labels=_idx_to_label,
    )


@app.post("/predict", response_model=PredictResponse, summary="Predict Single Article")
def predict_news(request: PredictRequest):
    """
    Classify a single news article as **REAL** or **FAKE**.

    Uses `bhavaygupta2002/truthlens2` (RoBERTa) — the same inference logic
    as `test.py`: tokenise → forward pass → softmax → argmax.
    """
    try:
        result = _predict_single(request.text)
        return PredictResponse(
            text_preview=request.text[:200],
            **result,
        )
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@app.post(
    "/batch-predict",
    response_model=BatchPredictResponse,
    summary="Predict Batch of Articles",
)
def batch_predict_news(request: BatchPredictRequest):
    """
    Classify up to 50 news articles in a single batched forward pass.

    All texts are padded/truncated to the same length and passed through the
    model together for efficiency.
    """
    try:
        if not request.texts:
            raise ValueError("texts list is empty")

        raw_results = _predict_batch(request.texts)

        responses = [
            PredictResponse(
                text_preview=text[:200],
                **result,
            )
            for text, result in zip(request.texts, raw_results)
        ]

        return BatchPredictResponse(results=responses, total=len(responses))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}")
