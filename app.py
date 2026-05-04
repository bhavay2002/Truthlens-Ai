"""
TruthLens AI — Render deployment entry point.

Designed for 512 MB RAM constraints:
  - Zero heavy ML imports at module level (no torch, transformers, spaCy at startup)
  - Predictions via the HuggingFace Inference API (no local model weights in RAM)
  - Lightweight regex/rules-based analyzers loaded lazily on first /analyze call
  - Port binds in < 1 second regardless of model availability
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("truthlens")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HF_REPO_V1 = "bhavaygupta2002/truthlens_v1"
HF_REPO_V2 = "bhavaygupta2002/truthlens2"
HF_API_BASE = "https://api-inference.huggingface.co/models"
MAX_TEXT_PREVIEW = 200
REQUEST_TIMEOUT = 30  # seconds per HuggingFace Inference API call

# ---------------------------------------------------------------------------
# HuggingFace Inference API helper
# ---------------------------------------------------------------------------

def _hf_headers() -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _hf_classify(model_id: str, text: str) -> list[dict]:
    """Call HuggingFace Inference API and return raw label/score list.

    Returns [] on any error (caller applies heuristic fallback).
    Handles the 503 'model loading' response with a single retry.
    """
    url = f"{HF_API_BASE}/{model_id}"
    payload = json.dumps({"inputs": text[:512]}).encode()
    headers = _hf_headers()

    for attempt in range(2):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read())
                # API returns [[{label, score}, ...]] for single input
                if isinstance(data, list) and data and isinstance(data[0], list):
                    return data[0]
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    return data
                return []
        except urllib.error.HTTPError as exc:
            if exc.code == 503 and attempt == 0:
                # Model is loading on HF side — wait and retry once
                logger.info("HF model %s loading, retrying in 10s …", model_id)
                time.sleep(10)
                continue
            logger.warning("HF Inference API error %s: %s", exc.code, exc)
            return []
        except Exception as exc:
            logger.warning("HF Inference API call failed: %s", exc)
            return []
    return []


def _heuristic_predict(text: str) -> dict:
    """Regex-based fallback when HuggingFace API is unavailable."""
    import re
    BIAS_WORDS = [
        "outrage", "shocking", "unbelievable", "exposed", "scandal",
        "hoax", "lie", "fake", "corrupt", "cover-up", "conspiracy",
        "manipulation", "propaganda", "disgrace",
    ]
    lower = text.lower()
    hits = sum(1 for w in BIAS_WORDS if re.search(r"\b" + w + r"\b", lower))
    exclamations = lower.count("!")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    score = min(hits * 0.08 + exclamations * 0.04 + caps_ratio * 0.3, 1.0)
    prediction = "FAKE" if score > 0.45 else "REAL"
    return {
        "fake_probability": round(score, 4),
        "real_probability": round(1.0 - score, 4),
        "prediction": prediction,
        "confidence": round(max(score, 1.0 - score), 4),
        "source": "heuristic_fallback",
    }


# ---------------------------------------------------------------------------
# Lazy analyzer registry (no torch/spaCy at import time)
# ---------------------------------------------------------------------------
_analyzers: dict[str, Any] = {}
_analyzer_lock = threading.Lock()
_analyzer_error: Optional[str] = None


def _get_analyzers() -> dict[str, Any]:
    """Load lightweight (non-ML) analyzers on first call."""
    global _analyzer_error
    if _analyzers:
        return _analyzers
    with _analyzer_lock:
        if _analyzers:
            return _analyzers
        try:
            from src.analysis.argument_mining import ArgumentMiningAnalyzer
            from src.analysis.bias_profile_builder import BiasProfileBuilder
            from src.analysis.context_omission_detector import ContextOmissionDetector
            from src.analysis.discourse_coherence_analyzer import DiscourseCoherenceAnalyzer
            from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
            from src.analysis.framing_analysis import FramingAnalyzer
            from src.analysis.ideological_language_detector import IdeologicalLanguageDetector
            from src.analysis.information_density_analyzer import InformationDensityAnalyzer
            from src.analysis.information_omission_detector import InformationOmissionDetector
            from src.analysis.narrative_conflict import NarrativeConflictAnalyzer
            from src.analysis.narrative_propagation import NarrativePropagationAnalyzer
            from src.analysis.narrative_role_extractor import NarrativeRoleExtractor
            from src.analysis.narrative_temporal_analyzer import NarrativeTemporalAnalyzer
            from src.analysis.propaganda_pattern_detector import PropagandaPatternDetector
            from src.analysis.rhetorical_device_detector import RhetoricalDeviceDetector
            from src.analysis.source_attribution_analyzer import SourceAttributionAnalyzer
            from src.features.bias.bias_lexicon import compute_bias_features
            from src.features.emotion.emotion_lexicon import EmotionLexiconAnalyzer

            _analyzers.update({
                "argument": ArgumentMiningAnalyzer(),
                "bias_profile": BiasProfileBuilder(),
                "context_omission": ContextOmissionDetector(),
                "discourse": DiscourseCoherenceAnalyzer(),
                "emotion_target": EmotionTargetAnalyzer(),
                "framing": FramingAnalyzer(),
                "ideological": IdeologicalLanguageDetector(),
                "info_density": InformationDensityAnalyzer(),
                "info_omission": InformationOmissionDetector(),
                "narrative_conflict": NarrativeConflictAnalyzer(),
                "narrative_propagation": NarrativePropagationAnalyzer(),
                "narrative_role": NarrativeRoleExtractor(),
                "narrative_temporal": NarrativeTemporalAnalyzer(),
                "propaganda": PropagandaPatternDetector(),
                "rhetorical": RhetoricalDeviceDetector(),
                "source_attribution": SourceAttributionAnalyzer(),
                "compute_bias_features": compute_bias_features,
                "emotion_lexicon": EmotionLexiconAnalyzer(),
            })
            logger.info("Analyzers loaded (%d)", len(_analyzers))
        except Exception:
            _analyzer_error = traceback.format_exc()
            logger.error("Analyzer load failed:\n%s", _analyzer_error)
    return _analyzers


def _safe_run(fn, *args, **kwargs) -> dict:
    try:
        result = fn(*args, **kwargs)
        if hasattr(result, "__dict__"):
            return vars(result)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        logger.warning("Analyzer %s failed: %s", getattr(fn, "__name__", fn), exc)
        return {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class NewsRequest(BaseModel):
    text: str


class BatchNewsRequest(BaseModel):
    texts: List[str]


class V2PredictRequest(BaseModel):
    text: str


class V2BatchPredictRequest(BaseModel):
    texts: List[str]


# ---------------------------------------------------------------------------
# FastAPI app  (created at module level — zero heavy imports above)
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TruthLens AI — Unified API",
    description=(
        "Multi-task NLP misinformation detection platform. "
        "Predictions are powered by the HuggingFace Inference API "
        "(bhavaygupta2002/truthlens_v1 and bhavaygupta2002/truthlens2)."
    ),
    version="2.1.0",
)

_START_TIME = time.time()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "TruthLens AI — Unified API",
        "status": "online",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "models": {
            "truthlens_v1": f"https://huggingface.co/{HF_REPO_V1}",
            "truthlens2": f"https://huggingface.co/{HF_REPO_V2}",
        },
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "v2_predict": "/v2/predict",
            "v2_batch_predict": "/v2/batch-predict",
            "analyze": "/analyze",
            "health": "/health",
            "v2_health": "/v2/health",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "inference": "huggingface_inference_api",
        "hf_repo_v1": HF_REPO_V1,
        "hf_repo_v2": HF_REPO_V2,
        "analyzers_loaded": bool(_analyzers),
    }


@app.get("/v2/health")
def v2_health():
    return {
        "status": "healthy",
        "model_repo": HF_REPO_V2,
        "inference": "huggingface_inference_api",
    }


@app.get("/project-view")
def project_view():
    return {
        "project": "TruthLens AI",
        "version": "2.1.0",
        "tasks": ["bias", "ideology", "propaganda", "emotion", "roles", "frames"],
        "models": [HF_REPO_V1, HF_REPO_V2],
    }


# ---------------------------------------------------------------------------
# /predict  (truthlens_v1 via HuggingFace Inference API)
# ---------------------------------------------------------------------------

@app.post("/predict")
def predict_news(request: NewsRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    results = _hf_classify(HF_REPO_V1, text)
    if results:
        label_scores = {r["label"].upper(): r["score"] for r in results}
        fake_prob = label_scores.get("FAKE", label_scores.get("LABEL_1", 0.5))
        real_prob = label_scores.get("REAL", label_scores.get("LABEL_0", 1.0 - fake_prob))
        prediction = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = round(max(fake_prob, real_prob), 4)
        source = "huggingface_inference_api"
    else:
        fb = _heuristic_predict(text)
        fake_prob = fb["fake_probability"]
        real_prob = fb["real_probability"]
        prediction = fb["prediction"]
        confidence = fb["confidence"]
        source = "heuristic_fallback"

    return {
        "text": text[:MAX_TEXT_PREVIEW],
        "prediction": prediction,
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "confidence": confidence,
        "source": source,
    }


@app.post("/batch-predict")
def batch_predict_news(request: BatchNewsRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="maximum 50 texts per batch")

    results = []
    for text in request.texts:
        text = text.strip()
        hf_results = _hf_classify(HF_REPO_V1, text)
        if hf_results:
            label_scores = {r["label"].upper(): r["score"] for r in hf_results}
            fake_prob = label_scores.get("FAKE", label_scores.get("LABEL_1", 0.5))
            real_prob = label_scores.get("REAL", label_scores.get("LABEL_0", 1.0 - fake_prob))
            prediction = "FAKE" if fake_prob > 0.5 else "REAL"
            confidence = round(max(fake_prob, real_prob), 4)
            source = "huggingface_inference_api"
        else:
            fb = _heuristic_predict(text)
            fake_prob, real_prob = fb["fake_probability"], fb["real_probability"]
            prediction, confidence = fb["prediction"], fb["confidence"]
            source = "heuristic_fallback"
        results.append({
            "text": text[:MAX_TEXT_PREVIEW],
            "prediction": prediction,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
            "confidence": confidence,
            "source": source,
        })

    return {"results": results, "total": len(results)}


# ---------------------------------------------------------------------------
# /v2/predict  (truthlens2 via HuggingFace Inference API)
# ---------------------------------------------------------------------------

@app.post("/v2/predict")
def v2_predict_news(request: V2PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    results = _hf_classify(HF_REPO_V2, text)
    if results:
        label_scores = {r["label"].upper(): r["score"] for r in results}
        fake_prob = label_scores.get("FAKE", label_scores.get("LABEL_1", 0.5))
        real_prob = label_scores.get("REAL", label_scores.get("LABEL_0", 1.0 - fake_prob))
        prediction = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = round(max(fake_prob, real_prob), 6)
        class_probabilities = {k: round(v, 6) for k, v in label_scores.items()}
        source = "huggingface_inference_api"
    else:
        fb = _heuristic_predict(text)
        fake_prob, real_prob = fb["fake_probability"], fb["real_probability"]
        prediction, confidence = fb["prediction"], fb["confidence"]
        class_probabilities = {"FAKE": round(fake_prob, 6), "REAL": round(real_prob, 6)}
        source = "heuristic_fallback"

    return {
        "text_preview": text[:MAX_TEXT_PREVIEW],
        "prediction": prediction,
        "fake_probability": round(fake_prob, 6),
        "real_probability": round(real_prob, 6),
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "source": source,
    }


@app.post("/v2/batch-predict")
def v2_batch_predict_news(request: V2BatchPredictRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="maximum 50 texts per batch")

    results = []
    for text in request.texts:
        text = text.strip()
        hf_results = _hf_classify(HF_REPO_V2, text)
        if hf_results:
            label_scores = {r["label"].upper(): r["score"] for r in hf_results}
            fake_prob = label_scores.get("FAKE", label_scores.get("LABEL_1", 0.5))
            real_prob = label_scores.get("REAL", label_scores.get("LABEL_0", 1.0 - fake_prob))
            prediction = "FAKE" if fake_prob > 0.5 else "REAL"
            confidence = round(max(fake_prob, real_prob), 6)
            class_probabilities = {k: round(v, 6) for k, v in label_scores.items()}
            source = "huggingface_inference_api"
        else:
            fb = _heuristic_predict(text)
            fake_prob, real_prob = fb["fake_probability"], fb["real_probability"]
            prediction, confidence = fb["prediction"], fb["confidence"]
            class_probabilities = {"FAKE": round(fake_prob, 6), "REAL": round(real_prob, 6)}
            source = "heuristic_fallback"
        results.append({
            "text_preview": text[:MAX_TEXT_PREVIEW],
            "prediction": prediction,
            "fake_probability": round(fake_prob, 6),
            "real_probability": round(real_prob, 6),
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "source": source,
        })

    return {"results": results, "total": len(results)}


# ---------------------------------------------------------------------------
# /analyze  (lazy lightweight analyzers — no torch/transformers)
# ---------------------------------------------------------------------------

@app.post("/analyze")
def analyze_news(request: NewsRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    # Run prediction via HF API
    hf_results = _hf_classify(HF_REPO_V2, text)
    if hf_results:
        label_scores = {r["label"].upper(): r["score"] for r in hf_results}
        fake_prob = label_scores.get("FAKE", label_scores.get("LABEL_1", 0.5))
        prediction = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = round(max(fake_prob, 1.0 - fake_prob), 4)
    else:
        fb = _heuristic_predict(text)
        fake_prob = fb["fake_probability"]
        prediction = fb["prediction"]
        confidence = fb["confidence"]

    # Lazy-load lightweight analyzers
    az = _get_analyzers()
    if not az:
        return JSONResponse({
            "status": "degraded",
            "prediction": prediction,
            "fake_probability": round(fake_prob, 4),
            "confidence": confidence,
            "error": "Analyzers unavailable",
            "analyzer_error": _analyzer_error,
        }, status_code=200)

    compute_bias = az.get("compute_bias_features")
    bias_result = _safe_run(compute_bias, text) if compute_bias else {}

    emotion_lex = az.get("emotion_lexicon")
    emotion_result = _safe_run(emotion_lex.analyze, text) if emotion_lex else {}
    emotion_scores: dict = getattr(emotion_result, "emotion_scores", {}) if hasattr(emotion_result, "emotion_scores") else {}
    if not emotion_scores and isinstance(emotion_result, dict):
        emotion_scores = emotion_result.get("emotion_scores", {})

    narrative_role = _safe_run(az["narrative_role"].analyze, text) if "narrative_role" in az else {}
    hero_entities = narrative_role.get("hero_entities", [])
    villain_entities = narrative_role.get("villain_entities", [])
    victim_entities = narrative_role.get("victim_entities", [])

    narrative_conflict = _safe_run(
        az["narrative_conflict"].analyze, text,
        hero_entities=hero_entities, villain_entities=villain_entities, victim_entities=victim_entities
    ) if "narrative_conflict" in az else {}

    narrative_propagation = _safe_run(
        az["narrative_propagation"].analyze, text,
        hero_entities=hero_entities, villain_entities=villain_entities, victim_entities=victim_entities
    ) if "narrative_propagation" in az else {}

    narrative_temporal = _safe_run(az["narrative_temporal"].analyze, text) if "narrative_temporal" in az else {}
    framing = _safe_run(az["framing"].analyze, text) if "framing" in az else {}
    rhetorical = _safe_run(az["rhetorical"].analyze, text) if "rhetorical" in az else {}
    argument = _safe_run(az["argument"].analyze, text) if "argument" in az else {}
    info_density = _safe_run(az["info_density"].analyze, text) if "info_density" in az else {}
    info_omission = _safe_run(az["info_omission"].analyze, text) if "info_omission" in az else {}
    context_omission = _safe_run(az["context_omission"].analyze, text) if "context_omission" in az else {}
    discourse = _safe_run(az["discourse"].analyze, text) if "discourse" in az else {}
    ideological = _safe_run(az["ideological"].analyze, text) if "ideological" in az else {}
    emotion_target = _safe_run(az["emotion_target"].analyze, text) if "emotion_target" in az else {}
    source_attribution = _safe_run(az["source_attribution"].analyze, text) if "source_attribution" in az else {}

    combined_narrative = {**narrative_conflict, **narrative_propagation, **narrative_temporal}
    combined_info = {**info_density, **info_omission}
    combined_discourse = {**discourse, **context_omission, **info_density, **info_omission, **source_attribution}

    propaganda_patterns = {}
    if "propaganda" in az:
        propaganda_patterns = _safe_run(
            az["propaganda"].analyze,
            emotion_features=emotion_scores,
            narrative_features=combined_narrative,
            rhetorical_features=rhetorical,
            argument_features=argument,
            information_features=combined_info,
        )

    credibility_profile = {}
    if "bias_profile" in az:
        credibility_profile = _safe_run(
            az["bias_profile"].build_profile,
            bias={"bias_score": float(bias_result.get("bias_score", 0.0))},
            emotion=emotion_scores,
            narrative=combined_narrative,
            discourse=combined_discourse,
            ideology=ideological,
        )

    return {
        "text_preview": text[:MAX_TEXT_PREVIEW],
        "prediction": prediction,
        "fake_probability": round(fake_prob, 4),
        "confidence": confidence,
        "bias": bias_result,
        "emotion": {"emotion_scores": emotion_scores},
        "narrative": {
            "roles": narrative_role,
            "conflict": narrative_conflict,
            "propagation": narrative_propagation,
            "temporal": narrative_temporal,
        },
        "framing": framing,
        "rhetorical_devices": rhetorical,
        "argument_mining": argument,
        "information": combined_info,
        "discourse_coherence": discourse,
        "context_omission": context_omission,
        "ideological_language": ideological,
        "emotion_targets": emotion_target,
        "source_attribution": source_attribution,
        "propaganda_patterns": propaganda_patterns,
        "credibility_profile": credibility_profile,
    }


# ---------------------------------------------------------------------------
# Stub endpoints for compatibility
# ---------------------------------------------------------------------------

@app.post("/explain")
def explain(request: NewsRequest):
    return JSONResponse(
        {"status": "unavailable", "message": "Explainability requires local model. Use /predict for inference."},
        status_code=503,
    )


@app.post("/report")
def report(request: NewsRequest):
    result = predict_news(request)
    return {
        "report": {
            "text_preview": result.get("text", ""),
            "prediction": result.get("prediction"),
            "fake_probability": result.get("fake_probability"),
            "confidence": result.get("confidence"),
            "model": HF_REPO_V2,
        }
    }


@app.get("/inference/model-info")
def model_info():
    return {
        "inference_mode": "huggingface_inference_api",
        "models": {"v1": HF_REPO_V1, "v2": HF_REPO_V2},
        "local_model_loaded": False,
        "note": "Running on HuggingFace Inference API to stay within 512 MB RAM.",
    }


@app.post("/cache/clear")
def cache_clear():
    return {"status": "ok", "message": "No local cache to clear (API-mode)"}


@app.get("/calibration/info")
def calibration_info():
    return {"status": "unavailable", "message": "Calibration requires local model."}


@app.post("/calibration/metrics")
def calibration_metrics(request: dict = {}):
    raise HTTPException(503, "Calibration requires local model.")


@app.get("/ensemble/info")
def ensemble_info():
    return {"status": "unavailable", "message": "Ensemble requires local model."}


@app.post("/ensemble/predict")
def ensemble_predict(request: NewsRequest):
    return predict_news(request)


@app.get("/export/info")
def export_info():
    return {"status": "unavailable", "message": "Export requires local model."}


@app.post("/export/onnx")
def export_onnx(request: dict = {}):
    raise HTTPException(503, "Export requires local model.")


@app.post("/export/torchscript")
def export_torchscript(request: dict = {}):
    raise HTTPException(503, "Export requires local model.")
