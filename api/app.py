from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from models.inference.predictor import predict, predict_batch
from src.analysis.argument_mining import ArgumentMiningAnalyzer
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.graph.graph_pipeline import GraphPipeline
from src.graph.graph_embeddings import GraphEmbeddingGenerator
from src.graph.temporal_graph import TemporalGraphAnalyzer
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
from src.explainability.emotion_explainer import explain_emotion
from src.explainability.lime_explainer import explain_prediction
from src.utils.logging_utils import configure_logging
from src.utils.settings import load_settings

# ── src.inference integration ──────────────────────────────────────────────────
from src.inference.inference_cache import InferenceCache, InferenceCacheConfig
from src.inference.inference_logger import InferenceLogger
from src.inference.result_formatter import ResultFormatter
from src.inference.report_generator import ReportGenerator, ReportConfig
from src.inference.inference_engine import (
    InferenceEngine,
    InferenceConfig as EngineConfig,
)

configure_logging()
logger = logging.getLogger(__name__)
SETTINGS = load_settings()
MODEL_PATH = SETTINGS.model.path
VECTORIZER_PATH = SETTINGS.paths.tfidf_vectorizer_path
TRAINING_TEXT_COLUMN = SETTINGS.training.text_column
APP_TITLE = SETTINGS.api.title
APP_DESCRIPTION = SETTINGS.api.description
APP_VERSION = SETTINGS.api.version
TEXT_PREVIEW_CHARS = max(int(SETTINGS.api.text_preview_chars), 1)
INFERENCE_ALLOW_RAW_TEXT_FALLBACK = bool(SETTINGS.inference.allow_raw_text_fallback)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SUBPACKAGES = (
    "emotion",
    "encoder",
    "ideology",
    "multitask",
    "narrative",
    "propaganda",
)
LIME_NUM_SAMPLES = 16

# ── Singleton analyzers ────────────────────────────────────────────────────────
EMOTION_ANALYZER = EmotionLexiconAnalyzer()
ARGUMENT_ANALYZER = ArgumentMiningAnalyzer()
BIAS_PROFILE_BUILDER = BiasProfileBuilder()
CONTEXT_OMISSION_DETECTOR = ContextOmissionDetector()
DISCOURSE_ANALYZER = DiscourseCoherenceAnalyzer()
EMOTION_TARGET_ANALYZER = EmotionTargetAnalyzer()
FRAMING_ANALYZER = FramingAnalyzer()
IDEOLOGICAL_DETECTOR = IdeologicalLanguageDetector()
INFO_DENSITY_ANALYZER = InformationDensityAnalyzer()
INFO_OMISSION_DETECTOR = InformationOmissionDetector()
NARRATIVE_CONFLICT_ANALYZER = NarrativeConflictAnalyzer()
NARRATIVE_PROPAGATION_ANALYZER = NarrativePropagationAnalyzer()
NARRATIVE_ROLE_EXTRACTOR = NarrativeRoleExtractor()
NARRATIVE_TEMPORAL_ANALYZER = NarrativeTemporalAnalyzer()
PROPAGANDA_PATTERN_DETECTOR = PropagandaPatternDetector()
RHETORICAL_DETECTOR = RhetoricalDeviceDetector()
SOURCE_ATTRIBUTION_ANALYZER = SourceAttributionAnalyzer()
GRAPH_PIPELINE = GraphPipeline()
GRAPH_EMBEDDING_GENERATOR = GraphEmbeddingGenerator()
TEMPORAL_GRAPH_ANALYZER = TemporalGraphAnalyzer()

# ── src.inference singletons ───────────────────────────────────────────────────
INFERENCE_CACHE = InferenceCache(
    InferenceCacheConfig(
        cache_dir="cache/inference",
        enable_disk_cache=False,
        enable_memory_cache=True,
        ttl_seconds=3600,
    )
)

INFERENCE_LOGGER = InferenceLogger(service_name="truthlens-api", enable_json_logs=True)

RESULT_FORMATTER = ResultFormatter()

REPORT_GENERATOR = ReportGenerator(
    ReportConfig(include_timestamp=True, pretty_json=False, validate_fields=True)
)

# InferenceEngine is initialised lazily so a missing model directory does not
# crash the server at startup.
_INFERENCE_ENGINE: Optional[InferenceEngine] = None


def _get_inference_engine() -> Optional[InferenceEngine]:
    """Return the shared InferenceEngine, or None if the model is not yet trained."""
    global _INFERENCE_ENGINE
    if _INFERENCE_ENGINE is None and MODEL_PATH.exists():
        try:
            _INFERENCE_ENGINE = InferenceEngine(
                EngineConfig(
                    model_path=str(MODEL_PATH),
                    tokenizer_path=str(MODEL_PATH),
                    max_length=SETTINGS.model.max_length,
                    device=SETTINGS.inference.device,
                )
            )
            logger.info("InferenceEngine initialised from %s", MODEL_PATH)
        except Exception as exc:
            logger.warning("InferenceEngine could not be initialised: %s", exc)
    return _INFERENCE_ENGINE


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)


# ── Request / response models ──────────────────────────────────────────────────

class NewsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Breaking news: Scientists discover new species in Amazon rainforest."
            }
        }
    )
    text: str = Field(..., min_length=10, max_length=10_000, description="News article text to analyze")


class BatchNewsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "texts": [
                    "First news article text here.",
                    "Second news article text here.",
                ]
            }
        }
    )
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of news article texts to analyze (max 50 items)",
    )


class NewsResponse(BaseModel):
    text: str
    fake_probability: float = Field(..., ge=0, le=1, description="Probability of being fake news (0-1)")
    prediction: str
    confidence: float


class BatchNewsResponse(BaseModel):
    results: list[NewsResponse]
    total: int
    cache_hits: int


class AnalysisResponse(BaseModel):
    text: str
    prediction: str
    fake_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    bias: dict[str, Any]
    emotion: dict[str, Any]
    narrative: dict[str, Any]
    framing: dict[str, Any]
    rhetoric: dict[str, Any]
    discourse: dict[str, Any]
    propaganda_analysis: dict[str, Any]
    credibility_profile: dict[str, Any]
    graph_analysis: dict[str, Any]
    explainability: dict[str, Any]


class ReportResponse(BaseModel):
    article_summary: dict[str, Any]
    bias_analysis: dict[str, Any]
    emotion_analysis: dict[str, Any]
    narrative_structure: dict[str, Any]
    entity_graph: dict[str, Any]
    credibility_score: Optional[float]


class ModelInfoResponse(BaseModel):
    available: bool
    model_path: str
    device: Optional[str]
    num_parameters: Optional[int]
    num_trainable_parameters: Optional[int]
    label_map: Optional[dict[str, Any]]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _preview_text(text: str) -> str:
    if len(text) <= TEXT_PREVIEW_CHARS:
        return text
    return text[:TEXT_PREVIEW_CHARS] + "..."


def _safe_run(fn, *args, **kwargs) -> dict:
    """Call an analysis function; return empty dict on any error so other
    sections of the /analyze response are unaffected."""
    try:
        result = fn(*args, **kwargs)
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        name = getattr(fn, "__qualname__", type(fn).__name__)
        logger.warning("Analysis step '%s' failed: %s", name, exc)
        return {}


def _serialize_graph_result(result: dict) -> dict:
    """Convert any numpy arrays in a graph pipeline result to Python lists
    so the response is JSON-serializable."""
    out = {}
    for k, v in result.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _build_project_view() -> dict[str, Any]:
    src_dir = PROJECT_ROOT / "src"
    model_dir = src_dir / "models"

    model_subpackages = {}
    for subpackage in MODEL_SUBPACKAGES:
        package_dir = model_dir / subpackage
        model_subpackages[subpackage] = {
            "directory_exists": package_dir.exists(),
            "package_init_exists": (package_dir / "__init__.py").exists(),
        }

    return {
        "project_root": str(PROJECT_ROOT),
        "api": {
            "title": APP_TITLE,
            "version": APP_VERSION,
            "description": APP_DESCRIPTION,
        },
        "config": {
            "model_name": SETTINGS.model.name,
            "model_path": str(MODEL_PATH),
            "training_text_column": TRAINING_TEXT_COLUMN,
            "vectorizer_path": str(VECTORIZER_PATH),
        },
        "structure": {
            "src_exists": src_dir.exists(),
            "api_exists": (PROJECT_ROOT / "api").exists(),
            "config_exists": (PROJECT_ROOT / "config").exists(),
            "tests_exists": (PROJECT_ROOT / "tests").exists(),
            "models_package_init_exists": (model_dir / "__init__.py").exists(),
            "model_subpackages": model_subpackages,
        },
    }


def _decode_prediction_result(prediction_result) -> tuple[float, str, float]:
    """Normalise the output of models.inference.predictor.predict into
    (fake_probability, prediction_label, confidence)."""
    if isinstance(prediction_result, dict):
        prob = float(prediction_result.get("fake_probability", 0.0))
        prediction = str(prediction_result.get("label", "Fake")).upper()
        confidence = float(prediction_result.get("confidence", max(prob, 1 - prob)))
    else:
        prob = float(prediction_result)
        prediction = "FAKE" if prob > 0.5 else "REAL"
        confidence = prob if prob > 0.5 else (1 - prob)
    return prob, prediction, confidence


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "message": APP_TITLE,
        "status": "online",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "analyze": "/analyze",
            "report": "/report",
            "inference_model_info": "/inference/model-info",
            "cache_clear": "/cache/clear",
            "health": "/health",
            "project_view": "/project-view",
            "docs": "/docs",
        },
    }


@app.get("/project-view")
def project_view():
    """Project-level view of API metadata, configuration, and package layout."""
    return _build_project_view()


@app.post("/predict", response_model=NewsResponse)
def predict_news(request: NewsRequest):
    """
    Predict whether a news article is fake or real.

    Results are cached in memory for one hour — repeated submissions of
    identical text are served instantly without rerunning the model.
    """
    try:
        logger.info("Received /predict request (text length: %d)", len(request.text))
        timer_start = INFERENCE_LOGGER.start_timer()

        # ── Cache lookup ───────────────────────────────────────────────────────
        cached = INFERENCE_CACHE.get(request.text)
        if cached is not None:
            logger.debug("Cache hit for /predict")
            return NewsResponse(**cached)

        # ── Model inference ────────────────────────────────────────────────────
        prediction_result = predict(request.text)
        prob, prediction, confidence = _decode_prediction_result(prediction_result)

        # ── Format via ResultFormatter ─────────────────────────────────────────
        raw_prediction = {
            "bias": None,
            "ideology": None,
            "propaganda_probability": None,
            "credibility_score": round(1.0 - prob, 4),
        }
        RESULT_FORMATTER.format_api_response(raw_prediction)  # validates schema

        response_data = {
            "text": _preview_text(request.text),
            "fake_probability": round(prob, 4),
            "prediction": prediction,
            "confidence": round(confidence, 4),
        }

        # ── Store in cache ─────────────────────────────────────────────────────
        INFERENCE_CACHE.set(request.text, response_data)

        # ── Structured inference log ───────────────────────────────────────────
        INFERENCE_LOGGER.log_prediction(
            article_id=None,
            start_time=timer_start,
            model_versions={"roberta": SETTINGS.model.name},
            feature_count=0,
            prediction_confidence=round(confidence, 4),
        )

        logger.info("Prediction: %s (confidence: %.4f)", prediction, confidence)
        return NewsResponse(**response_data)

    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available. Please train the model first.")
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during prediction")


@app.post("/batch-predict", response_model=BatchNewsResponse)
def batch_predict_news(request: BatchNewsRequest):
    """
    Predict fake/real for a batch of news articles (up to 50 at a time).

    Each text is individually checked against the cache before running the
    model, so cached entries are served without additional inference cost.
    """
    try:
        logger.info("Received /batch-predict request (%d texts)", len(request.texts))

        results: list[NewsResponse] = []
        cache_hits = 0
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # ── Separate cache hits from texts that need inference ─────────────────
        for i, text in enumerate(request.texts):
            cached = INFERENCE_CACHE.get(text)
            if cached is not None:
                results.append(NewsResponse(**cached))
                cache_hits += 1
            else:
                results.append(None)  # placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)

        # ── Run batch inference on uncached texts ──────────────────────────────
        if uncached_texts:
            timer_start = INFERENCE_LOGGER.start_timer()

            engine = _get_inference_engine()

            if engine is not None:
                # Use InferenceEngine for batch inference
                engine_results = engine.predict(uncached_texts)
                for idx, engine_result in zip(uncached_indices, engine_results):
                    text = request.texts[idx]
                    probs = engine_result.probabilities or [0.5, 0.5]
                    fake_index = 1
                    prob = round(float(probs[fake_index]), 4)
                    confidence = round(max(probs), 4)
                    prediction = "FAKE" if prob > 0.5 else "REAL"
                    response_data = {
                        "text": _preview_text(text),
                        "fake_probability": prob,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                    INFERENCE_CACHE.set(text, response_data)
                    results[idx] = NewsResponse(**response_data)
            else:
                # Fall back to existing predict_batch function
                batch_probs = predict_batch(uncached_texts)
                for idx, probs, text in zip(uncached_indices, batch_probs, uncached_texts):
                    prob_real, prob_fake = float(probs[0]), float(probs[1])
                    prob = round(prob_fake, 4)
                    confidence = round(max(prob_real, prob_fake), 4)
                    prediction = "FAKE" if prob > 0.5 else "REAL"
                    response_data = {
                        "text": _preview_text(text),
                        "fake_probability": prob,
                        "prediction": prediction,
                        "confidence": confidence,
                    }
                    INFERENCE_CACHE.set(text, response_data)
                    results[idx] = NewsResponse(**response_data)

            INFERENCE_LOGGER.log_prediction(
                article_id=None,
                start_time=timer_start,
                model_versions={"roberta": SETTINGS.model.name},
                feature_count=0,
                prediction_confidence=None,
            )

        return BatchNewsResponse(
            results=results,
            total=len(request.texts),
            cache_hits=cache_hits,
        )

    except FileNotFoundError as exc:
        logger.error("Model not found: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available. Please train the model first.")
    except ValueError as exc:
        logger.error("Invalid batch input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")


@app.get("/health")
def health_check():
    """Detailed health check."""
    try:
        model_exists = MODEL_PATH.exists()
        vectorizer_required = TRAINING_TEXT_COLUMN == "engineered_text"
        vectorizer_exists = (not vectorizer_required) or VECTORIZER_PATH.exists()
        vectorizer_fallback_enabled = INFERENCE_ALLOW_RAW_TEXT_FALLBACK
        vectorizer_effective_ready = (
            vectorizer_exists
            if not vectorizer_required
            else (vectorizer_exists or vectorizer_fallback_enabled)
        )

        required_files = ["config.json", "tokenizer.json"]
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        has_weight_file = any((MODEL_PATH / f).exists() for f in weight_files) if model_exists else False
        model_files_exist = (
            all((MODEL_PATH / f).exists() for f in required_files) and has_weight_file
            if model_exists
            else False
        )

        cache_size = len(INFERENCE_CACHE.memory_cache)

        return {
            "status": (
                "healthy"
                if model_exists and model_files_exist and vectorizer_effective_ready
                else "degraded"
            ),
            "model_path": str(MODEL_PATH),
            "model_exists": model_exists,
            "model_files_complete": model_files_exist,
            "training_text_column": TRAINING_TEXT_COLUMN,
            "vectorizer_required": vectorizer_required,
            "vectorizer_exists": vectorizer_exists,
            "vectorizer_fallback_enabled": vectorizer_fallback_enabled,
            "vectorizer_effective_ready": vectorizer_effective_ready,
            "vectorizer_path": str(VECTORIZER_PATH),
            "inference_cache_entries": cache_size,
        }
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return {
            "status": "unhealthy",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_news(request: NewsRequest):
    """
    Unified deep-analysis endpoint.

    Returns model prediction plus the full suite of linguistic, narrative,
    framing, rhetoric, discourse, propaganda-pattern, and credibility-profile
    analyses.

    Full analysis results are cached for one hour so repeated requests for
    the same text are served without re-running all subsystems.
    """
    try:
        logger.info("Received /analyze request (text length: %d)", len(request.text))
        timer_start = INFERENCE_LOGGER.start_timer()

        # ── Cache lookup ───────────────────────────────────────────────────────
        cache_key = f"analyze:{request.text}"
        cached = INFERENCE_CACHE.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for /analyze")
            return AnalysisResponse(**cached)

        # ── 1. Model prediction ────────────────────────────────────────────────
        prediction_result = predict(request.text)
        fake_probability, prediction, confidence = _decode_prediction_result(prediction_result)

        # ── 2. Bias + emotion (lexicon-based) ─────────────────────────────────
        bias_result = compute_bias_features(request.text)
        emotion_result = EMOTION_ANALYZER.analyze(request.text)
        emotion_scores: dict[str, float] = getattr(emotion_result, "emotion_scores", {})

        # ── 3. Narrative analysis ──────────────────────────────────────────────
        narrative_roles: dict = _safe_run(NARRATIVE_ROLE_EXTRACTOR.analyze, request.text)
        hero_entities: list = narrative_roles.get("hero_entities", [])
        villain_entities: list = narrative_roles.get("villain_entities", [])
        victim_entities: list = narrative_roles.get("victim_entities", [])

        narrative_conflict: dict = _safe_run(
            NARRATIVE_CONFLICT_ANALYZER.analyze,
            request.text,
            hero_entities=hero_entities,
            villain_entities=villain_entities,
            victim_entities=victim_entities,
        )
        narrative_propagation: dict = _safe_run(
            NARRATIVE_PROPAGATION_ANALYZER.analyze,
            request.text,
            hero_entities=hero_entities,
            villain_entities=villain_entities,
            victim_entities=victim_entities,
        )
        narrative_temporal: dict = _safe_run(NARRATIVE_TEMPORAL_ANALYZER.analyze, request.text)

        # ── 4. Framing ─────────────────────────────────────────────────────────
        framing: dict = _safe_run(FRAMING_ANALYZER.analyze, request.text)

        # ── 5. Rhetoric + argument structure ──────────────────────────────────
        rhetorical: dict = _safe_run(RHETORICAL_DETECTOR.analyze, request.text)
        argument: dict = _safe_run(ARGUMENT_ANALYZER.analyze, request.text)

        # ── 6. Discourse-level analyses ────────────────────────────────────────
        info_density: dict = _safe_run(INFO_DENSITY_ANALYZER.analyze, request.text)
        info_omission: dict = _safe_run(INFO_OMISSION_DETECTOR.analyze, request.text)
        context_omission: dict = _safe_run(CONTEXT_OMISSION_DETECTOR.analyze, request.text)
        discourse_coherence: dict = _safe_run(DISCOURSE_ANALYZER.analyze, request.text)
        ideological: dict = _safe_run(IDEOLOGICAL_DETECTOR.analyze, request.text)
        emotion_target: dict = _safe_run(EMOTION_TARGET_ANALYZER.analyze, request.text)
        source_attribution: dict = _safe_run(SOURCE_ATTRIBUTION_ANALYZER.analyze, request.text)

        # ── 7. Propaganda pattern detection ───────────────────────────────────
        combined_narrative: dict = {**narrative_conflict, **narrative_propagation, **narrative_temporal}
        combined_info: dict = {**info_density, **info_omission}
        propaganda_patterns: dict = _safe_run(
            PROPAGANDA_PATTERN_DETECTOR.analyze,
            emotion_features=emotion_scores,
            narrative_features=combined_narrative,
            rhetorical_features=rhetorical,
            argument_features=argument,
            information_features=combined_info,
        )

        # ── 8. Credibility profile ─────────────────────────────────────────────
        combined_discourse: dict = {
            **discourse_coherence,
            **context_omission,
            **info_density,
            **info_omission,
            **source_attribution,
        }
        credibility_profile: dict = _safe_run(
            BIAS_PROFILE_BUILDER.build_profile,
            bias_features={"bias_score": float(bias_result.bias_score)},
            emotion_features=emotion_scores,
            narrative_features=combined_narrative,
            discourse_features=combined_discourse,
            ideology_predictions=ideological,
        )

        # ── 9. Graph analysis ──────────────────────────────────────────────────
        raw_graph_result: dict = _safe_run(GRAPH_PIPELINE.run, request.text)
        graph_result: dict = _serialize_graph_result(raw_graph_result)

        entity_graph: dict = graph_result.get("entity_graph", {})
        entity_embeddings: list = []
        if entity_graph:
            try:
                embedding_arr = GRAPH_EMBEDDING_GENERATOR.generate_embedding(entity_graph)
                entity_embeddings = embedding_arr.tolist()
            except Exception as emb_err:
                logger.warning("Entity graph embedding failed: %s", emb_err)

        raw_temporal = _safe_run(
            lambda t: TEMPORAL_GRAPH_ANALYZER.analyze(t).to_dict(), request.text
        )

        graph_analysis: dict = {
            "entity_graph": entity_graph,
            "entity_graph_metrics": graph_result.get("entity_graph_metrics", {}),
            "entity_embeddings": entity_embeddings,
            "narrative_graph": graph_result.get("narrative_graph", {}),
            "narrative_graph_metrics": graph_result.get("narrative_graph_metrics", {}),
            "graph_features": graph_result.get("graph_features", {}),
            "temporal_graph": raw_temporal,
        }

        # ── 10. Explainability ─────────────────────────────────────────────────
        emotion_explanation = _safe_run(explain_emotion, request.text)
        try:
            lime_result = explain_prediction(
                predict_batch,
                request.text,
                num_features=8,
                num_samples=LIME_NUM_SAMPLES,
            )
        except Exception as lime_error:
            logger.warning("LIME explanation unavailable: %s", lime_error)
            lime_result = {
                "text": request.text,
                "important_features": [],
                "error": "lime_unavailable",
            }

        # ── 11. Build response dict ────────────────────────────────────────────
        response_data: dict[str, Any] = {
            "text": _preview_text(request.text),
            "prediction": prediction,
            "fake_probability": round(fake_probability, 4),
            "confidence": round(confidence, 4),
            "bias": {
                "bias_score": round(float(bias_result.bias_score), 4),
                "media_bias": bias_result.media_bias,
                "biased_tokens": bias_result.biased_tokens,
                "sentence_heatmap": bias_result.sentence_heatmap,
            },
            "emotion": {
                "dominant_emotion": emotion_result.dominant_emotion,
                "emotion_scores": emotion_scores,
                "emotion_distribution": emotion_result.emotion_distribution,
            },
            "narrative": {
                "roles": narrative_roles,
                "conflict": narrative_conflict,
                "propagation": narrative_propagation,
                "temporal": narrative_temporal,
            },
            "framing": framing,
            "rhetoric": {
                "rhetorical_devices": rhetorical,
                "argument_structure": argument,
            },
            "discourse": {
                "coherence": discourse_coherence,
                "context_omission": context_omission,
                "information_density": info_density,
                "information_omission": info_omission,
                "source_attribution": source_attribution,
                "ideological_language": ideological,
                "emotion_targets": emotion_target,
            },
            "propaganda_analysis": propaganda_patterns,
            "credibility_profile": credibility_profile,
            "graph_analysis": graph_analysis,
            "explainability": {
                "emotion_explanation": emotion_explanation,
                "lime": lime_result,
            },
        }

        # ── Cache the full analysis result ─────────────────────────────────────
        INFERENCE_CACHE.set(cache_key, response_data)

        # ── Structured inference log ───────────────────────────────────────────
        credibility_score = credibility_profile.get("credibility_score")
        INFERENCE_LOGGER.log_prediction(
            article_id=None,
            start_time=timer_start,
            model_versions={"roberta": SETTINGS.model.name},
            feature_count=len(combined_info) + len(combined_narrative),
            prediction_confidence=round(confidence, 4),
        )

        return AnalysisResponse(**response_data)

    except FileNotFoundError as exc:
        logger.error("Model not found during analysis: %s", exc)
        raise HTTPException(status_code=503, detail="Model not available. Please train the model first.")
    except ValueError as exc:
        logger.error("Invalid analysis input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Analysis error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during analysis")


@app.post("/report", response_model=ReportResponse)
def generate_report(request: NewsRequest):
    """
    Generate a structured analysis report for a news article.

    Runs bias and emotion analysis then packages the results into a
    standardised report using ReportGenerator.  Lighter than /analyze —
    no graph, LIME, or discourse sub-systems are invoked.
    """
    try:
        logger.info("Received /report request (text length: %d)", len(request.text))

        # ── Cache lookup ───────────────────────────────────────────────────────
        cache_key = f"report:{request.text}"
        cached = INFERENCE_CACHE.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for /report")
            return ReportResponse(**cached)

        # ── Lightweight analysis ───────────────────────────────────────────────
        bias_result = compute_bias_features(request.text)
        emotion_result = EMOTION_ANALYZER.analyze(request.text)
        emotion_scores: dict[str, float] = getattr(emotion_result, "emotion_scores", {})

        narrative_roles: dict = _safe_run(NARRATIVE_ROLE_EXTRACTOR.analyze, request.text)
        combined_narrative: dict = {**_safe_run(NARRATIVE_CONFLICT_ANALYZER.analyze, request.text)}

        combined_discourse: dict = {**_safe_run(DISCOURSE_ANALYZER.analyze, request.text)}

        credibility_profile: dict = _safe_run(
            BIAS_PROFILE_BUILDER.build_profile,
            bias_features={"bias_score": float(bias_result.bias_score)},
            emotion_features=emotion_scores,
            narrative_features=combined_narrative,
            discourse_features=combined_discourse,
            ideology_predictions={},
        )

        credibility_score: Optional[float] = credibility_profile.get("credibility_score")

        # ── Generate report via ReportGenerator ────────────────────────────────
        report = REPORT_GENERATOR.generate_report(
            article_text=request.text,
            title=None,
            source=None,
            bias_analysis={
                "bias_score": round(float(bias_result.bias_score), 4),
                "media_bias": bias_result.media_bias,
                "biased_tokens": bias_result.biased_tokens,
            },
            emotion_analysis={
                "dominant_emotion": emotion_result.dominant_emotion,
                "emotion_scores": emotion_scores,
            },
            narrative_structure=narrative_roles,
            entity_graph={},
            credibility_score=credibility_score,
        )

        response_data = {
            "article_summary": report.get("article_summary", {}),
            "bias_analysis": report.get("bias_analysis", {}),
            "emotion_analysis": report.get("emotion_analysis", {}),
            "narrative_structure": report.get("narrative_structure", {}),
            "entity_graph": report.get("entity_graph", {}),
            "credibility_score": report.get("credibility_score"),
        }

        INFERENCE_CACHE.set(cache_key, response_data)

        return ReportResponse(**response_data)

    except ValueError as exc:
        logger.error("Invalid report input: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Report generation error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during report generation")


@app.get("/inference/model-info", response_model=ModelInfoResponse)
def inference_model_info():
    """
    Return metadata about the InferenceEngine's loaded model.

    If the model directory does not yet exist (i.e. the model has not been
    trained), returns available=false with a descriptive message.
    """
    engine = _get_inference_engine()
    if engine is None:
        return ModelInfoResponse(
            available=False,
            model_path=str(MODEL_PATH),
            device=None,
            num_parameters=None,
            num_trainable_parameters=None,
            label_map=None,
        )

    try:
        info = engine.get_model_info()
        label_map = (
            {str(k): v for k, v in engine.label_map.items()}
            if engine.label_map
            else None
        )
        return ModelInfoResponse(
            available=True,
            model_path=info["model_path"],
            device=info["device"],
            num_parameters=info["num_parameters"],
            num_trainable_parameters=info["num_trainable_parameters"],
            label_map=label_map,
        )
    except Exception as exc:
        logger.error("Failed to retrieve model info: %s", exc)
        raise HTTPException(status_code=500, detail="Could not retrieve model information")


@app.post("/cache/clear")
def clear_inference_cache():
    """
    Clear all cached inference results.

    Useful after retraining the model to ensure stale predictions are
    not served from cache.
    """
    try:
        INFERENCE_CACHE.clear()
        logger.info("Inference cache cleared via /cache/clear")
        return {"message": "Inference cache cleared successfully"}
    except Exception as exc:
        logger.error("Cache clear failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to clear inference cache")
