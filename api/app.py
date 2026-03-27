from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
from typing import Any
import logging

from src.models.predict import predict, predict_batch
from src.features.bias.bias_lexicon import compute_bias_features
from src.features.emotion.emotion_lexicon import EmotionLexiconAnalyzer
from src.explainability.emotion_explainer import explain_emotion
from src.explainability.lime_explainer import explain_prediction
from src.utils.logging_utils import configure_logging
from src.utils.settings import load_settings

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
INFERENCE_ALLOW_RAW_TEXT_FALLBACK = bool(
    SETTINGS.inference.allow_raw_text_fallback
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SUBPACKAGES = (
    "emotion",
    "encoder",
    "ideology",
    "multitask",
    "narrative",
    "propaganda",
)
LIME_NUM_SAMPLES = 256
EMOTION_ANALYZER = EmotionLexiconAnalyzer()

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)


class NewsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Breaking news: Scientists discover new species in Amazon rainforest."
            }
        }
    )
    text: str = Field(..., min_length=10, max_length=10000, description="News article text to analyze")


class NewsResponse(BaseModel):
    text: str
    fake_probability: float = Field(..., ge=0, le=1, description="Probability of being fake news (0-1)")
    prediction: str
    confidence: float


class AnalysisResponse(BaseModel):
    text: str
    prediction: str
    fake_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    bias: dict[str, Any]
    emotion: dict[str, Any]
    explainability: dict[str, Any]


def _preview_text(text: str) -> str:
    if len(text) <= TEXT_PREVIEW_CHARS:
        return text
    return text[:TEXT_PREVIEW_CHARS] + "..."


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


@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": APP_TITLE,
        "status": "online",
        "endpoints": {
            "predict": "/predict",
            "analyze": "/analyze",
            "health": "/health",
            "project_view": "/project-view",
            "docs": "/docs"
        }
    }


@app.get("/project-view")
def project_view():
    """Project-level view of API metadata, configuration, and package layout."""
    return _build_project_view()


@app.post("/predict", response_model=NewsResponse)
def predict_news(request: NewsRequest):
    """
    Predict whether news article is fake or real
    
    Args:
        request: NewsRequest with text field
    
    Returns:
        NewsResponse with prediction results
    """
    try:
        logger.info(f"Received prediction request for text of length: {len(request.text)}")
        
        # Get prediction
        prediction_result = predict(request.text)

        # Backward-compatible handling if predict() returns float or dict
        if isinstance(prediction_result, dict):
            prob = float(prediction_result.get("fake_probability", 0.0))
            prediction = str(prediction_result.get("label", "Fake")).upper()
            confidence = float(prediction_result.get("confidence", max(prob, 1 - prob)))
        else:
            prob = float(prediction_result)
            prediction = "FAKE" if prob > 0.5 else "REAL"
            confidence = prob if prob > 0.5 else (1 - prob)
        
        response = NewsResponse(
            text=_preview_text(request.text),
            fake_probability=round(prob, 4),
            prediction=prediction,
            confidence=round(confidence, 4)
        )
        
        logger.info(f"Prediction: {prediction} with confidence: {confidence:.4f}")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


@app.get("/health")
def health_check():
    """Detailed health check"""
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
        
        # Check for required model files
        required_files = ["config.json", "tokenizer.json"]
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        has_weight_file = any((MODEL_PATH / f).exists() for f in weight_files) if model_exists else False
        model_files_exist = (
            all((MODEL_PATH / f).exists() for f in required_files) and has_weight_file
            if model_exists
            else False
        )
        
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
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.post("/analyze", response_model=AnalysisResponse)
def analyze_news(request: NewsRequest):
    """
    Unified analysis endpoint.

    Returns model prediction + bias/emotion signals + lightweight explainability.
    """

    try:
        prediction_result = predict(request.text)

        if isinstance(prediction_result, dict):
            fake_probability = float(prediction_result.get("fake_probability", 0.0))
            prediction = str(prediction_result.get("label", "Fake")).upper()
            confidence = float(prediction_result.get("confidence", max(fake_probability, 1 - fake_probability)))
        else:
            fake_probability = float(prediction_result)
            prediction = "FAKE" if fake_probability > 0.5 else "REAL"
            confidence = fake_probability if fake_probability > 0.5 else (1 - fake_probability)

        bias_result = compute_bias_features(request.text)
        emotion_result = EMOTION_ANALYZER.analyze(request.text)
        emotion_explanation = explain_emotion(request.text)

        # LIME can fail if model assets are unavailable; keep analysis resilient.
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
                "error": "lime_unavailable"
            }

        return AnalysisResponse(
            text=_preview_text(request.text),
            prediction=prediction,
            fake_probability=round(fake_probability, 4),
            confidence=round(confidence, 4),
            bias={
                "bias_score": round(float(bias_result.bias_score), 4),
                "media_bias": bias_result.media_bias,
                "biased_tokens": bias_result.biased_tokens,
                "sentence_heatmap": bias_result.sentence_heatmap,
            },
            emotion={
                "dominant_emotion": emotion_result.dominant_emotion,
                "emotion_scores": emotion_result.emotion_scores,
                "emotion_distribution": emotion_result.emotion_distribution,
            },
            explainability={
                "emotion_explanation": emotion_explanation,
                "lime": lime_result,
            },
        )

    except FileNotFoundError as e:
        logger.error("Model not found during analysis: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first.",
        )
    except ValueError as e:
        logger.error("Invalid analysis input: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Analysis error: %s", e)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during analysis",
        )
