from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from src.models.predict import predict
from src.utils.logging_utils import configure_logging
from src.utils.settings import load_settings
import logging

configure_logging()
logger = logging.getLogger(__name__)
SETTINGS = load_settings()
MODEL_PATH = SETTINGS.model.path

app = FastAPI(
    title="TruthLens AI - Fake News Detection API",
    description="Detect fake news using RoBERTa-based NLP model",
    version="1.0.0"
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


@app.get("/")
def home():
    """Health check endpoint"""
    return {
        "message": "TruthLens AI - Fake News Detection API",
        "status": "online",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }


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
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
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
            "status": "healthy" if model_exists and model_files_exist else "degraded",
            "model_path": str(MODEL_PATH),
            "model_exists": model_exists,
            "model_files_complete": model_files_exist
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "error_type": type(e).__name__
        }
