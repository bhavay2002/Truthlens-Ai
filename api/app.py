from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.models.predict import predict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TruthLens AI - Fake News Detection API",
    description="Detect fake news using RoBERTa-based NLP model",
    version="1.0.0"
)


class NewsRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000, description="News article text to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Breaking news: Scientists discover new species in Amazon rainforest."
            }
        }


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
        prob = predict(request.text)
        
        # Determine prediction label
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
        from pathlib import Path
        model_exists = Path("./models/roberta_model").exists()
        
        return {
            "status": "healthy",
            "model_loaded": model_exists
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
