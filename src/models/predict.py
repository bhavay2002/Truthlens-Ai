from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
_tokenizer = None
_model = None


def load_model_and_tokenizer():
    """Lazy load model and tokenizer"""
    global _tokenizer, _model
    
    if _tokenizer is None or _model is None:
        model_path = Path("./models/roberta_model")
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Please train the model first by running: python main.py"
            )
        
        try:
            logger.info("Loading model and tokenizer...")
            _tokenizer = RobertaTokenizer.from_pretrained(str(model_path))
            _model = RobertaForSequenceClassification.from_pretrained(str(model_path))
            _model.eval()  # Set to evaluation mode
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    return _tokenizer, _model


def predict(text: str) -> float:
    """
    Predict fake news probability for given text
    
    Args:
        text: Input text to classify
    
    Returns:
        float: Probability of being fake news (0-1)
    """
    try:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        tokenizer, model = load_model_and_tokenizer()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)
        fake_prob = probs[0][1].item()
        
        return fake_prob
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
