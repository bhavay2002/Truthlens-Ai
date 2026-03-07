# TruthLens AI - Quick Reference Guide

## 🚀 Quick Commands

### Setup
```bash
# Initial setup
python setup.py

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
```

### Training
```bash
# Train the model
python main.py

# This will:
# - Load data from data/raw/
# - Clean and split data (70/15/15)
# - Train RoBERTa model
# - Evaluate on test set
# - Save model to models/roberta_model/
```

### Evaluation
```bash
# Evaluate saved model on test set
python evaluate.py

# Results saved to:
# - reports/evaluation_results.json
# - reports/confusion_matrix.png
```

### API
```bash
# Start API server
uvicorn api.app:app --reload

# Custom host/port
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Production mode (no reload)
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_smoke.py::TestAPI -v
```

### Docker
```bash
# Build image
docker build -t truthlens-ai .

# Run container
docker run -p 8000:8000 truthlens-ai

# Using Docker Compose
docker-compose up --build

# Stop containers
docker-compose down
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .

# Type checking (if using mypy)
mypy src/
```

---

## 📡 API Usage Examples

### Python Requests
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "text": "Breaking news: Scientists discover new species in Amazon."
}

response = requests.post(url, json=data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Fake Probability: {result['fake_probability']:.4f}")
```

### cURL
```bash
# Predict
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here"}'

# Health check
curl http://localhost:8000/health

# Home
curl http://localhost:8000/
```

### JavaScript (Fetch)
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Your news article text here'
  })
});

const result = await response.json();
console.log('Prediction:', result.prediction);
console.log('Confidence:', result.confidence);
```

---

## 📊 Project Structure Quick Reference

```
Truthlens Ai/
├── main.py              # Main training pipeline ⭐
├── evaluate.py          # Evaluation script ⭐
├── setup.py             # Setup automation ⭐
├── api/
│   └── app.py          # FastAPI application ⭐
├── src/
│   ├── data/           # Data processing
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   └── validate_data.py
│   ├── models/         # Model training & prediction
│   │   ├── train_roberta.py  ⭐
│   │   └── predict.py        ⭐
│   ├── features/       # Feature engineering
│   ├── evaluation/     # Metrics & evaluation
│   └── explainability/ # SHAP & LIME
└── tests/              # Unit tests

⭐ = Main files to interact with
```

---

## 🔧 Common Tasks

### Add New Data Source
1. Place CSV in `data/raw/`
2. Update `load_data.py` if needed
3. Run validation: `python -c "from src.data.validate_data import validate_dataset; validate_dataset('data/raw/yourfile.csv')"`
4. Retrain: `python main.py`

### Change Model Hyperparameters
Edit `config/config.yaml`:
```yaml
training:
  epochs: 3  # Increase epochs
  batch_size: 16  # Increase batch size
```

### Deploy to Production
```bash
# Option 1: Docker
docker-compose up -d

# Option 2: Direct
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4

# Option 3: With Gunicorn
gunicorn api.app:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Monitor API
```bash
# Check logs
tail -f logs/app.log

# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
```

### Troubleshooting
```bash
# Check model exists
ls -la models/roberta_model/

# Check data files
ls -la data/raw/

# View logs
tail -n 100 logs/training.log

# Test import
python -c "from src.models.predict import predict; print('OK')"

# Run single test
pytest tests/test_smoke.py::test_project_structure_exists -v
```

---

## 📈 Performance Metrics

After training, check:
- `logs/training.log` - Training progress
- `reports/evaluation_results.json` - Metrics
- `reports/confusion_matrix.png` - Visual results

Key metrics:
- **Accuracy**: Overall correctness
- **Precision**: How many predicted fakes are actually fake
- **Recall**: How many actual fakes were caught
- **F1 Score**: Balance between precision and recall
- **ROC-AUC**: Model's discrimination ability

---

## 🎯 Environment Variables

Edit `.env` file:
```bash
MODEL_NAME=roberta-base
BATCH_SIZE=8
EPOCHS=2
MAX_LENGTH=256
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

---

## 🔑 Important Notes

1. **Data files** not included - download separately
2. **Model files** generated after training (2-3 GB)
3. **First run** may take 10-30 minutes (downloads pre-trained model)
4. **GPU recommended** but not required
5. **Memory**: ~8GB RAM minimum

---

## 📞 Getting Help

- Check `README.md` for detailed docs
- Review `PROJECT_REVIEW.md` for architecture
- See `CONTRIBUTING.md` for contribution guide
- Open issue on GitHub for bugs
- Visit http://localhost:8000/docs for API docs

---

## ⚡ Pro Tips

1. Use `--reload` flag only in development
2. Monitor GPU usage: `nvidia-smi` (if available)
3. Use smaller batch size if OOM errors
4. Enable CUDA for faster training if available
5. Cache predictions for repeated queries
6. Use Docker for consistent environments
7. Run tests before committing: `pytest tests/`
8. Format code: `black .` before PR
9. Keep .env out of version control
10. Backup models before retraining

---

**Quick Start**: `python setup.py` → `python main.py` → `uvicorn api.app:app --reload`
