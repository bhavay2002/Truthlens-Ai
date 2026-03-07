# 🔍 TruthLens AI - Fake News Detection

An advanced fake news detection system using RoBERTa transformer model with explainable AI capabilities.

## ✨ Features

- **Deep Learning**: RoBERTa-based transformer model for accurate text classification
- **Feature Engineering**: Text analysis, source credibility scoring, and metadata features
- **Explainable AI**: SHAP and LIME integration for model interpretability
- **REST API**: FastAPI-based API for easy integration
- **Production Ready**: Proper error handling, logging, and validation

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "Truthlens Ai"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Download the dataset from [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Place the files in:
- `data/raw/fake.csv`
- `data/raw/real.csv`

### 3. Train Model

```bash
python main.py
```

This will:
- Load and clean the data
- Split into train/validation/test sets (70/15/15)
- Train RoBERTa model
- Evaluate on test set
- Save model to `./models/roberta_model/`

### 4. Run API

```bash
uvicorn api.app:app --reload
```

API will be available at: http://localhost:8000

## 📡 API Usage

### Health Check
```bash
curl http://localhost:8000/
```

### Predict

**Using curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news: Scientists discover new species in Amazon rainforest."}'
```

**Response:**
```json
{
  "text": "Breaking news: Scientists discover...",
  "fake_probability": 0.1234,
  "prediction": "REAL",
  "confidence": 0.8766
}
```

### API Documentation
Interactive docs: http://localhost:8000/docs

## 📊 Project Structure

```
Truthlens Ai/
├── api/                    # FastAPI application
├── config/                 # Configuration files
├── data/                   # Data storage
│   ├── raw/               # Raw datasets
│   ├── processed/         # Processed data
│   └── interim/           # Intermediate data
├── models/                # Saved models
├── src/                   # Source code
│   ├── data/             # Data processing
│   ├── features/         # Feature engineering
│   ├── models/           # Model training & prediction
│   ├── evaluation/       # Model evaluation
│   ├── explainability/   # SHAP & LIME
│   ├── visualization/    # Plotting utilities
│   └── utils/            # Helper functions
├── tests/                # Unit tests
├── logs/                 # Application logs
├── main.py              # Training pipeline
└── requirements.txt     # Python dependencies
```

## 🔧 Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Edit `config/config.yaml` for model hyperparameters.

## 🧪 Testing

```bash
pytest tests/
```

## 📈 Model Performance

The model is evaluated on:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Results are logged during training.

## 🔮 Future Improvements

- [ ] Add real-time training/fine-tuning
- [ ] Implement cross-validation
- [ ] Add more feature engineering (sentiment, entity recognition)
- [ ] Integrate with fact-checking APIs
- [ ] Deploy with Docker
- [ ] Add model versioning (MLflow)
- [ ] Implement A/B testing
- [ ] Add monitoring and alerting

## 📝 License

MIT License (add LICENSE file)

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📧 Contact

For questions or support, please open an issue on GitHub.
