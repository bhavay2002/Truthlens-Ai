import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path to trained model
model_path = "models/roberta_model"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded successfully on", device)


title = "Mixed messages from Trump leave more questions than answers over war's end"

text = """President Donald Trump and his administration have so far offered mixed messages and contradictory explanations on the joint US Israeli military campaign against Iran."""

input_text = title + " " + text


# Tokenize
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

inputs = {k: v.to(device) for k, v in inputs.items()}

# Prediction
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

# Convert to probabilities
probs = F.softmax(logits, dim=1)

label2id = getattr(model.config, "label2id", None) or {}
normalized = {str(k).strip().lower(): int(v) for k, v in label2id.items()}
real_idx = normalized.get("real", 0)
fake_idx = normalized.get("fake", 1)

if real_idx == fake_idx:
    real_idx, fake_idx = 0, 1

fake_prob = probs[0][fake_idx].item()
real_prob = probs[0][real_idx].item()

print("\nFake News Probability:", round(fake_prob * 100, 2), "%")
print("Real News Probability:", round(real_prob * 100, 2), "%")

if fake_prob > real_prob:
    print("\nPrediction: Fake News")
else:
    print("\nPrediction: Real News")
