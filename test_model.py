from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from src.utils.settings import load_settings

SETTINGS = load_settings()
MODEL_PATH = SETTINGS.model.path
MAX_LENGTH = SETTINGS.model.max_length

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained(str(MODEL_PATH))
model = RobertaForSequenceClassification.from_pretrained(str(MODEL_PATH))

model.eval()

# Example input (title + article text)
title = "Iranian ambassador warns UK to be 'very careful' about further involvement in war"
article = "The Iranian ambassador in London has warned the UK to be very careful about becoming further involved in the war.Seyed Ali Mousavi told Sunday with Laura Kuenssberg his country would have a right to self-defence if the UK directly joined US-Israeli attacks on Iran. He warned that Iran expected the British government, and others, to be very delicate, very careful in their actions The UK has given permission for the US to use British bases for what ministers describe as defensive strikes on Iranian facilities, but has not taken part in any direct attacks itself.The ambassador said it was good that the UK was not involved with this aggression, adding he believed the British government had learnt lessons from the 2003 invasion of Iraq.Despite the Iranian president's apology to its Gulf neighbours on Saturday, Mousavi made clear Iran would continue to attack US bases if strikes on Iran continued.Days of strikes across the Middle East have caused enormous disruption and damage in many different countries.Mousavi said that if facilities or properties or bases are used against the Iranian nation, they would be considered legitimate targets.In the last few hours, Gulf countries including Qatar and the UAE have been hit by Iran, while the US and Israel have continued their attacks as the war enters a second week."

# Combine exactly like training
input_text = f"{title} {article}"

# Tokenize
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH
)

# Model prediction
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to probabilities
probabilities = torch.softmax(outputs.logits, dim=1)

# Detect label mapping automatically
id2label = getattr(model.config, "id2label", {0: "REAL", 1: "FAKE"})
label2id = {v.lower(): k for k, v in id2label.items()}

fake_idx = label2id.get("fake", 1)
real_idx = label2id.get("real", 0)

fake_prob = probabilities[0][fake_idx].item()
real_prob = probabilities[0][real_idx].item()

print("Real News Probability:", round(real_prob * 100, 2), "%")
print("Fake News Probability:", round(fake_prob * 100, 2), "%")

# Final prediction
prediction = "FAKE" if fake_prob > real_prob else "REAL"
print("Prediction:", prediction)
