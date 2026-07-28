import os
import re

import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.preprocessing import preprocess_text

# --- Model Loading (happens once at import time) ---

LABELS = ["Negative", "Neutral", "Positive"]

# Baseline: TF-IDF + Logistic Regression
try:
    _vectorizer = joblib.load("models/baseline/tfidf_vectorizer.joblib")
    _base_model = joblib.load("models/baseline/logistic_model.joblib")
    HAS_BASELINE = True
except Exception:
    HAS_BASELINE = False

# Transformer: DistilBERT (fine-tuned or pretrained fallback)
try:
    if os.path.exists("models/distilbert"):
        _tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert")
        _transformer = DistilBertForSequenceClassification.from_pretrained("models/distilbert")
    else:
        _tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        _transformer = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3
        )
    _transformer.eval()
    HAS_TRANSFORMER = True
except Exception:
    HAS_TRANSFORMER = False

_vader = SentimentIntensityAnalyzer()


def _vader_classify(text: str) -> str:
    """Map VADER compound score to a sentiment label."""
    score = _vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"


def _vader_confidence(text: str, label: str) -> float:
    """Derive a lightweight confidence score from the VADER compound score."""
    score = _vader.polarity_scores(text)["compound"]
    if label == "Positive":
        return round((score + 1.0) / 2.0, 4)
    if label == "Negative":
        return round((1.0 - score) / 2.0, 4)
    return round(1.0 - abs(score), 4)


def predict_review(review: str, model_type: str = "baseline") -> str:
    """
    Predict sentiment for a single review string.

    Args:
        review: The review text to classify.
        model_type: One of 'baseline' (TF-IDF + LogReg), 'transformer' (DistilBERT),
                     or 'vader' (rule-based fallback).

    Returns:
        One of 'Positive', 'Neutral', or 'Negative'.
    """
    if not review or str(review).strip() == "":
        return "Neutral"

    text = str(review)

    # Non-alphabetic inputs (emoji-only, numbers-only) go straight to VADER
    if not re.search(r"[a-zA-Z]", text):
        return _vader_classify(text)

    # Baseline: TF-IDF + Logistic Regression
    if model_type == "baseline" and HAS_BASELINE:
        # The vectorizer was trained on preprocessed text, so we must
        # apply the same preprocessing before inference.
        cleaned = preprocess_text(text)
        text_vec = _vectorizer.transform([cleaned])
        pred_idx = _base_model.predict(text_vec)[0]
        return LABELS[pred_idx]

    # Transformer: DistilBERT
    if model_type == "transformer" and HAS_TRANSFORMER:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _transformer.to(device)

        inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = _transformer(input_ids=input_ids, attention_mask=attention_mask)

        pred_idx = torch.argmax(outputs.logits, dim=1).item()
        return LABELS[pred_idx]

    # Fallback: VADER rule-based sentiment
    return _vader_classify(text)


def predict_batch(reviews: list[str], model_type: str = "baseline") -> list[str]:
    """Predict sentiment for a list of review strings."""
    if not reviews:
        return []
    return [predict_review(r, model_type=model_type) for r in reviews]


def predict_batch_with_confidence(
    reviews: list[str], model_type: str = "baseline"
) -> list[tuple[str, float]]:
    """Predict sentiment and confidence for a list of review strings."""
    if not reviews:
        return []
    return [predict_review_with_confidence(r, model_type=model_type) for r in reviews]


def predict_review_with_confidence(review: str, model_type: str = "baseline") -> tuple[str, float]:
    """Predict sentiment and return a confidence score for the chosen model."""
    if not review or str(review).strip() == "":
        return "Neutral", 0.0

    text = str(review)

    if not re.search(r"[a-zA-Z]", text):
        label = _vader_classify(text)
        return label, _vader_confidence(text, label)

    if model_type == "baseline" and HAS_BASELINE:
        cleaned = preprocess_text(text)
        text_vec = _vectorizer.transform([cleaned])
        label_idx = int(_base_model.predict(text_vec)[0])
        label = LABELS[label_idx]
        if hasattr(_base_model, "predict_proba"):
            confidence = float(max(_base_model.predict_proba(text_vec)[0]))
        else:
            confidence = 0.0
        return label, round(confidence, 4)

    if model_type == "transformer" and HAS_TRANSFORMER:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _transformer.to(device)

        inputs = _tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = _transformer(input_ids=input_ids, attention_mask=attention_mask)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        return LABELS[pred_idx], round(float(torch.max(probs).item()), 4)

    label = _vader_classify(text)
    return label, _vader_confidence(text, label)
