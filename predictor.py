import os
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load baseline models
try:
    vectorizer = joblib.load("models/baseline/tfidf_vectorizer.joblib")
    base_model = joblib.load("models/baseline/logistic_model.joblib")
    HAS_BASELINE = True
except Exception:
    HAS_BASELINE = False

# Load DistilBERT model
try:
    if os.path.exists("models/distilbert"):
        tokenizer = DistilBertTokenizer.from_pretrained("models/distilbert")
        transformer_model = DistilBertForSequenceClassification.from_pretrained("models/distilbert")
    else:
        # Automatically download pretrained Hugging Face model when missing
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        transformer_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
    transformer_model.eval()
    HAS_TRANSFORMER = True
except Exception:
    HAS_TRANSFORMER = False

vader = SentimentIntensityAnalyzer()
labels = ["Negative", "Neutral", "Positive"]


def predict_review(review, model_type="baseline"):
    """
    Predicts the sentiment of a single review.
    Supported model types: 'baseline' (TF-IDF + LogReg) or 'transformer' (DistilBERT).
    """
    if not review or str(review).strip() == "":
        return "Neutral"
        
    text = str(review)
    
    # Fallback to VADER for non-alphabetic texts (numbers-only, emoji-only, punctuation-only)
    import re
    if not re.search(r'[a-zA-Z]', text):
        score = vader.polarity_scores(text)["compound"]
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    # 1. Baseline logic
    if model_type == "baseline" and HAS_BASELINE:
        text_vec = vectorizer.transform([text])
        pred_idx = base_model.predict(text_vec)[0]
        return labels[pred_idx]
        
    # 2. Transformer logic
    elif model_type == "transformer" and HAS_TRANSFORMER:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformer_model.to(device)
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = transformer_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()
        return labels[pred_idx]
        
    # 2b. Custom BERT logic
    elif model_type == "bert":
        if not os.path.exists("bert_model.pth"):
            raise FileNotFoundError(
                "❌ Custom trained checkpoint 'bert_model.pth' is missing! "
                "This is a custom-trained checkpoint and cannot be downloaded automatically. "
                "Please run train_model.py to train the model and generate this checkpoint, "
                "or place the pre-trained 'bert_model.pth' file in the root directory."
            )
        
        from transformers import BertTokenizer, BertForSequenceClassification
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
        bert_model.load_state_dict(torch.load("bert_model.pth", map_location=device))
        bert_model.to(device)
        bert_model.eval()
        
        inputs = bert_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=1).item()
        
        labels_5 = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        sentiment = labels_5[pred_idx]
        if sentiment in ["Very Positive", "Positive"]:
            return "Positive"
        elif sentiment in ["Very Negative", "Negative"]:
            return "Negative"
        else:
            return "Neutral"
        
    # 3. Fallback to VADER (Rule-based)
    score = vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def predict_batch(reviews, model_type="baseline"):
    """
    Predicts the sentiment of a batch of reviews.
    """
    if not reviews:
        return []
        
    return [predict_review(r, model_type=model_type) for r in reviews]