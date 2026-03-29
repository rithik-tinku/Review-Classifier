import torch
from transformers import BertTokenizer, BertForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- LOAD MODELS (ONCE) ---
# Note: Ensure 'bert_model.pth' exists in your working directory.
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Initializing model structure for 5-class sentiment
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=5
)

# Load your custom weights
try:
    model.load_state_dict(torch.load("bert_model.pth", map_location=torch.device("cpu")))
    model.eval()
except FileNotFoundError:
    print("Warning: bert_model.pth not found. Using pre-trained base weights.")

vader = SentimentIntensityAnalyzer()

# Label mapping for the 5-class BERT output
labels = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive"
]

# --- SINGLE PREDICTION ---
def predict_review(review):
    if not review or str(review).strip() == "":
        return "Neutral"

    text = str(review)
    score = vader.polarity_scores(text)["compound"]

    # Fast path: VADER (Rule-based)
    if score >= 0.6:
        return "Positive"
    elif score <= -0.6:
        return "Negative"

    # Slow path: BERT (Transformer-based)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    sentiment = labels[predicted_class]

    # Map 5 classes -> 3 simplified classes for the Dashboard
    if sentiment in ["Very Positive", "Positive"]:
        return "Positive"
    elif sentiment in ["Very Negative", "Negative"]:
        return "Negative"
    else:
        return "Neutral"

# --- BATCH PREDICTION (FAST 🚀) ---
def predict_batch(reviews):
    if not reviews:
        return []

    # Pre-allocate results to maintain order
    results = [None] * len(reviews)
    bert_reviews = []
    bert_indices = []

    # Step 1: VADER fast filtering
    for i, review in enumerate(reviews):
        text = str(review)
        score = vader.polarity_scores(text)["compound"]

        if score >= 0.6:
            results[i] = "Positive"
        elif score <= -0.6:
            results[i] = "Negative"
        else:
            # Uncertain/Neutral cases queued for BERT
            bert_reviews.append(text)
            bert_indices.append(i)

    # Step 2: Process uncertain reviews in a single BERT batch
    if bert_reviews:
        inputs = tokenizer(
            bert_reviews,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).tolist()

        for idx, pred_val in zip(bert_indices, preds):
            sentiment = labels[pred_val]

            if sentiment in ["Very Positive", "Positive"]:
                results[idx] = "Positive"
            elif sentiment in ["Very Negative", "Negative"]:
                results[idx] = "Negative"
            else:
                results[idx] = "Neutral"

    # Final pass: Ensure no 'None' values remain (safety)
    return [r if r is not None else "Neutral" for r in results]