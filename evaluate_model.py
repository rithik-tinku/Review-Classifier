import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from predictor import predict_review

def evaluate_model(dataset_path):
    # --- LOAD DATASET ---
    data = pd.read_csv(dataset_path)

    print("\nDataset Loaded Successfully\n")
    print(data.head())

    # --- DETECT REVIEW COLUMN ---
    review_column = None
    possible_review_columns = [
        "Review",
        "review",
        "reviewText",
        "reviews.text",
        "text",
        "content"
    ]

    for col in possible_review_columns:
        if col in data.columns:
            review_column = col
            break

    if review_column is None:
        raise ValueError("No review column found in dataset.")

    # --- DETECT RATING COLUMN ---
    rating_column = None
    possible_rating_columns = [
        "Rating",
        "rating",
        "overall",
        "score"
    ]

    for col in possible_rating_columns:
        if col in data.columns:
            rating_column = col
            break

    if rating_column is None:
        raise ValueError("No rating column found in dataset.")

    print(f"\nUsing review column: {review_column}")
    print(f"Using rating column: {rating_column}\n")

    # --- CLEAN DATA ---
    data = data.dropna(subset=[review_column, rating_column])
    reviews = data[review_column].astype(str).tolist()

    # --- PREDICT SENTIMENT ---
    print("Running predictions...\n")
    predictions = []

    for review in reviews:
        sentiment = predict_review(review)
        predictions.append(sentiment)

    data["Predicted Sentiment"] = predictions

    # --- MAP LABELS ---
    label_map = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }

    predicted_labels = [label_map.get(p, 1) for p in predictions]
    
    # Ensuring true_labels are integers for comparison
    true_labels = (data[rating_column].astype(int)) - 1

    # --- MODEL ACCURACY ---
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("\nModel Accuracy:")
    print(f"{accuracy:.2f}\n")

    # --- CLASSIFICATION REPORT ---
    print("Classification Report:\n")
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=["Negative", "Neutral", "Positive"]
    )
    print(report)

    # --- CONFUSION MATRIX ---
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Neg", "Neu", "Pos"],
        yticklabels=["Neg", "Neu", "Pos"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # --- SAMPLE PREDICTIONS ---
    print("\nSample Predictions:\n")
    print(data[[review_column, "Predicted Sentiment"]].head(10))

# --- RUN SCRIPT ---
if __name__ == "__main__":
    dataset_path = "dataset.csv"
    evaluate_model(dataset_path)