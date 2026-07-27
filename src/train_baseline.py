import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train_baseline(data_path, out_dir):
    print("Initiating traditional ML baseline training (TF-IDF + Logistic Regression)...")
    
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # 80-20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["Cleaned_Review"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Logistic Regression Classifier
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # Evaluation
    predictions = model.predict(X_test_vec)
    acc = accuracy_score(y_test, predictions)
    print(f"\nBaseline Model Test Accuracy: {acc * 100:.2f}%")
    
    report = classification_report(
        y_test, 
        predictions, 
        target_names=["Negative", "Neutral", "Positive"]
    )
    print("\nBaseline Model Classification Report:")
    print(report)
    
    # Save checkpoints
    model_path = os.path.join(out_dir, "logistic_model.joblib")
    vec_path = os.path.join(out_dir, "tfidf_vectorizer.joblib")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    print(f"Baseline artifacts successfully saved to {out_dir}!")
    
    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, predictions)
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
    plt.title("Baseline Model Confusion Matrix")
    
    # Ensure plots folder exists
    plot_path = "plots/baseline_confusion_matrix.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")

if __name__ == "__main__":
    train_baseline("data/processed/dataset_cleaned.csv", "models/baseline")
