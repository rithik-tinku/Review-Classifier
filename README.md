# Review Sentiment Classifier Project

A professional, production-grade, end-to-end Machine Learning and NLP pipeline that classifies product and service reviews into Negative, Neutral, and Positive sentiments. It compares an optimized TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT transformer model and supports rule-driven operational routing.

## 🚀 Key Features
- **Dual Model Pipeline:** TF-IDF + Logistic Regression baseline and fine-tuned DistilBERT transformer.
- **Rule-Driven Business Engine:** Centralized priority (P0-P3) routing, churn propensity warnings, and complaint category assignments (Refund, Late Delivery, Missing Item, App Crash, etc.).
- **Hybrid VADER Routing:** Graceful fallback handling for non-alphabetic inputs (emojis, punctuation, numbers).
- **Interactive Streamlit Dashboard:** Dynamic accuracy, pain points metrics, executive CSV exports, and automatic fallback data loading.
- **Robust Unit Testing:** Validates multilingual inputs, Hinglish, sarcasm, and edge-cases.
- **Automatic Model Downloads:** Pretrained Hugging Face transformer models (like `distilbert-base-uncased`) are downloaded automatically if they are missing locally.
- **Isolated Custom Checkpoint Management:** Keeps custom-trained large weights (e.g., `bert_model.pth`) out of Git history and provides clear error prompts and training recipes if they are missing.

---

## 📂 Project Directory Structure

```text
ReviewClassifier/
├── data/
│   ├── raw/                 # Raw dataset (dataset.csv)
│   └── processed/           # Cleansed dataset (dataset_cleaned.csv)
├── models/                  # [Ignored from Git] Checkpoints directory
│   ├── baseline/            # TF-IDF + Logistic Regression checkpoints
│   └── distilbert/          # Fine-tuned DistilBERT checkpoints
├── src/
│   ├── business_rules.py    # Priority, churn, and category mappings
│   ├── download_dataset.py  # Yelp reviews downloader
│   ├── preprocessing.py     # Regex-based text cleaner
│   ├── train_baseline.py    # Baseline training script
│   ├── train_transformer.py # DistilBERT fine-tuning script
│   └── evaluate.py          # Model latency & memory comparison reporter
├── tests/
│   └── test_predictor.py    # Automated unit tests
├── plots/                   # Saved confusion matrices
├── reports/                 # Performance reports
├── app.py                   # Streamlit dashboard
├── predictor.py             # Inference router
├── requirements.txt         # Pinned python dependencies
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Virtual Environment Setup
It is highly recommended to use a clean virtual environment to prevent package version conflicts:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Requirements Installation
Install all required dependencies:
```bash
pip install -r requirements.txt
```

---

## 🤖 Model Management

### Automatic Pretrained Downloads
This project refactors the Hugging Face transformer loading so that if the fine-tuned local `models/distilbert` checkpoints are missing, the pipeline will **automatically download and load** `distilbert-base-uncased` from Hugging Face hub.

### Custom Trained Checkpoint (`bert_model.pth`)
The legacy custom checkpoint `bert_model.pth` is a large file and is **not** stored in Git tracking. 
- If a script or function tries to load using the `"bert"` model type and `bert_model.pth` is missing from the root directory, it will throw a clear `FileNotFoundError` explaining that this is a custom checkpoint.
- To generate the `bert_model.pth` custom checkpoint locally, run the training pipeline (see below). Alternatively, obtain the custom checkpoint from your team's designated model registry and place it in the root of the project directory.

---

## 🏃 Run Instructions

### 1. Execute Training Pipelines
To generate datasets and train models (which will save weights to ignored `models/` directory or `bert_model.pth`):
```bash
# 1. Download dataset (cached locally)
python src/download_dataset.py

# 2. Preprocess text
python src/preprocessing.py

# 3. Train models
python src/train_baseline.py
python src/train_transformer.py
```

### 2. Run Evaluation
To compare performance, latency, and memory footprints between models:
```bash
python src/evaluate.py
```

### 3. Run Inference
To run sentiment prediction router:
```bash
python predictor.py
```

### 4. Run Unit Tests
To run the test suite:
```bash
python -m pytest tests/
# or
python tests/test_predictor.py
```

### 5. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 📸 Screenshots
*(Place application and dashboard screenshots here to showcase UI)*

---

## 🔮 Future Improvements
- **Dockerization:** Containerize the pipeline and dashboard for standardized deployments.
- **Model Registry Integration:** Integrate MLflow or W&B to track checkpoint versions and artifacts.
- **API Endpoint:** Expose predictions via a FastAPI backend service.

---

## 🛡️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
