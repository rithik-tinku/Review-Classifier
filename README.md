# Review Classifier

Review Classifier is a Streamlit dashboard and inference pipeline for classifying customer reviews into `Positive`, `Neutral`, and `Negative` sentiment labels.

It combines:
- a TF-IDF + Logistic Regression baseline
- a DistilBERT fallback model
- rule-based business routing for issue category, priority, and churn risk

## Features

- CSV upload or sample dataset fallback
- Batch sentiment prediction
- Issue categorization and priority assignment
- Churn-risk signaling
- KPI cards and charts
- Critical-issue ticket queue
- CSV export
- Automated tests for preprocessing, prediction, and business rules

## Project Layout

```text
Review-Classifier/
├── app.py
├── predictor.py
├── src/
│   ├── business_rules.py
│   ├── preprocessing.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── evaluate.py
│   └── download_dataset.py
├── tests/
├── models/
├── plots/
├── reports/
├── docs/
├── dataset.csv
├── requirements.txt
└── pyproject.toml
```

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

If no file is uploaded, the dashboard uses `dataset.csv` from the project root or `data/raw/dataset.csv` if present.

## Run Tests

```bash
python -m pytest tests
```

## Screenshots

Dashboard screenshot:

- [docs/streamlit_dashboard_screenshot.png](/C:/Users/ACER/Desktop/PROJECTS/ReviewClassifier/Review-Classifier/docs/streamlit_dashboard_screenshot.png)

## Notes

- Model artifacts are stored under `models/`.
- The repository keeps generated training scripts and notebooks out of the current runtime path.
- No API keys or secrets are required for local use.

## License

MIT. See [LICENSE](/C:/Users/ACER/Desktop/PROJECTS/ReviewClassifier/Review-Classifier/LICENSE).

