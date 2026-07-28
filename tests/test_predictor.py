"""Tests for the prediction pipeline and business rules engine."""

import os
import sys

import pytest

# Ensure the project root is on the path so imports work when running tests directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import predictor
from src.business_rules import analyze_business_rules
from src.preprocessing import preprocess_text

DATASET_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "dataset.csv"),
    os.path.join(os.path.dirname(__file__), "..", "dataset.csv"),
]


def _resolve_dataset_path() -> str:
    for path in DATASET_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No dataset.csv found in data/raw or project root")


# ---------------------------------------------------------------------------
# Preprocessing Tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_basic_cleaning(self):
        assert preprocess_text("HELLO WORLD") == "hello world"

    def test_html_removal(self):
        assert "br" not in preprocess_text("Great product<br>Really good")

    def test_url_removal(self):
        cleaned = preprocess_text("Check https://example.com for details")
        assert "https" not in cleaned
        assert "example" not in cleaned

    def test_whitespace_normalization(self):
        assert preprocess_text("too   many    spaces") == "too many spaces"

    def test_none_input(self):
        assert preprocess_text(None) == ""

    def test_empty_string(self):
        assert preprocess_text("") == ""

    def test_numeric_string(self):
        assert preprocess_text("12345") == "12345"


# ---------------------------------------------------------------------------
# Prediction Tests
# ---------------------------------------------------------------------------

class TestPrediction:
    def test_positive_review(self):
        result = predictor.predict_review("This is a fantastic product!")
        assert result in ["Positive", "Neutral", "Negative"]

    def test_negative_review(self):
        result = predictor.predict_review("Awful service and poor quality.")
        assert result in ["Positive", "Neutral", "Negative"]

    def test_neutral_review(self):
        result = predictor.predict_review("The package arrived on time.")
        assert result in ["Positive", "Neutral", "Negative"]

    def test_empty_string_returns_neutral(self):
        assert predictor.predict_review("") == "Neutral"

    def test_whitespace_only_returns_neutral(self):
        assert predictor.predict_review("   ") == "Neutral"

    def test_none_returns_neutral(self):
        assert predictor.predict_review(None) == "Neutral"

    def test_emoji_only(self):
        assert predictor.predict_review("😊😍👍") in ["Positive", "Neutral"]
        assert predictor.predict_review("😭😡👎") in ["Negative", "Neutral"]

    def test_numbers_only(self):
        assert predictor.predict_review("1234567890") == "Neutral"

    def test_long_review(self):
        long_text = "Great product! " * 50
        assert predictor.predict_review(long_text) == "Positive"

    def test_short_review(self):
        assert predictor.predict_review("Ok") in ["Positive", "Neutral", "Negative"]

    def test_mixed_sentiment(self):
        result = predictor.predict_review("Good but late.")
        assert result in ["Positive", "Neutral", "Negative"]

    def test_multilingual_no_crash(self):
        assert isinstance(predictor.predict_review("bahut achha product hai"), str)
        assert isinstance(predictor.predict_review("bekaar delivery service"), str)

    def test_sarcasm_no_crash(self):
        assert isinstance(predictor.predict_review("Oh great, another broken screen."), str)


# ---------------------------------------------------------------------------
# Batch Prediction Tests
# ---------------------------------------------------------------------------

class TestBatchPrediction:
    def test_basic_batch(self):
        reviews = [
            "Love this, highly recommend it.",
            "Worst purchase of my life.",
            "Average experience, nothing special.",
        ]
        preds = predictor.predict_batch(reviews)
        assert len(preds) == 3
        for p in preds:
            assert p in ["Positive", "Neutral", "Negative"]

    def test_empty_batch(self):
        assert predictor.predict_batch([]) == []

    def test_single_item_batch(self):
        preds = predictor.predict_batch(["Great product"])
        assert len(preds) == 1

    def test_batch_with_edge_cases(self):
        reviews = ["", "   ", "12345", "😊", "Normal review text"]
        preds = predictor.predict_batch(reviews)
        assert len(preds) == 5


# ---------------------------------------------------------------------------
# Business Rules Tests
# ---------------------------------------------------------------------------

class TestBusinessRules:
    def test_damaged_product(self):
        cat, pri, churn = analyze_business_rules("Received broken item.", "Negative")
        assert cat == "Damaged Product"
        assert pri == "P0 - CRITICAL"
        assert churn == "High"

    def test_worst_purchase_override(self):
        cat, pri, churn = analyze_business_rules("Worst purchase ever.", "Negative")
        assert cat == "Refund"
        assert pri == "P1 - HIGH"
        assert churn == "High"

    def test_late_delivery(self):
        cat, pri, churn = analyze_business_rules("Delivery delayed by 5 days.", "Negative")
        assert cat == "Late Delivery"
        assert pri == "P2 - MEDIUM"

    def test_positive_review_downgrade(self):
        cat, pri, churn = analyze_business_rules("Excellent quality.", "Positive")
        assert pri == "P3 - LOW"
        assert churn == "Low"

    def test_positive_with_safety_stays_p0(self):
        cat, pri, churn = analyze_business_rules("Great but the screen was broken.", "Positive")
        assert pri == "P0 - CRITICAL"

    def test_competitor_mention_high_churn(self):
        cat, pri, churn = analyze_business_rules("Switching to Amazon.", "Negative")
        assert churn == "High"

    def test_neutral_with_competitor_medium_churn(self):
        cat, pri, churn = analyze_business_rules("Might try Swiggy instead.", "Neutral")
        assert churn == "Medium"

    def test_neutral_no_competitor_low_churn(self):
        cat, pri, churn = analyze_business_rules("It was okay.", "Neutral")
        assert churn == "Low"

    def test_all_categories_in_list(self):
        from src.business_rules import CATEGORIES
        assert len(CATEGORIES) == 21
        assert "Other" in CATEGORIES


# ---------------------------------------------------------------------------
# CSV Loading Edge Cases
# ---------------------------------------------------------------------------

class TestCSVEdgeCases:
    def test_dataset_file_exists(self):
        assert os.path.exists(_resolve_dataset_path())

    def test_dataset_has_expected_columns(self):
        import pandas as pd
        df = pd.read_csv(_resolve_dataset_path(), nrows=5)
        assert "Review" in df.columns
        assert "Rating" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
