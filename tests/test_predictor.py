import os
import sys
# Add workspace root to python path to import predictor cleanly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import predictor
from src.business_rules import analyze_business_rules

def test_single_prediction():
    # Test positive, negative, and neutral reviews return valid labels
    assert predictor.predict_review("This is a fantastic product!") in ["Positive", "Neutral", "Negative"]
    assert predictor.predict_review("Awful service and poor quality.") in ["Positive", "Neutral", "Negative"]
    assert predictor.predict_review("The package arrived on time.") in ["Positive", "Neutral", "Negative"]

def test_batch_prediction():
    reviews = [
        "Love this, highly recommend it.",
        "Worst purchase of my life.",
        "Average experience, nothing special."
    ]
    preds = predictor.predict_batch(reviews)
    assert len(preds) == 3
    for p in preds:
        assert p in ["Positive", "Neutral", "Negative"]

def test_empty_string():
    # Empty string should fallback to Neutral
    assert predictor.predict_review("") == "Neutral"
    assert predictor.predict_review("   ") == "Neutral"

def test_emoji_only():
    # Emojis should evaluate to correct sentiments or default to neutral gracefully
    assert predictor.predict_review("😊😍👍") in ["Positive", "Neutral"]
    assert predictor.predict_review("😭😡👎") in ["Negative", "Neutral"]

def test_numbers_only():
    assert predictor.predict_review("1234567890") == "Neutral"

def test_long_review():
    long_txt = "Great product! " * 50
    assert predictor.predict_review(long_txt) == "Positive"

def test_short_review():
    assert predictor.predict_review("Ok") == "Neutral"

def test_mixed_sentiment():
    assert predictor.predict_review("Good but late.") in ["Positive", "Neutral", "Negative"]

def test_hindi_hinglish_sarcasm():
    # Test multilingual and sarcasm inputs return strings gracefully without crashes
    assert isinstance(predictor.predict_review("bahut achha product hai"), str)
    assert isinstance(predictor.predict_review("bekaar delivery service"), str)
    assert isinstance(predictor.predict_review("Oh great, another broken screen."), str)

def test_business_rules_validation():
    # Example 1: "Worst purchase ever." -> Negative, Refund, High Churn, P1
    cat, pri, churn = analyze_business_rules("Worst purchase ever.", "Negative")
    assert cat == "Refund"
    assert pri == "P1 - HIGH"
    assert churn == "High"

    # Example 2: "Received broken item." -> Negative, Damaged Product, High Churn, P0
    cat, pri, churn = analyze_business_rules("Received broken item.", "Negative")
    assert cat == "Damaged Product"
    assert pri == "P0 - CRITICAL"
    assert churn == "High"

    # Example 3: "Delivery delayed by 5 days." -> Negative, Late Delivery, P2
    cat, pri, churn = analyze_business_rules("Delivery delayed by 5 days.", "Negative")
    assert cat == "Late Delivery"
    assert pri == "P2 - MEDIUM"

    # Example 4: "Excellent quality." -> Positive, P3
    cat, pri, churn = analyze_business_rules("Excellent quality.", "Positive")
    assert pri == "P3 - LOW"

if __name__ == "__main__":
    test_single_prediction()
    test_batch_prediction()
    test_empty_string()
    test_emoji_only()
    test_numbers_only()
    test_long_review()
    test_short_review()
    test_mixed_sentiment()
    test_hindi_hinglish_sarcasm()
    test_business_rules_validation()
    print("All unit tests and business rules validation passed successfully!")
