CATEGORIES = [
    "Refund",
    "Late Delivery",
    "Damaged Product",
    "Missing Item",
    "Poor Packaging",
    "Wrong Item",
    "Payment Failure",
    "Customer Support",
    "Subscription",
    "Pricing",
    "App Crash",
    "Performance",
    "Battery",
    "Warranty",
    "Returns",
    "Shipping",
    "Installation",
    "General Complaint",
    "Fake Product",
    "Account Issues",
    "Other",
]

# Keywords that trigger each issue category (checked in priority order)
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Damaged Product": ["broken", "damaged", "scratched", "smashed", "shattered", "leaking", "torn"],
    "Late Delivery": ["delay", "late", "waiting", "delayed"],
    "Refund": ["refund", "cashback", "money back", "reimbursement"],
    "Missing Item": ["missing", "not received", "incomplete", "forgot"],
    "Poor Packaging": ["packaging", "box torn", "wrap", "packaged"],
    "Wrong Item": ["wrong item", "different product", "wrong color", "incorrect item"],
    "Payment Failure": ["payment", "transaction", "charged twice", "billing", "declined"],
    "Customer Support": ["support", "agent", "customer service", "chat", "helpdesk"],
    "Subscription": ["subscription", "subscribe", "membership", "monthly plan"],
    "Pricing": ["price", "expensive", "costly", "overpriced", "cheap"],
    "App Crash": ["crash", "app freeze", "not opening", "laggy", "bug", "otp"],
    "Performance": ["performance", "speed", "working well", "efficient", "quality"],
    "Battery": ["battery", "charge", "power", "drain"],
    "Warranty": ["warranty", "guarantee", "replace", "repair"],
    "Returns": ["return", "send back"],
    "Shipping": ["shipping", "courier", "postage", "transit"],
    "Installation": ["install", "setup", "assembly", "build"],
    "Fake Product": ["fake", "counterfeit", "copy", "not genuine"],
    "Account Issues": ["account", "login", "password", "sign up"],
    "General Complaint": ["worst", "terrible", "bad", "garbage", "trash", "useless"],
}

P0_CATEGORIES = {"Damaged Product", "Fake Product", "App Crash"}
P1_CATEGORIES = {"Refund", "Payment Failure", "Missing Item", "Account Issues"}
P2_CATEGORIES = {
    "Late Delivery", "Poor Packaging", "Wrong Item",
    "Returns", "Customer Support", "Shipping", "Installation",
}

COMPETITORS = [
    "zepto", "swiggy", "instamart", "dunzo",
    "blinkit", "bigbasket", "amazon", "walmart", "ebay",
]


def analyze_business_rules(
    review_text: str, sentiment: str
) -> tuple[str, str, str]:
    """
    Classify a review into an issue category, priority level, and churn risk.

    Args:
        review_text: Raw review string.
        sentiment: Predicted sentiment label ('Positive', 'Neutral', or 'Negative').

    Returns:
        Tuple of (category, priority, churn_risk).
    """
    text = str(review_text).lower().strip()
    sentiment = str(sentiment)

    # --- Issue Categorization ---
    category = "Other"

    # Special-case override
    if "worst purchase" in text:
        category = "Refund"
    else:
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                category = cat
                break

    # --- Priority Assignment ---
    if category in P0_CATEGORIES or "broken" in text:
        priority = "P0 - CRITICAL"
    elif category in P1_CATEGORIES or "refund" in text:
        priority = "P1 - HIGH"
    elif category in P2_CATEGORIES:
        priority = "P2 - MEDIUM"
    else:
        priority = "P3 - LOW"

    # Positive reviews get downgraded unless it's a safety issue
    if sentiment == "Positive" and priority != "P0 - CRITICAL":
        priority = "P3 - LOW"

    # --- Churn Risk ---
    has_competitor = any(c in text for c in COMPETITORS)

    if sentiment == "Negative" and (has_competitor or priority in ("P0 - CRITICAL", "P1 - HIGH")):
        churn_risk = "High"
    elif sentiment == "Negative" or (sentiment == "Neutral" and has_competitor):
        churn_risk = "Medium"
    else:
        churn_risk = "Low"

    return category, priority, churn_risk
