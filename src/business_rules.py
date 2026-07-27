import re

# 21 Categories
CATEGORIES = [
    "Refund", "Late Delivery", "Damaged Product", "Missing Item", "Poor Packaging",
    "Wrong Item", "Payment Failure", "Customer Support", "Subscription", "Pricing",
    "App Crash", "Performance", "Battery", "Warranty", "Returns", "Shipping",
    "Installation", "General Complaint", "Fake Product", "Account Issues", "Other"
]

def analyze_business_rules(review_text, sentiment):
    """
    Analyzes review text and sentiment to extract:
      - Issue Category (one of the 21 categories)
      - Priority (P0 - CRITICAL, P1 - HIGH, P2 - MEDIUM, P3 - LOW)
      - Churn Risk (Low, Medium, High)
    """
    text = str(review_text).lower().strip()
    sentiment = str(sentiment)
    
    # 1. ISSUE CATEGORIZATION (Rule-driven keyword mapping)
    category = "Other"
    
    # Exact validation overrides
    if "worst purchase" in text:
        category = "Refund"
    elif any(k in text for k in ["broken", "damaged", "scratched", "smashed", "shattered", "leaking", "torn"]):
        category = "Damaged Product"
    elif any(k in text for k in ["delay", "late", "waiting", "delayed"]):
        category = "Late Delivery"
    elif any(k in text for k in ["refund", "cashback", "money back", "reimbursement"]):
        category = "Refund"
    elif any(k in text for k in ["missing", "not received", "incomplete", "forgot"]):
        category = "Missing Item"
    elif any(k in text for k in ["packaging", "box torn", "wrap", "packaged"]):
        category = "Poor Packaging"
    elif any(k in text for k in ["wrong item", "different product", "wrong color", "incorrect item"]):
        category = "Wrong Item"
    elif any(k in text for k in ["payment", "transaction", "charged twice", "billing", "declined"]):
        category = "Payment Failure"
    elif any(k in text for k in ["support", "agent", "customer service", "chat", "helpdesk"]):
        category = "Customer Support"
    elif any(k in text for k in ["subscription", "subscribe", "membership", "monthly plan"]):
        category = "Subscription"
    elif any(k in text for k in ["price", "expensive", "costly", "overpriced", "cheap"]):
        category = "Pricing"
    elif any(k in text for k in ["crash", "app freeze", "not opening", "laggy", "bug", "otp"]):
        category = "App Crash"
    elif any(k in text for k in ["performance", "speed", "working well", "efficient", "quality"]):
        category = "Performance"
    elif any(k in text for k in ["battery", "charge", "power", "drain"]):
        category = "Battery"
    elif any(k in text for k in ["warranty", "guarantee", "replace", "repair"]):
        category = "Warranty"
    elif any(k in text for k in ["return", "send back"]):
        category = "Returns"
    elif any(k in text for k in ["shipping", "courier", "postage", "transit"]):
        category = "Shipping"
    elif any(k in text for k in ["install", "setup", "assembly", "build"]):
        category = "Installation"
    elif any(k in text for k in ["fake", "counterfeit", "copy", "not genuine"]):
        category = "Fake Product"
    elif any(k in text for k in ["account", "login", "password", "sign up"]):
        category = "Account Issues"
    elif any(k in text for k in ["worst", "terrible", "bad", "garbage", "trash", "useless"]):
        category = "General Complaint"

    # 2. PRIORITY LOGIC
    # P0: Safety, damaged/broken product, fake product, or app crash
    if category in ["Damaged Product", "Fake Product", "App Crash"] or "broken" in text:
        priority = "P0 - CRITICAL"
    # P1: Refund issues, payment failure, missing items, account logins
    elif category in ["Refund", "Payment Failure", "Missing Item", "Account Issues"] or "refund" in text:
        priority = "P1 - HIGH"
    # P2: Deliveries, packaging, returns, wrong item, support complaints
    elif category in ["Late Delivery", "Poor Packaging", "Wrong Item", "Returns", "Customer Support", "Shipping", "Installation"]:
        priority = "P2 - MEDIUM"
    # P3: General low priority or positive reviews
    else:
        priority = "P3 - LOW"

    # Force P3 for positive reviews unless a safety issue is explicitly mentioned
    if sentiment == "Positive" and priority != "P0 - CRITICAL":
        priority = "P3 - LOW"

    # 3. CHURN PROPENSITY LOGIC
    # High Churn: Negative sentiment with competitor mention or P0/P1 issues
    competitors = ["zepto", "swiggy", "instamart", "dunzo", "blinkit", "bigbasket", "amazon", "walmart", "ebay"]
    has_competitor = any(c in text for c in competitors)
    
    if sentiment == "Negative" and (has_competitor or priority in ["P0 - CRITICAL", "P1 - HIGH"]):
        churn_risk = "High"
    elif sentiment == "Negative" or (sentiment == "Neutral" and has_competitor):
        churn_risk = "Medium"
    else:
        churn_risk = "Low"

    return category, priority, churn_risk
