import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

from predictor import predict_batch_with_confidence
from src.business_rules import analyze_business_rules

# --- Page Configuration ---
st.set_page_config(
    page_title="Review Classifier — Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Quick-Commerce Reviews Dashboard")
st.caption("Decision-grade insights for Product, Operations, and CX teams")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    batch_size = st.slider("Batch Size", min_value=16, max_value=256, value=64, step=16)
    st.divider()
    st.markdown(
        "**How it works:** Upload a CSV with a review column "
        "(named `Review`, `Text`, `Content`, or `Feedback`) "
        "and the model will classify each review as Positive, Neutral, or Negative."
    )

# --- Data Loading ---
uploaded_file = st.file_uploader("Upload Customer Reviews (CSV)", type=["csv"])
sample_paths = [Path("data/raw/dataset.csv"), Path("dataset.csv")]

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.info("💡 No file uploaded — showing results from the built-in sample dataset.")
    for sample_path in sample_paths:
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            break
    else:
        st.error(
            "Sample dataset not found at `data/raw/dataset.csv` or `dataset.csv`. "
            "Please upload a CSV file or run `python src/download_dataset.py`."
        )
        st.stop()

# Detect the review text column
REVIEW_COL_NAMES = ["review", "text", "content", "feedback"]
review_col = next((c for c in df.columns if c.lower() in REVIEW_COL_NAMES), None)

if not review_col:
    st.error(
        "Could not find a review column. "
        "Expected one of: `Review`, `Text`, `Content`, or `Feedback`."
    )
    st.stop()

reviews = df[review_col].astype(str).tolist()

# --- Batch Prediction with Progress ---
progress_bar = st.progress(0)
status_text = st.empty()
predictions = []
confidences = []
total = len(reviews)

if total == 0:
    st.warning("The selected dataset does not contain any reviews.")
    st.stop()
else:
    for i in range(0, total, batch_size):
        batch = reviews[i : i + batch_size]
        batch_results = predict_batch_with_confidence(batch)
        predictions.extend([label for label, _ in batch_results])
        confidences.extend([confidence for _, confidence in batch_results])
        progress = min((i + batch_size) / total, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Analyzed {min(i + batch_size, total)} / {total} reviews")

    progress_bar.empty()
    status_text.success(f"✅ Analysis complete — {total} reviews processed.")

# --- Enrich Data ---
df["Sentiment"] = predictions
df["Confidence"] = confidences

results = [
    analyze_business_rules(row[review_col], row["Sentiment"]) for _, row in df.iterrows()
]
df["Issue Category"] = [r[0] for r in results]
df["Priority"] = [r[1] for r in results]
df["Churn Risk"] = [r[2] for r in results]

# --- KPI Cards ---
st.divider()

rating_col = next((c for c in df.columns if c.lower() in ["rating", "score", "overall"]), None)
accuracy_text = "N/A"
accuracy_delta = "No ground truth"

if rating_col:
    try:
        rating_to_label = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
        sentiment_to_label = {"Negative": 0, "Neutral": 1, "Positive": 2}

        valid = df.dropna(subset=[rating_col, "Sentiment"])
        y_true = [rating_to_label.get(int(r), 1) for r in valid[rating_col]]
        y_pred = [sentiment_to_label.get(s, 1) for s in valid["Sentiment"]]

        if y_true:
            acc = accuracy_score(y_true, y_pred)
            accuracy_text = f"{acc * 100:.1f}%"
            accuracy_delta = f"Across {len(y_true)} reviews"
    except Exception:
        accuracy_text = "Error"
        accuracy_delta = "Calculation failed"

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Model Accuracy", accuracy_text, delta=accuracy_delta)
kpi2.metric("P0 — Critical Issues", int((df["Priority"] == "P0 - CRITICAL").sum()))
kpi3.metric(
    "Avg. Confidence",
    f"{df['Confidence'].mean() * 100:.1f}%",
    delta="Model certainty",
)

# --- Charts ---
st.divider()

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Sentiment Distribution")
    sentiment_counts = df["Sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

with chart_right:
    st.subheader("Top Customer Pain Points")
    neg_df = df[df["Sentiment"] == "Negative"]
    if not neg_df.empty:
        pain_points = neg_df["Issue Category"].value_counts().head(10)
        st.bar_chart(pain_points)
    else:
        st.info("No negative reviews detected.")

# --- Intelligence Feed ---
st.divider()

with st.expander("📋 Full Intelligence Feed", expanded=False):
    display_cols = [review_col, "Sentiment", "Confidence", "Priority", "Churn Risk", "Issue Category"]
    view_df = df[display_cols].copy()
    view_df.insert(0, "ID", range(1, len(view_df) + 1))
    st.dataframe(view_df, use_container_width=True, hide_index=True)

# --- Critical Priority Action Queue ---
st.divider()
st.subheader("🚨 Critical Priority Action Queue")

critical_cases = df[df["Priority"] == "P0 - CRITICAL"].head(5)

if critical_cases.empty:
    st.success("No critical (P0) issues detected.")
else:
    ACTION_MAP = {
        "Refund": ("Fix refund SLA & automate payouts", "Finance & Tech", "+15% CSAT"),
        "Late Delivery": ("Optimize dark-store routing & ETA accuracy", "Logistics", "+18% CSAT"),
        "Missing Item": ("Audit warehouse picking & scanning flow", "Warehouse Ops", "+12% CSAT"),
        "Customer Support": ("Reduce agent response time & training", "CX Strategy", "+10% CSAT"),
        "Damaged Product": ("Immediate cold-chain audit & vendor QC", "Quality Control", "+25% CSAT"),
    }
    DEFAULT_ACTION = ("Perform root-cause operational analysis", "Operations Team", "+8% CSAT")

    for i, (_, row) in enumerate(critical_cases.iterrows(), 1):
        review_text = row[review_col]
        category = row.get("Issue Category", "Other")
        action, team, impact = ACTION_MAP.get(category, DEFAULT_ACTION)

        st.error(f"**P0 ALERT — Ticket #{i} — {category}**")
        st.markdown(f"> *\"{review_text}\"*")

        col_a, col_b, col_c = st.columns(3)
        col_a.info(f"📍 **Action:** {action}")
        col_b.write(f"💼 **Owner:** {team}")
        col_c.write(f"📈 **Target:** {impact}")

        btn_key = f"ticket_btn_{i}"
        state_key = f"assigned_{i}"

        if state_key not in st.session_state:
            st.session_state[state_key] = False

        if not st.session_state[state_key]:
            if st.button(f"Assign Ticket #{i}", key=btn_key):
                st.session_state[state_key] = True
                st.rerun()
        else:
            st.success(f"Ticket #{i} assigned to {team} ✓")

        st.divider()

# --- Export ---
st.download_button(
    label="📥 Download Executive Report (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="executive_review_report.csv",
    mime="text/csv",
    use_container_width=True,
)
