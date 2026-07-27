import streamlit as st
import pandas as pd
from predictor import predict_batch
from src.business_rules import analyze_business_rules

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Batch C19", layout="wide")
st.title("Quick-Commerce Reviews Dashboard")
st.caption("Decision-grade insights for Product, Operations, and CX teams")
st.write("---")

# --- FILE UPLOAD ---
file = st.file_uploader("Upload Customer Reviews (CSV)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.info("AI is extracting intelligence from customer signals...")
else:
    st.info("💡 Loading default product reviews dataset (dataset.csv) for visualization...")
    try:
        df = pd.read_csv("dataset.csv")
    except Exception as e:
        st.error(f"Failed to load fallback dataset: {e}")
        st.stop()

# Validate presence of review/text columns
review_col = next((c for c in df.columns if c.lower() in ["review", "text", "content", "feedback"]), None)

if not review_col:
    st.error("No valid review column detected. Please check your CSV headers (Review, Text, Content, or Feedback).")
    st.stop()

reviews = df[review_col].astype(str).tolist()

# --- PROGRESSIVE BATCH ANALYSIS ---
progress_bar = st.progress(0)
status_msg = st.empty()
preds = []
total = len(reviews)
batch_size = 64

for i in range(0, total, batch_size):
    batch = reviews[i : i + batch_size]
    preds.extend(predict_batch(batch))
    p_val = min((i + batch_size) / total, 1.0)
    progress_bar.progress(p_val)
    status_msg.text(f"Processed {min(i+batch_size, total)} of {total} signals")

progress_bar.empty()
status_msg.success("Analysis Complete")

# --- DATA ENRICHMENT ---
df["Sentiment"] = preds

# Route through production business rules engine
results = [analyze_business_rules(row[review_col], row["Sentiment"]) for _, row in df.iterrows()]
df["Issue Category"] = [r[0] for r in results]
df["Priority"] = [r[1] for r in results]
df["Churn Risk"] = [r[2] for r in results]

# --- TOP DASHBOARD VIEW ---
# Detect ground truth rating column for dynamic accuracy calculation
rating_col = next((c for c in df.columns if c.lower() in ["rating", "score", "overall"]), None)
accuracy_text = "N/A"
accuracy_delta = "No Ground Truth"

if rating_col:
    try:
        rating_map = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
        label_map = {"Negative": 0, "Neutral": 1, "Positive": 2}
        
        # Match ratings and predictions
        valid_rows = df.dropna(subset=[rating_col, "Sentiment"])
        y_true = [rating_map.get(int(r), 1) for r in valid_rows[rating_col]]
        y_pred = [label_map.get(p, 1) for p in valid_rows["Sentiment"]]
        
        if len(y_true) > 0:
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_true, y_pred)
            accuracy_text = f"{acc * 100:.1f}%"
            accuracy_delta = f"On {len(y_true)} items"
    except Exception:
        accuracy_text = "Error"
        accuracy_delta = "Calculation Error"

m1, m2, m3 = st.columns(3)
m1.metric("Model/Dataset Accuracy", accuracy_text, delta=accuracy_delta)
m2.metric("P0 - Critical Issues", len(df[df["Priority"] == "P0 - CRITICAL"]))
# Churn risk alerts: Sum of High and Medium churn alerts
churn_alerts_count = len(df[df["Churn Risk"].isin(["High", "Medium"])])
m3.metric("Churn High/Medium Alerts", churn_alerts_count)

st.write("---")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Sentiment Distribution")
    st.bar_chart(df["Sentiment"].value_counts())
with c2:
    st.subheader("Top Customer Pain Points")
    neg_df = df[df["Sentiment"] == "Negative"]
    if not neg_df.empty:
        pain_points = neg_df["Issue Category"].value_counts().head(10)
        st.bar_chart(pain_points)
    else:
        st.info("No Negative Pain Points Detected")

st.write("---")

# --- INTELLIGENCE FEED ---
with st.expander("View Full Intelligence Feed"):
    view_df = df[[review_col, "Sentiment", "Priority", "Churn Risk", "Issue Category"]].copy()
    view_df.insert(0, "ID", range(1, len(view_df) + 1))
    st.dataframe(view_df, use_container_width=True)

st.write("---")

# --- ACTIONABLE QUEUE ---
st.subheader("Critical Priority Action Queue")
critical_cases = df[df["Priority"] == "P0 - CRITICAL"].head(5)

if critical_cases.empty:
    st.success("No critical safety/P0 issues detected.")
else:
    for i, (_, row) in enumerate(critical_cases.iterrows(), 1):
        review_text = row[review_col]
        
        # Determine actionable team based on issue category
        category_val = row.get("Issue Category", "Other")
        
        if category_val == "Refund":
            action, team, impact = "Fix refund SLA & automate payouts", "Finance & Tech", "+15% CSAT"
        elif category_val == "Late Delivery":
            action, team, impact = "Optimize dark-store routing & ETA accuracy", "Logistics", "+18% CSAT"
        elif category_val == "Missing Item":
            action, team, impact = "Audit warehouse picking & scanning flow", "Warehouse Ops", "+12% CSAT"
        elif category_val == "Customer Support":
            action, team, impact = "Reduce agent response time (AHT) & training", "CX Strategy", "+10% CSAT"
        elif category_val == "Damaged Product":
            action, team, impact = "Immediate cold-chain audit & vendor QC", "Quality Control", "+25% CSAT"
        else:
            action, team, impact = "Perform root-cause operational analysis", "Operations Team", "+8% CSAT"

        # Styled Warning Box
        st.error(f"**P0 ALERT | Ticket #{i} | Category: {category_val}**")
        st.markdown(f"> *\"{review_text}\"*")
        
        col1, col2, col3 = st.columns(3)
        col1.info(f"📍 **Action:** {action}")
        col2.write(f"💼 **Owner:** {team}")
        col3.write(f"📈 **Target:** {impact}")

        # Session State Assignment Logic
        btn_key = f"ticket_btn_{i}"
        state_key = f"assigned_{i}"
        
        if state_key not in st.session_state:
            st.session_state[state_key] = False

        if not st.session_state[state_key]:
            if st.button(f"Assign Ticket #{i}", key=btn_key):
                st.session_state[state_key] = True
                st.rerun() # Refresh to update UI immediately
        else:
            st.success(f"Ticket #{i} synced to {team} Workflow ✓")
        
        st.write("---")

# --- EXPORT REPORT ---
st.download_button(
    label="Download Executive CSV Report",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="executive_intel_report.csv",
    mime="text/csv",
    use_container_width=True
)