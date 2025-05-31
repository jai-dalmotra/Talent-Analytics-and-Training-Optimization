# streamlit_app.py
# ==================
# AI-Powered Feedback & Recommendation System Dashboard

import streamlit as st
import pandas as pd
from src.data_loader import preprocess_feedback_df, load_csv
from src.sentiment_analysis import add_sentiment_columns
from src.recommender import (
    build_interaction_matrix,
    train_implicit_model,
    recommend_items
)
from src.visualization import (
    plot_sentiment_distribution,
    plot_avg_rating_per_trainer,
    plot_sentiment_vs_rating,
    plot_learner_engagement,
    plot_learner_journey
)

# 🎛️ App Config
st.set_page_config(page_title="Talent Analytics and Training Optimization", layout="wide")
st.title("🎓 AI-Powered Feedback & Recommendation System")

# 📅 Load & Prepare Data
feedback_df = load_csv("data/session_feedback.csv")
feedback_df = preprocess_feedback_df(feedback_df)
feedback_df = add_sentiment_columns(feedback_df)

# 📊 Prepare Implicit ALS Recommender
ratings_df = feedback_df[['learner_id', 'trainer_id', 'rating']].dropna()
interaction_matrix, user_to_idx, item_to_idx, idx_to_item = build_interaction_matrix(ratings_df)
model = train_implicit_model(interaction_matrix)

# 📊 Sentiment Summary (optional for analysis)
trainer_sentiment_df = (
    feedback_df.groupby("trainer_id")["vader_score"]
    .mean().reset_index()
    .rename(columns={"vader_score": "avg_sentiment"})
)

# 🔍 Sidebar Controls
st.sidebar.header("🔧 Recommendation Engine Settings")

learner_ids = sorted(feedback_df["learner_id"].unique().tolist())
selected_learner = st.sidebar.selectbox("👤 Select a Learner", learner_ids)

n_recommendations = st.sidebar.slider("📌 Number of Recommendations", 1, 10, 5)

# 🧠 Generate Recommendations
recommendations = recommend_items(
    model=model,
    user_id=selected_learner,
    user_to_idx=user_to_idx,
    idx_to_item=idx_to_item,
    interaction_matrix=interaction_matrix,
    n=n_recommendations
)

# 📄 Display Top Recommendations
st.subheader(f"🌟 Top {n_recommendations} Trainer Recommendations for Learner `{selected_learner}`")
for trainer_id, score in recommendations:
    st.markdown(f"✅ **Trainer ID:** `{trainer_id}` — 💡 **Score:** `{score:.2f}`")

# 📂 Download Option
rec_df = pd.DataFrame(recommendations, columns=["trainer_id", "score"])
csv = rec_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📅 Download Recommendations as CSV",
    data=csv,
    file_name=f"als_recommendations_{selected_learner}.csv",
    mime="text/csv"
)

st.markdown("---")

# 📊 Dashboard Tabs
tabs = st.tabs([
    "📊 Sentiment Distribution",
    "📈 Top Trainer Ratings",
    "🔍 Rating vs Sentiment",
    "👥 Learner Engagement",
    "📘 Learner Feedback Journey"
])

with tabs[0]:
    st.subheader("Sentiment Analysis Overview")
    plot_sentiment_distribution(feedback_df)

with tabs[1]:
    st.subheader("Top 10 Trainers by Average Rating")
    plot_avg_rating_per_trainer(feedback_df)

with tabs[2]:
    st.subheader("Sentiment Score vs Feedback Rating")
    plot_sentiment_vs_rating(feedback_df)

with tabs[3]:
    st.subheader("Learner Engagement Distribution")
    plot_learner_engagement(feedback_df)

with tabs[4]:
    st.subheader(f"Session Feedback Journey: Learner `{selected_learner}`")
    plot_learner_journey(feedback_df[feedback_df["learner_id"] == selected_learner])

# 👣 Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Jai Dalmotra | 2025")
