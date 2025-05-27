# streamlit_app.py
# ==================
# Full interactive dashboard for the AI-powered EdTech feedback system.

import streamlit as st
import pandas as pd
from src.data_loader import preprocess_feedback_df, load_csv
from src.sentiment_analysis import add_sentiment_columns
from src.recommender import prepare_surprise_data, train_svd_model, recommend_top_n
from src.hybrid_recommender import hybrid_recommend_top_n
from src.visualization import (
    plot_sentiment_distribution,
    plot_avg_rating_per_trainer,
    plot_sentiment_vs_rating,
    plot_learner_engagement
)

st.set_page_config(page_title="Talent Analytics and Training Optimization", layout="wide")
st.title("🎓 AI-Powered Feedback & Recommendation System")

# Load and preprocess data
feedback_df = load_csv("data/session_feedback.csv")
feedback_df = preprocess_feedback_df(feedback_df)
feedback_df = add_sentiment_columns(feedback_df)

# Sidebar - Select Learner ID
st.sidebar.header("🔍 Explore Recommendations")
learner_ids = feedback_df['learner_id'].unique().tolist()
selected_learner = st.sidebar.selectbox("Choose a Learner ID", learner_ids)

# Train recommender model
data = prepare_surprise_data(feedback_df)
algo, _ = train_svd_model(data)

# Get sentiment summary
trainer_sentiment_df = feedback_df.groupby('trainer_id')['vader_score'].mean().reset_index()
trainer_sentiment_df.rename(columns={'vader_score': 'avg_sentiment'}, inplace=True)

# Generate top 5 hybrid recommendations
rated = feedback_df[feedback_df['learner_id'] == selected_learner]['trainer_id'].tolist()
all_trainers = feedback_df['trainer_id'].unique().tolist()
recommendations = hybrid_recommend_top_n(
    algo, selected_learner, all_trainers, rated,
    trainer_sentiment_df=trainer_sentiment_df,
    weight_rating=0.6, weight_sentiment=0.4, n=5
)

# Display recommendations
st.subheader(f"Top Trainer Recommendations for Learner: {selected_learner}")
for trainer, score in recommendations:
    st.markdown(f"✅ **Trainer ID:** `{trainer}` — 📈 **Hybrid Score:** `{score:.2f}`")

st.markdown("---")

# Tabs for visualizations
tabs = st.tabs(["📊 Sentiment", "📈 Trainer Ratings", "🔍 Rating vs Sentiment", "👥 Learner Engagement"])

with tabs[0]:
    st.subheader("Sentiment Distribution (VADER)")
    plot_sentiment_distribution(feedback_df)

with tabs[1]:
    st.subheader("Top 10 Trainers by Average Rating")
    plot_avg_rating_per_trainer(feedback_df)

with tabs[2]:
    st.subheader("Sentiment vs Rating Scatter Plot")
    plot_sentiment_vs_rating(feedback_df)

with tabs[3]:
    st.subheader("Learner Session Participation")
    plot_learner_engagement(feedback_df)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Jai Dalmotra | 2025")
