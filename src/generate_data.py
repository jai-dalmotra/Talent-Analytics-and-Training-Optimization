# generate_data.py
# ============================
# Script to create and save cleaned + enriched datasets for feedback and trainer info

from src.data_loader import load_csv, preprocess_feedback_df, preprocess_trainer_df, save_processed
from src.sentiment_analysis import add_sentiment_columns, compute_avg_sentiment_per_trainer
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# --- Step 1: Load raw CSV files ---
feedback_raw = load_csv("data/session_feedback.csv")
trainer_raw = load_csv("data/trainer_profiles.csv")

# --- Step 2: Clean feedback and trainer data ---
feedback_clean = preprocess_feedback_df(feedback_raw)
trainer_clean = preprocess_trainer_df(trainer_raw)

# --- Step 3: Enrich feedback with sentiment scores ---
feedback_sentiment = add_sentiment_columns(feedback_clean)

# --- Step 4: Aggregate average sentiment per trainer (for hybrid recommender) ---
avg_sentiment_df = compute_avg_sentiment_per_trainer(feedback_sentiment)

# --- Step 5: Save outputs ---
save_processed(feedback_sentiment, "data/preprocessed_feedback.csv")
save_processed(trainer_clean, "data/preprocessed_trainers.csv")
save_processed(avg_sentiment_df, "data/avg_sentiment_per_trainer.csv")

print("✅ All processed data files saved in /data")
