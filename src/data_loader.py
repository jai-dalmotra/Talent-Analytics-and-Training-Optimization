"""
data_loader.py
==============
This module loads, validates, and preprocesses datasets used in the EdTech recommendation system.
It handles trainer profiles, learner feedback, and outputs cleaned versions for modeling and dashboards.
"""

import pandas as pd
import os
import re
import string
from typing import Optional


# -----------------------------
# 📁 Data Loading Functions
# -----------------------------

def load_csv(filepath: str) -> Optional[pd.DataFrame]:
    """
    Safely load a CSV file into a DataFrame.
    Returns None if file doesn't exist or fails to load.
    """
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {filepath} - shape: {df.shape}")
        return df
    except Exception as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return None


# -----------------------------
# 📊 Data Profiling & Validation
# -----------------------------

def profile_df(df: pd.DataFrame, name: str = "Data") -> None:
    """
    Print a quick profile of the dataframe: columns, nulls, dtypes.
    """
    print(f"\n📌 Profile: {name}")
    print(df.info())
    print("\n🧪 Null Values:")
    print(df.isnull().sum())
    print("\n📊 Sample Rows:")
    print(df.head())


def validate_columns(df: pd.DataFrame, required_cols: list, name: str = "Data") -> bool:
    """
    Check if required columns exist in the dataframe.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing columns in {name}: {missing}")
        return False
    return True


# -----------------------------
# ✂️ Text Cleaning
# -----------------------------

def clean_text(text: str) -> str:
    """
    Basic NLP-cleaning: lowercase, remove punctuation, strip whitespace.
    """
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_feedback_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare feedback DataFrame.
    Expected columns: ['learner_id', 'trainer_id', 'rating', 'feedback_text']
    Adds: 'cleaned_feedback'
    """
    # Validate
    required_cols = ['learner_id', 'trainer_id', 'rating', 'feedback_text']
    if not validate_columns(df, required_cols, name="Feedback Data"):
        return df

    # Drop rows with critical nulls
    df = df.dropna(subset=['learner_id', 'trainer_id', 'rating'])

    # Fill blank feedback text
    df['feedback_text'] = df['feedback_text'].fillna("")

    # Clean feedback column
    df['cleaned_feedback'] = df['feedback_text'].apply(clean_text)

    # Drop duplicates
    df = df.drop_duplicates(subset=['learner_id', 'trainer_id', 'feedback_text'])

    print(f"✅ Feedback data cleaned. Shape: {df.shape}")
    return df


# -----------------------------
# 🧼 Trainer Data Cleaning
# -----------------------------

def preprocess_trainer_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate trainer DataFrame.
    Expected columns: ['trainer_id', 'name', 'domain', 'experience']
    """
    required_cols = ['trainer_id', 'name', 'domain']
    if not validate_columns(df, required_cols, name="Trainer Data"):
        return df

    df = df.dropna(subset=['trainer_id', 'name'])

    df['domain'] = df['domain'].fillna('Unknown')
    df = df.drop_duplicates(subset=['trainer_id'])

    print(f"✅ Trainer data cleaned. Shape: {df.shape}")
    return df


# -----------------------------
# 💾 Save Processed Data
# -----------------------------

def save_processed(df: pd.DataFrame, path: str) -> None:
    """
    Save DataFrame to CSV.
    """
    try:
        df.to_csv(path, index=False)
        print(f"💾 Saved cleaned data to {path}")
    except Exception as e:
        print(f"❌ Failed to save {path}: {e}")
