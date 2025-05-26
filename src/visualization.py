"""
visualization.py
================
This module contains reusable functions for plotting KPIs and trends
from the feedback and recommendation system.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_sentiment_distribution(df: pd.DataFrame, column='vader_sentiment', title='Sentiment Distribution'):
    """
    Bar chart of sentiment classes (Positive / Neutral / Negative).
    """
    plt.figure(figsize=(6, 4))
    sns.countplot(x=column, data=df, palette='Set2')
    plt.title(title)
    plt.xlabel("Sentiment Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_avg_rating_per_trainer(df: pd.DataFrame):
    """
    Bar plot of average rating per trainer (top 10).
    """
    avg_ratings = df.groupby('trainer_id')['rating'].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=avg_ratings.index, y=avg_ratings.values, palette='Blues_r')
    plt.title("Top 10 Trainers by Avg Rating")
    plt.xlabel("Trainer ID")
    plt.ylabel("Average Rating")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_sentiment_vs_rating(df: pd.DataFrame):
    """
    Scatter plot: Sentiment score (VADER) vs Rating.
    """
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='vader_score', y='rating', alpha=0.6, hue='vader_sentiment')
    plt.title("VADER Sentiment Score vs Rating")
    plt.xlabel("VADER Score")
    plt.ylabel("Rating")
    plt.tight_layout()
    plt.show()


def plot_learner_engagement(df: pd.DataFrame):
    """
    Histogram: Sessions attended per learner.
    """
    learner_sessions = df.groupby('learner_id').size()
    plt.figure(figsize=(8, 4))
    sns.histplot(learner_sessions, bins=15, kde=False, color='teal')
    plt.title("Learner Engagement (Sessions Attended)")
    plt.xlabel("Number of Sessions")
    plt.ylabel("Learner Count")
    plt.tight_layout()
    plt.show()
