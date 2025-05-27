# 💡 Project Workflow – AI-Powered Feedback & Recommendation System

This document outlines the chronological steps followed to complete the project.

---

## 1️⃣ Data Collection
- Collected learner feedback and trainer data (Kaggle + simulated)
- Saved raw CSVs in `/data/`

## 2️⃣ Preprocessing
- Used `data_loader.py` to clean feedback text, remove nulls, duplicates
- Saved clean version to `preprocessed_feedback.csv`

## 3️⃣ Sentiment Analysis
- Built `sentiment_analysis.py` using TextBlob and VADER
- Labeled feedback: Positive / Neutral / Negative
- Aggregated average sentiment score per trainer

## 4️⃣ Recommendation System
- Used `surprise.SVD` in `recommender.py` to model learner-trainer interactions
- Predicted top trainers for each learner

## 5️⃣ Hybrid Recommendation
- Created `hybrid_recommender.py` to blend SVD score + sentiment weight
- Re-ranked top-N trainers per learner based on hybrid score

## 6️⃣ Visualization
- Created `visualization.py` for plotting KPIs:
  - Sentiment distribution
  - Avg trainer rating
  - Learner session histograms
  - Sentiment vs Rating scatter

## 7️⃣ Dashboard
- Used `streamlit_app.py` to build interactive frontend
- Sidebar to input learner ID, see top trainers
- Tabbed visualizations using Seaborn and Matplotlib

## 8️⃣ Deployment
- App pushed to GitHub
- Deployed live via Streamlit Cloud
- Final documentation and README prepared

---

✅ Project Completed: End-to-End ML + NLP system  
📁 Structure: Modular, reproducible, deployable  
🚀 Skills: Data preprocessing, NLP, recommender systems, Streamlit, deployment
