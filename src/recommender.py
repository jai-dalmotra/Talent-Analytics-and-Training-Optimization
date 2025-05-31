"""
recommender.py
==============
This module builds and applies a collaborative filtering-based recommendation system using LightFM.
"""


import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k


def build_dataset(df):
    """
    Builds the LightFM dataset and interaction matrix from the ratings DataFrame.
    Returns the dataset object, interaction matrix, and weights.
    """
    dataset = Dataset()
    dataset.fit(
        users=df['learner_id'].unique(),
        items=df['trainer_id'].unique()
    )
    (interactions, weights) = dataset.build_interactions(
        [(row['learner_id'], row['trainer_id'], row['rating']) for _, row in df.iterrows()]
    )
    return dataset, interactions, weights


def train_lightfm_model(interactions, epochs=10, loss='warp'):
    """
    Trains a LightFM model using the given interactions.
    Returns the trained model.
    """
    model = LightFM(loss=loss)
    model.fit(interactions, epochs=epochs, num_threads=2)
    return model


def recommend_top_n(model, dataset, learner_id, rated_trainers, all_trainers, n=5):
    """
    Returns Top-N trainer recommendations for a learner based on LightFM predictions.
    """
    trainer_candidates = list(set(all_trainers) - set(rated_trainers))

    user_id_map, _, item_id_map, _ = dataset.mapping()
    uid = user_id_map[learner_id]

    scores = []
    for trainer_id in trainer_candidates:
        tid = item_id_map[trainer_id]
        score = model.predict(uid, np.array([tid]))[0]
        scores.append((trainer_id, score))

    top_n = sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    return top_n
