"""
recommender.py
==============
This module builds and applies a collaborative filtering-based recommendation system using LightFM.
"""


# src/recommender.py
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from implicit.als import AlternatingLeastSquares


def build_interaction_matrix(df, user_col='learner_id', item_col='trainer_id', rating_col='rating'):
    users = list(df[user_col].unique())
    items = list(df[item_col].unique())

    user_to_idx = {user: idx for idx, user in enumerate(users)}
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    rows = df[user_col].map(user_to_idx)
    cols = df[item_col].map(item_to_idx)
    data = df[rating_col].astype(float)

    interaction_matrix = coo_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    return interaction_matrix.tocsr(), user_to_idx, item_to_idx, idx_to_item


def train_implicit_model(interaction_matrix, factors=20, iterations=15, regularization=0.1):
    model = AlternatingLeastSquares(factors=factors, iterations=iterations, regularization=regularization)
    model.fit(interaction_matrix)
    return model


def recommend_items(model, learner_id, user_to_idx, idx_to_item, interaction_matrix, N=5):
    if learner_id not in user_to_idx:
        return []

    user_idx = user_to_idx[learner_id]

    # Extract the user's interaction row as a CSR matrix
    user_items = interaction_matrix[user_idx]
    from scipy.sparse import csr_matrix
    user_items_csr = user_items.tocsr() if not isinstance(user_items, csr_matrix) else user_items

    # Get recommended items and scores separately
    item_ids, scores = model.recommend(user_idx, user_items_csr, N=N)

    # Pair them together and map back to original item IDs
    return [(idx_to_item[int(item_id)], float(score)) for item_id, score in zip(item_ids, scores)]
