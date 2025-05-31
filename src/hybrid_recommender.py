def hybrid_recommend_top_n(model, learner_id: str, user_to_idx: dict, idx_to_item: dict,
                           interaction_matrix, all_trainers: list, rated_trainers: list,
                           trainer_sentiment_df, item_to_idx: dict,
                           weight_rating=0.7, weight_sentiment=0.3, n=5):
    """
    Returns Top-N trainers for a learner using:
    - ALS predicted score
    - Average sentiment score per trainer
    - Weighted hybrid scoring
    """

    # Filter unseen trainers
    unseen = [t for t in all_trainers if t not in rated_trainers]

    # Fast lookup for sentiment scores
    sentiment_dict = dict(zip(trainer_sentiment_df['trainer_id'], trainer_sentiment_df['avg_sentiment']))

    results = []

    if learner_id not in user_to_idx:
        return []

    user_idx = user_to_idx[learner_id]
    user_vector = model.user_factors[user_idx]

    for trainer_id in unseen:
        item_idx = item_to_idx.get(trainer_id)
        if item_idx is None:
            continue

        item_vector = model.item_factors[item_idx]
        rating_score = np.dot(user_vector, item_vector)

        sentiment_score = sentiment_dict.get(trainer_id, 0.0)
        hybrid_score = (weight_rating * rating_score) + (weight_sentiment * sentiment_score * 5)  # normalize to 5

        results.append((trainer_id, hybrid_score))

    top_n = sorted(results, key=lambda x: x[1], reverse=True)[:n]
    return top_n
