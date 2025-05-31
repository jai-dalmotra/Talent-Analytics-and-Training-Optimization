import pandas as pd
from src.recommender import build_interaction_matrix, train_implicit_model, recommend_items


def test_implicit_workflow():
    df = pd.DataFrame({
        'learner_id': ['L1', 'L2', 'L3', 'L1', 'L2'],
        'trainer_id': ['T1', 'T1', 'T2', 'T2', 'T3'],
        'rating': [5, 4, 3, 2, 4]
    })

    interaction_matrix, user_to_idx, item_to_idx, idx_to_item = build_interaction_matrix(df)
    model = train_implicit_model(interaction_matrix, factors=5, iterations=5)
    recs = recommend_items(model, 'L1', user_to_idx, idx_to_item, interaction_matrix)

    assert isinstance(recs, list)
    assert all(isinstance(x, tuple) and len(x) == 2 for x in recs)
