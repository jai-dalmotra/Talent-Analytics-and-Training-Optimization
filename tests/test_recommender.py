# test_recommender.py
from src.recommender import prepare_surprise_data, train_svd_model, recommend_top_n
import pandas as pd

def test_svd_training():
    df = pd.DataFrame({
        'learner_id': ['L1', 'L2', 'L3', 'L1', 'L2'],
        'trainer_id': ['T1', 'T1', 'T2', 'T2', 'T3'],
        'rating': [5, 4, 3, 2, 4]
    })
    data = prepare_surprise_data(df)
    model, rmse = train_svd_model(data)
    assert model is not None
    assert rmse >= 0.0


def test_recommendation_output():
    df = pd.DataFrame({
        'learner_id': ['L1', 'L2'],
        'trainer_id': ['T1', 'T2'],
        'rating': [5, 3]
    })
    data = prepare_surprise_data(df)
    model, _ = train_svd_model(data)
    output = recommend_top_n(model, 'L1', ['T1', 'T2', 'T3'], ['T1'], n=2)
    assert isinstance(output, list)
    assert len(output) <= 2
    assert all(isinstance(x, tuple) and len(x) == 2 for x in output)
