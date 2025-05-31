# test_recommender.py
from src.recommender import build_dataset, train_lightfm_model, recommend_trainers
import pandas as pd


def test_lightfm_pipeline():
    df = pd.DataFrame({
        'learner_id': ['L1', 'L2', 'L3', 'L1', 'L2'],
        'trainer_id': ['T1', 'T1', 'T2', 'T2', 'T3'],
        'rating': [5, 4, 3, 2, 4]
    })
    dataset, interactions, _ = build_dataset(df)
    model = train_lightfm_model(interactions, epochs=2)
    known = df[df['learner_id'] == 'L1']['trainer_id'].tolist()
    all_trainers = df['trainer_id'].unique().tolist()
    recs = recommend_trainers(model, dataset, 'L1', known, all_trainers)
    assert isinstance(recs, list)
    assert len(recs) > 0
