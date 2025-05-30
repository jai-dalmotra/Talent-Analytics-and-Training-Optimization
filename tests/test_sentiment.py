from src.sentiment_analysis import add_sentiment_columns
import pandas as pd


def test_sentiment_labels():
    sample = pd.DataFrame({
        'cleaned_feedback': [
            "This was fantastic!",
            "Not great, could be better.",
            "Just okay."
        ]
    })
    df = add_sentiment_columns(sample)
    assert 'tb_sentiment' in df.columns
    assert 'vader_sentiment' in df.columns
    assert set(df['tb_sentiment'].unique()).issubset({"Positive", "Negative", "Neutral"})
