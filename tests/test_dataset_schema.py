import pandas as pd


def test_dataset_schema():

    df = pd.DataFrame({
        "text": ["news article"],
        "label": [1]
    })

    assert "text" in df.columns
    assert "label" in df.columns