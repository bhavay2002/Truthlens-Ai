import pandas as pd


def test_dataset_schema():

    df = pd.DataFrame({
        "title": [""],
        "text": ["news article"],
        "bias_label": [1],
        "ideology_label": [0],
        "propaganda_label": [0],
        "frame": ["economic"],
        "CO": [1],
        "EC": [0],
        "HI": [0],
        "MO": [1],
        "RE": [0],
        "hero": [0],
        "villain": [1],
        "victim": [0],
        "hero_entities": [""],
        "villain_entities": ["corporation"],
        "victim_entities": [""],
        "dataset": ["unit_test"],
        **{f"emotion_{idx}": [0] for idx in range(20)},
    })

    assert "title" in df.columns
    assert "text" in df.columns
    assert "bias_label" in df.columns
    assert "ideology_label" in df.columns
    assert "propaganda_label" in df.columns
    assert "frame" in df.columns
    for column in ["CO", "EC", "HI", "MO", "RE"]:
        assert column in df.columns
    for column in ["hero", "villain", "victim"]:
        assert column in df.columns
    for column in ["hero_entities", "villain_entities", "victim_entities"]:
        assert column in df.columns
    assert "dataset" in df.columns
    for idx in range(20):
        assert f"emotion_{idx}" in df.columns
