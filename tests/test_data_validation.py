import pandas as pd
from src.data.validate_data import DataValidator


def test_validator_schema():

    validator = DataValidator(required_columns=['text', 'label'])

    df = pd.DataFrame({'text': ['sample'], 'label': [0]})

    assert validator.validate_schema(df)