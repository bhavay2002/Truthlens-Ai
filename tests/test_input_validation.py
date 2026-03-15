import pytest
from src.utils.input_validation import ensure_positive_int


def test_positive_int_valid():

    assert ensure_positive_int(5, name="value", min_value=1) == 5


def test_positive_int_invalid():

    with pytest.raises(ValueError):
        ensure_positive_int(0, name="value", min_value=1)