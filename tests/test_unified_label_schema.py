from __future__ import annotations

import pandas as pd
import pytest

from src.data.unified_label_schema import (
    normalize_unified_labels,
    validate_unified_labels,
)


def test_normalize_unified_labels_maps_expected_values() -> None:
    df = pd.DataFrame(
        {
            "text": ["a", "b"],
            "bias": ["Non-biased", "Biased"],
            "ideology": ["Left", "Right"],
            "propaganda": ["No", "Yes"],
            "narrative": ["Hero,Victim", "Villain"],
            "emotion": [0, "19"],
        }
    )

    normalized = normalize_unified_labels(df)

    assert normalized["bias"].tolist() == [0, 1]
    assert normalized["ideology"].tolist() == [0, 2]
    assert normalized["propaganda"].tolist() == [0, 1]
    assert normalized["emotion"].tolist() == [0, 19]
    assert normalized["narrative_hero"].tolist() == [1, 0]
    assert normalized["narrative_villain"].tolist() == [0, 1]
    assert normalized["narrative_victim"].tolist() == [1, 0]


def test_validate_unified_labels_reports_invalid_range() -> None:
    df = pd.DataFrame(
        {
            "bias": [0],
            "ideology": [1],
            "propaganda": [0],
            "narrative_hero": [0],
            "narrative_villain": [0],
            "narrative_victim": [0],
            "emotion": [22],
        }
    )

    errors = validate_unified_labels(df)

    assert errors
    assert "Unified label normalization failed" in errors[0]


def test_normalize_unified_labels_raises_for_missing_required_column() -> None:
    df = pd.DataFrame(
        {
            "bias": [0],
            "propaganda": [1],
            "emotion": [3],
        }
    )

    with pytest.raises(ValueError):
        normalize_unified_labels(df)
