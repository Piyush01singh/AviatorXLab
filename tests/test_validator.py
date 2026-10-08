"""Unit tests for DataValidator."""

import pandas as pd
import pytest
from src.data.validator import DataValidator


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "round_id": [101, 102, 103, 104, 105],
        "timestamp": [
            "2026-01-01T12:00:00",
            "2026-01-01T12:00:15",
            "2026-01-01T12:00:30",
            "2026-01-01T12:00:45",
            "2026-01-01T12:01:00",
        ],
        "multiplier": [1.25, 2.40, 1.05, 5.80, 1.95],
    })


def test_validator_valid_dataset(valid_df):
    validator = DataValidator()
    report = validator.validate(valid_df)
    assert report.is_valid is True
    assert len(report.errors) == 0
    assert report.total_rows == 5


def test_validator_missing_column(valid_df):
    df = valid_df.drop(columns=["multiplier"])
    validator = DataValidator()
    report = validator.validate(df)
    assert report.is_valid is False
    assert any("Missing required columns" in e for e in report.errors)


def test_validator_duplicate_round_id(valid_df):
    df = valid_df.copy()
    df.loc[1, "round_id"] = 101  # duplicate
    validator = DataValidator()
    report = validator.validate(df)
    assert report.is_valid is False
    assert any("duplicate round_id" in e for e in report.errors)


def test_validator_invalid_multiplier(valid_df):
    df = valid_df.copy()
    df.loc[2, "multiplier"] = 0.85  # Below theoretical minimum 1.00x
    validator = DataValidator()
    report = validator.validate(df)
    assert report.is_valid is False
    assert any("below theoretical minimum" in e for e in report.errors)


def test_validator_non_chronological(valid_df):
    df = valid_df.copy()
    # Swap timestamps
    df.loc[0, "timestamp"] = "2026-01-01T12:05:00"
    validator = DataValidator()
    report = validator.validate(df)
    assert any("not strictly in chronological order" in w for w in report.warnings)
