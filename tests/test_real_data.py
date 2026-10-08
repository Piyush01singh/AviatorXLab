"""Unit tests verifying real Aviator dataset integrity and provenance."""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.data.loader import load_csv
from src.data.validator import DataValidator


def test_real_aviator_dataset_integrity():
    raw_path = Path("data/raw/real_aviator_historical.csv")
    assert raw_path.is_file(), "Real dataset raw file must exist."

    df = load_csv(str(raw_path))
    assert len(df) >= 10000, f"Expected at least 10,000 real rounds, got {len(df)}"

    validator = DataValidator()
    report = validator.validate(df)
    assert report.is_valid, f"Validation failed: {report.errors}"
    assert report.stats["min_multiplier"] >= 1.00
    assert report.stats["max_multiplier"] > 100.0


def test_real_dataset_provenance_metadata():
    meta_path = Path("data/processed/real_dataset_v001_metadata.json")
    assert meta_path.is_file(), "Provenance metadata must exist."

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["dataset_version"] == "real_dataset_v001"
    assert "sha256" in meta
    assert len(meta["sha256"]) == 64  # valid SHA-256 hex string
    assert meta["row_count"] >= 10000
