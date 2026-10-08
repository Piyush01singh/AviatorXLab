"""Unit and integration tests for Aviator ML Research API and Frontend backend."""

import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.inference import process_round_records


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["models_count"] == 6


def test_api_benchmark_endpoint(client):
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "benchmark" in data
    assert len(data["benchmark"]) >= 6
    # Verify required keys
    for row in data["benchmark"]:
        assert "model" in row
        assert "roc_auc" in row
        assert "brier_score" in row
        assert "log_loss" in row


def test_api_serve_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Aviator ML Research" in response.text
    assert "Upload Round Records" in response.text


def test_api_sample_endpoint(client):
    response = client.get("/api/sample?source=synthetic")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["file_summary"]["total_rounds"] >= 51
    assert "next_round_prediction" in data
    assert "model_predictions" in data

    # Verify all 6 models exist in model_predictions
    expected_models = ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"]
    for m in expected_models:
        assert m in data["model_predictions"]
        for th in ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]:
            assert th in data["model_predictions"][m]
            assert 0.0 <= data["model_predictions"][m][th] <= 1.0


def test_api_upload_valid_csv(client):
    # Create valid synthetic CSV buffer with 60 rounds
    rows = []
    base_ts = pd.Timestamp("2026-01-01 12:00:00")
    for i in range(60):
        rows.append({
            "round_id": 1000 + i,
            "timestamp": (base_ts + pd.Timedelta(seconds=20 * i)).isoformat(),
            "multiplier": 1.5 + (i % 5) * 0.5,
        })
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/api/upload",
        files={"file": ("test_rounds.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["file_summary"]["total_rounds"] == 60
    assert data["next_round_prediction"]["round_id"] == 1060


def test_api_upload_invalid_csv_missing_columns(client):
    # CSV missing multiplier column
    df = pd.DataFrame({"round_id": [1, 2], "timestamp": ["2026-01-01", "2026-01-02"]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/api/upload",
        files={"file": ("bad_schema.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert any("multiplier" in err.lower() for err in data["errors"])


def test_api_upload_insufficient_rounds(client):
    # CSV with only 10 rounds (fewer than 51 required for 50-round lag/rolling window)
    rows = []
    for i in range(10):
        rows.append({
            "round_id": i + 1,
            "timestamp": f"2026-01-01T12:0{i}:00",
            "multiplier": 2.0,
        })
    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    response = client.post(
        "/api/upload",
        files={"file": ("few_rounds.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert "Insufficient rounds" in data["message"]


def test_api_upload_non_csv(client):
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"not a csv", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
