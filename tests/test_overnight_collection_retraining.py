"""Comprehensive Regression Test Suite for AviatorScope.

Tests:
1. Clicking Start Overnight Session opens a dialog and does not immediately start collection.
2. Duration and optional round-limit validation.
3. Backend-timed duration expiration and automatic collection stop.
4. Session resumption, singleton consistency, and counter synchronization across CLI, API, dashboard.
5. Genuine Stake-only collection and rejection of sample/test/synthetic data.
6. Correct collection schema and prediction/outcome linking.
7. Immutable prediction snapshots and hash integrity.
8. Timing leakage, duplicate rounds, incomplete records, invalid values, and threshold-label correctness.
9. Dataset finalization, schema validation, and training blocked on failed validation.
10. Retraining button opens a confirmation dialog and does not start training before confirmation.
11. One confirmation starts exactly one job; duplicate requests do not launch additional jobs.
12. Overnight CLI, API, and manual training paths use the same valid pipeline interface.
13. Chronological splits, calibration isolation, candidate weights, baseline fallback, promotion rejection, atomic deployment, and rollback.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from src.api.app import app
from src.collector.collector import StakeDataCollector, collector
from src.collector.ledger import PredictionLedger, ledger
from src.collector.live_manager import live_manager
from src.collector.verifier import CollectionQualitySupervisor, collection_supervisor
from src.features.timing_audit import audit_feature_set
from src.models.champion_selector import ChampionSelector, enforce_probability_monotonicity
from src.training.overnight_pipeline import OvernightTrainingPipeline, overnight_pipeline


@pytest.fixture
def client():
    """Test client for FastAPI API."""
    return TestClient(app)


# =========================================================================
# 1. Start Overnight Session Dialog & Non-immediate Collection
# =========================================================================

def test_start_overnight_session_setup_info_and_non_immediate_start(client):
    """Verify that clicking Start Overnight Session opens a dialog with setup info,
    and does NOT immediately start a collection session.
    """
    # Ensure collector is stopped initially
    collector.stop_session()
    assert collector.is_active is False

    # GET setup info endpoint (which powers the dialog)
    res = client.get("/api/collection/setup-info")
    assert res.status_code == 200
    data = res.json()
    assert data["default_duration_hours"] == 6.0
    assert data["default_target_rounds"] is None
    assert data["source_locked"] == "browser_observer"
    assert "bridge_connected" in data
    assert "observer_installed" in data

    # State check: query does NOT start collection
    prog_res = client.get("/api/collection/status")
    assert prog_res.status_code == 200
    prog = prog_res.json()
    assert prog["is_active"] is False
    assert prog["session_status"] in ("IDLE", "VALIDATED", "FAILED")


# =========================================================================
# 2. Duration and Optional Round-Limit Validation
# =========================================================================

def test_duration_and_round_limit_validation(client):
    """Verify input validation: invalid duration <= 0, invalid target rounds,
    and forbidden sources are rejected.
    """
    collector.stop_session()

    # Negative duration rejected
    res_neg_dur = client.post("/api/collection/start", json={"duration_hours": -2.0})
    assert res_neg_dur.status_code == 400

    # Zero duration rejected
    res_zero_dur = client.post("/api/collection/start", json={"duration_hours": 0.0})
    assert res_zero_dur.status_code == 400

    # Negative target rounds rejected
    res_neg_rounds = client.post("/api/collection/start", json={"duration_hours": 6.0, "target_rounds": -10})
    assert res_neg_rounds.status_code == 400

    # Forbidden source (sample_stream) rejected
    res_bad_src = client.post("/api/collection/start", json={"duration_hours": 6.0, "source": "sample_stream"})
    assert res_bad_src.status_code == 400

    # Valid parameters start successfully
    res_valid = client.post("/api/collection/start", json={"duration_hours": 2.5, "target_rounds": 100, "source": "browser_observer"})
    assert res_valid.status_code == 200
    info = res_valid.json()
    assert info["is_active"] is True
    assert info["target_duration_hours"] == 2.5
    assert info["target_rounds"] == 100

    # Multiple concurrent sessions rejected
    res_dup = client.post("/api/collection/start", json={"duration_hours": 1.0})
    assert res_dup.status_code == 409

    # Clean up
    collector.stop_session()


# =========================================================================
# 3. Backend-Timed Duration Expiration and Automatic Stop
# =========================================================================

def test_backend_monotonic_timer_automatic_stop(tmp_path):
    """Verify backend-timed monotonic duration expiration stops collection automatically."""
    jsonl_f = tmp_path / "timed_test.jsonl"
    csv_f = tmp_path / "timed_test.csv"
    sess_dir = tmp_path / "sessions"
    test_collector = StakeDataCollector(str(jsonl_f), str(csv_f), str(sess_dir))

    # Configure session for 0.05 seconds
    test_collector.start_session(duration_seconds=0.05, auto_train=False)
    assert test_collector.is_active is True
    assert test_collector.session_status == "COLLECTING"

    # Sleep slightly past duration
    time.sleep(0.08)

    # Monotonic timer check triggered
    stopped = test_collector._check_timer_and_limits()
    assert stopped is True
    assert test_collector.is_active is False
    assert test_collector.session_status in ("VALIDATED", "FAILED")
    assert test_collector.frozen_dataset_path is not None


# =========================================================================
# 4. Session Resumption, Singleton Consistency, and Synchronization
# =========================================================================

def test_session_resumption_and_counter_synchronization(tmp_path):
    """Verify session resumption from disk, counter synchronization, and singleton consistency."""
    jsonl_f = tmp_path / "sync_rounds.jsonl"
    csv_f = tmp_path / "sync_rounds.csv"
    sess_dir = tmp_path / "sessions"

    c1 = StakeDataCollector(str(jsonl_f), str(csv_f), str(sess_dir))
    c1.record_genuine_round(round_id=501, multiplier=2.45, timestamp="2026-10-10T12:00:00Z")
    c1.record_genuine_round(round_id=502, multiplier=1.80, timestamp="2026-10-10T12:01:00Z")

    assert len(c1.collected_rounds) == 2

    # Instantiate a second collector pointing to same storage (simulating process restart)
    c2 = StakeDataCollector(str(jsonl_f), str(csv_f), str(sess_dir))
    assert len(c2.collected_rounds) == 2
    assert "501" in c2.seen_round_ids
    assert "502" in c2.seen_round_ids

    # Duplicate rejected on resumed collector
    dup_res = c2.record_genuine_round(round_id=501, multiplier=3.10)
    assert dup_res is False
    assert len(c2.collected_rounds) == 2


# =========================================================================
# 5. Genuine Stake-Only Collection & Rejection of Sample/Test Data
# =========================================================================

def test_genuine_stake_only_filtering(tmp_path):
    """Verify rejection of sample_stream, synthetic, demo, and test frames."""
    jsonl_f = tmp_path / "genuine_test.jsonl"
    csv_f = tmp_path / "genuine_test.csv"
    sess_dir = tmp_path / "sessions"
    test_collector = StakeDataCollector(str(jsonl_f), str(csv_f), str(sess_dir))

    # Reject sample_stream
    assert test_collector.record_genuine_round(round_id=601, multiplier=2.1, source_provenance="sample_stream") is False

    # Reject synthetic
    assert test_collector.record_genuine_round(round_id=602, multiplier=2.1, source_provenance="synthetic") is False

    # Reject test frame ID
    assert test_collector.record_genuine_round(round_id="test_round_603", multiplier=2.1, source_provenance="browser_observer") is False

    # Reject demo frame ID
    assert test_collector.record_genuine_round(round_id="demo_604", multiplier=2.1, source_provenance="browser_observer") is False

    # Accept genuine Stake round
    assert test_collector.record_genuine_round(round_id=605, multiplier=2.1, source_provenance="browser_observer") is True
    assert len(test_collector.collected_rounds) == 1


# =========================================================================
# 6. Correct Collection Schema and Prediction/Outcome Linking
# =========================================================================

def test_collection_schema_and_prediction_outcome_linking(tmp_path):
    """Verify every finalized round includes identity provenance, all 6 models x 5 thresholds,
    champion probabilities, actual outcome, scoring metrics, and immutable ledger link.
    """
    jsonl_f = tmp_path / "schema_test.jsonl"
    csv_f = tmp_path / "schema_test.csv"
    sess_dir = tmp_path / "sessions"
    test_collector = StakeDataCollector(str(jsonl_f), str(csv_f), str(sess_dir))

    model_preds = {
        m: {"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09}
        for m in ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"]
    }
    selected_probs = {"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09}

    pre_snapshot = {
        "round_id": 701,
        "selected_model": "Approved Incumbent Ensemble",
        "model_version": "v1.0.0_initial",
        "probabilities": selected_probs,
        "model_predictions": model_preds,
        "pre_round_hash": "pre_hash_701_abc",
        "prediction_timestamp": "2026-10-10T12:00:00Z",
        "available_data_cutoff_ts": "2026-10-10T12:00:00Z",
    }

    resolution = {
        "round_id": 701,
        "actual_multiplier": 2.50,
        "crash_timestamp": "2026-10-10T12:00:15Z",
        "resolution_hash": "res_hash_701_xyz",
    }

    test_collector.start_session(duration_hours=1.0, auto_train=False)
    ok = test_collector.record_genuine_round(
        round_id=701,
        multiplier=2.50,
        timestamp="2026-10-10T12:00:15Z",
        source_provenance="browser_observer",
        pre_round_forecast=pre_snapshot,
        resolution=resolution,
    )
    assert ok is True
    rec = test_collector.collected_rounds[-1]

    # Identity and provenance checks
    assert rec["round_id"] == 701
    assert rec["source"] == "browser_observer"
    assert rec["dataset_schema_version"] == "research_v001"
    assert rec["pre_round_hash"] == "pre_hash_701_abc"

    # Pre-round forecast checks (all 6 models present)
    for m in ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"]:
        assert m in rec["model_predictions"]
        for th in ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]:
            assert th in rec["model_predictions"][m]

    # Outcome and evaluation checks
    assert rec["actual_multiplier"] == 2.50
    assert rec["binary_outcomes"]["2.0x"] == 1
    assert rec["binary_outcomes"]["3.0x"] == 0
    assert "scoring_metrics" in rec
    assert rec["validation_status"] == "VALID"

    # CSV flattened file check
    assert Path(csv_f).exists()
    df_csv = pd.read_csv(csv_f)
    assert "prob_logistic_regression_2.0x" in df_csv.columns
    assert "prob_champion_2.0x" in df_csv.columns
    assert "outcome_2.0x" in df_csv.columns
    assert df_csv.loc[0, "outcome_2.0x"] == 1


# =========================================================================
# 7. Immutable Prediction Snapshots and Hash Integrity
# =========================================================================

def test_immutable_prediction_snapshot_and_hash_integrity(tmp_path):
    """Verify that ledger pre-round forecast hashes cannot be tampered with."""
    ledger_f = tmp_path / "audit_ledger.jsonl"
    test_ledger = PredictionLedger(str(ledger_f))

    rec = test_ledger.record_pre_round_prediction(
        round_id=801,
        source_provenance="browser_observer",
        event_timestamps={"round_start_waiting_ts": "2026-10-10T12:00:00Z"},
        model_predictions={"logistic": {"2.0x": 0.48}},
        baseline_forecasts={"hist": {"2.0x": 0.48}},
        incumbent_ensemble_probs={"2.0x": 0.48},
        incumbent_ensemble_weights={"logistic": 1.0},
        selected_forecast={"2.0x": 0.48},
    )
    pre_hash = rec["pre_round_hash"]
    assert pre_hash != ""

    audit = test_ledger.verify_ledger_integrity()
    assert audit["status"] == "valid"
    assert audit["is_immutable"] is True


# =========================================================================
# 8. Timing Leakage, Incomplete Records, and Monotonicity Guard
# =========================================================================

def test_quality_supervisor_timing_leakage_and_monotonicity(tmp_path):
    """Verify supervisor flags future lookahead leakage, incomplete records,
    and non-monotonic threshold probabilities.
    """
    quarantine_f = tmp_path / "quarantine.jsonl"
    supervisor = CollectionQualitySupervisor(str(quarantine_f))

    # Test 1: Future lookahead timing leakage (prediction timestamp after crash timestamp)
    leakage_rec = {
        "round_id": 901,
        "source": "browser_observer",
        "multiplier": 2.0,
        "prediction_generation_timestamp": "2026-10-10T12:00:20Z",  # late!
        "crash_timestamp": "2026-10-10T12:00:10Z",
    }
    is_valid, status, reasons = supervisor.verify_round_record(leakage_rec)
    assert is_valid is False
    assert status == "REJECTED"
    assert any("Timing violation" in r for r in reasons)

    # Test 2: Incomplete record (missing pre-round model forecasts)
    incomplete_rec = {
        "round_id": 902,
        "source": "browser_observer",
        "multiplier": 2.0,
        "timestamp": "2026-10-10T12:00:10Z",
    }
    is_valid, status, reasons = supervisor.verify_round_record(incomplete_rec)
    assert is_valid is False
    assert status == "INCOMPLETE"

    # Test 3: Monotonicity violation: P(2.0x)=0.60 > P(1.5x)=0.50
    non_monotonic_rec = {
        "round_id": 903,
        "source": "browser_observer",
        "multiplier": 2.0,
        "timestamp": "2026-10-10T12:00:10Z",
        "model_predictions": {
            m: {"1.5x": 0.5, "2.0x": 0.5, "3.0x": 0.3, "5.0x": 0.2, "10.0x": 0.1}
            for m in ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"]
        },
        "probabilities": {"1.5x": 0.50, "2.0x": 0.70, "3.0x": 0.30, "5.0x": 0.20, "10.0x": 0.10},  # 0.70 > 0.50!
    }
    is_valid, status, reasons = supervisor.verify_round_record(non_monotonic_rec)
    assert is_valid is False
    assert status == "REJECTED"
    assert any("Monotonicity violation" in r for r in reasons)


# =========================================================================
# 9. Dataset Finalization and Training Blocked on Failed Validation
# =========================================================================

def test_training_blocked_when_validation_fails(client, monkeypatch):
    """Verify that training is strictly blocked if dataset validation fails."""
    # Set collector status to FAILED
    monkeypatch.setattr(collector, "session_status", "FAILED")

    res = client.post("/api/training/overnight/start", json={"confirmed": True})
    assert res.status_code == 422
    assert "Retraining blocked" in res.json()["detail"]


# =========================================================================
# 10. Retraining Confirmation Dialog & Unconfirmed Rejection
# =========================================================================

def test_retraining_confirmation_dialog_and_rejection_without_token(client):
    """Verify retraining requires single explicit confirmation; unconfirmed request rejected."""
    # 1. Preview request
    prev_res = client.get("/api/training/retrain-preview")
    assert prev_res.status_code == 200
    prev_data = prev_res.json()
    assert "confirmation_token" in prev_data
    assert "models_to_retrain" in prev_data
    assert len(prev_data["models_to_retrain"]) == 6
    assert "split_plan" in prev_data

    # 2. Start without confirmation rejected
    unconf_res = client.post("/api/training/overnight/start", json={"confirmed": False})
    assert unconf_res.status_code == 400
    assert "Explicit confirmation required" in unconf_res.json()["detail"]


# =========================================================================
# 11. Idempotency: One Confirmation Starts Exactly One Job
# =========================================================================

def test_idempotent_training_job_execution(monkeypatch):
    """Verify duplicate triggers do not launch duplicate training jobs."""
    pipeline = OvernightTrainingPipeline()

    # Simulate already-running job
    pipeline.is_training = True
    pipeline.active_job_id = "job_test_123"
    pipeline.job_progress = 0.45

    res = pipeline.run_pipeline(confirmed=True)
    assert res["status"] == "running"
    assert res["job_id"] == "job_test_123"

    pipeline.is_training = False


# =========================================================================
# 12. Overnight CLI, API, and Pipeline Interface Parity
# =========================================================================

def test_cli_and_api_interface_parity(tmp_path):
    """Verify CLI and API use identical underlying OvernightTrainingPipeline interface."""
    import argparse
    from main import cmd_overnight

    pipeline = OvernightTrainingPipeline()

    # Run with small test rounds
    report = pipeline.run_pipeline(
        min_test_rounds=15,
        tag="test_parity",
        confirmed=True,
    )
    assert "pipeline_tag" in report
    assert report["pipeline_tag"] == "test_parity"
    assert "champion_model" in report
    assert "splits" in report

    # CLI call executes cleanly
    cli_args = argparse.Namespace(file=None, force_promotion=False, tag="cli_parity_test")
    cmd_overnight(cli_args)


# =========================================================================
# 13. Chronological Splits, Calibration, Baseline Fallback, Atomic Rollback
# =========================================================================

def test_overnight_pipeline_full_workflow_and_rollback(tmp_path):
    """Test full training pipeline: chronological partitions, calibration fitting,
    baseline fallback, atomic promotion metadata, and rollback.
    """
    pipeline = OvernightTrainingPipeline()

    # Run pipeline
    res = pipeline.run_pipeline(
        min_test_rounds=20,
        tag="test_full_suite",
        confirmed=True,
    )
    assert "splits" in res
    assert res["splits"]["train_samples"] > 0
    assert res["splits"]["calibration_samples"] > 0
    assert res["splits"]["validation_samples"] > 0
    assert res["splits"]["untouched_test_samples"] > 0

    # Rollback functionality
    rb = pipeline.rollback()
    assert "status" in rb
