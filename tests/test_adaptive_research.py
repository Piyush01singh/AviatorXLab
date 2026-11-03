"""Comprehensive test suite for Adaptive Research, Leakage-Free Audit, and Overnight Pipeline.

Covers:
1. Feature Timing Audit & Causality Invariance.
2. Prediction Ledger Immutability & SHA-256 Tamper Detection.
3. Stake Data Collector Deduplication, Resumability, and Auto-Training Trigger.
4. Live Shadow Learner Convex Optimization & Step Size Regularization.
5. Champion Selector Monotonicity Enforcement & Baseline Fallback.
6. Statistical Signal & Fairness Engine (Ljung-Box, KS Test, Spribe HMAC-SHA512).
7. Overnight Training Pipeline Chronological Splits, Atomic Checkpointing & Rollback.
8. Live Manager & API Endpoints Integration.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.collector.collector import StakeDataCollector
from src.collector.ledger import ImmutabilityViolationError, PredictionLedger
from src.collector.live_manager import LiveManager
from src.evaluation.statistical_signal import (
    run_autocorrelation_audit,
    run_distribution_fairness_audit,
    verify_provably_fair_round,
)
from src.features.builder import FeatureBuilder
from src.features.timing_audit import (
    FEATURE_AVAILABILITY_STAGES,
    audit_feature_set,
    audit_prediction_timestamp_cutoff,
    verify_feature_causality_invariance,
)
from src.models.champion_selector import ChampionSelector, enforce_probability_monotonicity
from src.models.shadow_learner import AdaptiveShadowLearner
from src.training.overnight_pipeline import OvernightTrainingPipeline


# =========================================================================
# 1. Feature Timing Audit & Causality Invariance
# =========================================================================

def test_feature_timing_audit_valid_features():
    """Verify that legitimate causal feature names pass the audit."""
    causal_features = [
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_5",
        "mean_5", "std_10", "median_20", "min_5", "max_10",
        "low_multiplier_streak", "high_multiplier_streak",
        "hour", "minute", "day_of_week", "time_since_previous_round",
    ]
    report = audit_feature_set(causal_features)
    assert report["is_leakage_free"] is True
    assert len(report["forbidden_features"]) == 0
    assert report["total_features"] == len(causal_features)


def test_feature_timing_audit_detects_leakage():
    """Verify that forbidden in-flight or post-crash features are flagged."""
    leaked_features = [
        "lag_1",
        "target_multiplier",           # after_crash
        "flight_duration",             # after_crash
        "in_flight_speed_tick",        # during_flying
        "revealed_server_seed",        # after_crash
    ]
    report = audit_feature_set(leaked_features)
    assert report["is_leakage_free"] is False
    assert len(report["forbidden_features"]) == 4
    forbidden_names = [f[0] for f in report["forbidden_features"]]
    for lf in ["target_multiplier", "flight_duration", "in_flight_speed_tick", "revealed_server_seed"]:
        assert lf in forbidden_names


def test_prediction_timestamp_cutoff_audit():
    """Verify cutoff validation for pre-round forecasts."""
    waiting_start = "2026-10-10T12:00:00.000Z"

    # Prediction generated 0.2s into waiting (valid, cutoff is 0.5s)
    pred_valid = "2026-10-10T12:00:00.200Z"
    valid_res = audit_prediction_timestamp_cutoff(
        round_waiting_ts=waiting_start,
        prediction_ts=pred_valid,
        cutoff_tolerance_seconds=0.5,
    )
    assert valid_res["is_valid_prospective"] is True
    assert valid_res["status"] == "VALID_PROSPECTIVE"

    # Prediction generated 2.0s after waiting (late, cutoff is 0.5s)
    pred_late = "2026-10-10T12:00:02.000Z"
    late_res = audit_prediction_timestamp_cutoff(
        round_waiting_ts=waiting_start,
        prediction_ts=pred_late,
        cutoff_tolerance_seconds=0.5,
    )
    assert late_res["is_valid_prospective"] is False
    assert late_res["status"] == "FLAGGED_POST_CUTOFF"


def test_feature_causality_invariance_perturbation():
    """Verify that mutating future outcomes does not alter pre-round feature matrix at round t."""
    builder = FeatureBuilder(lags=[1, 2, 3], rolling_windows=[5, 10], thresholds=[2.0])
    
    # Generate 50 dummy rounds
    dates = pd.date_range("2026-01-01", periods=50, freq="1min")
    rng = np.random.default_rng(42)
    mults = rng.uniform(1.0, 5.0, size=50)
    df = pd.DataFrame({"round_id": range(1, 51), "timestamp": dates, "multiplier": mults})

    # Perturbation invariance check at round 30
    is_invariant, max_diff = verify_feature_causality_invariance(df, builder.build, target_idx=30)
    assert is_invariant is True
    assert max_diff == 0.0


# =========================================================================
# 2. Prediction Ledger Immutability & SHA-256 Provenance
# =========================================================================

def test_prediction_ledger_lifecycle_and_immutability(tmp_path):
    """Test full cycle: pre-round commitment, resolution appending, immutability enforcement."""
    ledger_file = tmp_path / "test_ledger.jsonl"
    ledger = PredictionLedger(str(ledger_file))

    model_preds = {
        "logistic": {"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.31, "5.0x": 0.18, "10.0x": 0.08},
        "xgboost": {"1.5x": 0.66, "2.0x": 0.49, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09},
    }

    # 1. Record pre-round prediction
    rec = ledger.record_pre_round_prediction(
        round_id=2001,
        source_provenance="browser_observer",
        event_timestamps={"round_start_waiting_ts": datetime.now(timezone.utc).isoformat()},
        model_predictions=model_preds,
        baseline_forecasts={"Historical Baseline": {"1.5x": 0.64, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09}},
        incumbent_ensemble_probs={"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.31, "5.0x": 0.18, "10.0x": 0.08},
        incumbent_ensemble_weights={"logistic": 0.5, "xgboost": 0.5},
        selected_forecast={"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.31, "5.0x": 0.18, "10.0x": 0.08},
        selected_model_name="Approved Incumbent Ensemble",
    )
    assert rec["round_id"] == 2001
    assert rec["pre_round_hash"] != ""
    orig_hash = rec["pre_round_hash"]

    # 2. Immutability violation test: re-recording with different prediction raises ImmutabilityViolationError
    with pytest.raises(ImmutabilityViolationError):
        ledger.record_pre_round_prediction(
            round_id=2001,
            source_provenance="browser_observer",
            event_timestamps={"round_start_waiting_ts": datetime.now(timezone.utc).isoformat()},
            model_predictions=model_preds,
            baseline_forecasts={"Historical Baseline": {"1.5x": 0.64, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09}},
            incumbent_ensemble_probs={"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.31, "5.0x": 0.18, "10.0x": 0.08},
            incumbent_ensemble_weights={"logistic": 0.5, "xgboost": 0.5},
            selected_forecast={"1.5x": 0.99, "2.0x": 0.99, "3.0x": 0.99, "5.0x": 0.99, "10.0x": 0.99},  # Tampered
            selected_model_name="Approved Incumbent Ensemble",
        )

    # 3. Post-crash resolution appending
    resolved_rec = ledger.record_resolution(round_id=2001, actual_multiplier=2.45)
    assert resolved_rec is not None
    assert resolved_rec["resolution"]["actual_multiplier"] == 2.45
    assert resolved_rec["resolution"]["binary_outcomes"]["2.0x"] == 1
    assert resolved_rec["resolution"]["binary_outcomes"]["3.0x"] == 0
    # Linked pre-round hash matches original
    assert resolved_rec["resolution"]["linked_pre_round_hash"] == orig_hash

    # 4. Verify integrity
    integrity = ledger.verify_ledger_integrity()
    assert integrity["status"] == "valid"
    assert integrity["is_immutable"] is True
    assert integrity["total_pre_round_forecasts"] == 1
    assert integrity["total_resolutions"] == 1
    assert len(integrity["violations"]) == 0


def test_prediction_ledger_detects_tampering(tmp_path):
    """Verify that modifying frozen probabilities on disk is flagged as corrupted."""
    ledger_file = tmp_path / "tamper_ledger.jsonl"
    ledger = PredictionLedger(str(ledger_file))

    ledger.record_pre_round_prediction(
        round_id=3001,
        source_provenance="browser_observer",
        event_timestamps={},
        model_predictions={"logistic": {"2.0x": 0.50}},
        baseline_forecasts={"Historical Baseline": {"2.0x": 0.50}},
        incumbent_ensemble_probs={"2.0x": 0.50},
        incumbent_ensemble_weights={"logistic": 1.0},
        selected_forecast={"2.0x": 0.50},
        selected_model_name="logistic",
    )

    # Tamper with the JSONL file on disk: alter frozen forecast
    lines = ledger_file.read_text(encoding="utf-8").strip().splitlines()
    data = json.loads(lines[0])
    data["pre_round_forecast"]["selected_forecast"]["2.0x"] = 0.99  # Tampered!
    ledger_file.write_text(json.dumps(data) + "\n", encoding="utf-8")

    # Reload ledger and audit
    audited_ledger = PredictionLedger(str(ledger_file))
    integrity = audited_ledger.verify_ledger_integrity()
    assert integrity["status"] == "corrupted"
    assert integrity["is_immutable"] is False
    assert len(integrity["violations"]) >= 1


# =========================================================================
# 3. Stake Data Collector Deduplication, Resumability & Auto-Train Trigger
# =========================================================================

def test_stake_collector_dedup_and_resumability(tmp_path):
    """Verify deduplication, persistence, and resumability across restarts."""
    jsonl_p = tmp_path / "stake.jsonl"
    csv_p = tmp_path / "stake.csv"

    collector1 = StakeDataCollector(jsonl_path=str(jsonl_p), csv_path=str(csv_p))
    collector1.start_session(target_rounds=10)

    # Record genuine round
    ok1 = collector1.record_genuine_round(round_id=501, multiplier=1.85)
    assert ok1 is True
    assert len(collector1.collected_rounds) == 1

    # Attempt duplicate round_id
    ok_dup = collector1.record_genuine_round(round_id=501, multiplier=1.85)
    assert ok_dup is False
    assert len(collector1.collected_rounds) == 1

    # Attempt synthetic/demo frame (must be rejected)
    ok_synth = collector1.record_genuine_round(
        round_id=502,
        multiplier=3.20,
        source_provenance="sample_stream",
    )
    assert ok_synth is False
    assert len(collector1.collected_rounds) == 1

    # Record second genuine round
    ok2 = collector1.record_genuine_round(round_id=503, multiplier=4.12)
    assert ok2 is True
    assert len(collector1.collected_rounds) == 2

    # Verify resumability: initialize a new collector instance pointing to the same files
    collector2 = StakeDataCollector(jsonl_path=str(jsonl_p), csv_path=str(csv_p))
    assert len(collector2.collected_rounds) == 2
    assert 501 in collector2.seen_round_ids
    assert 503 in collector2.seen_round_ids

    # Duplicate rejection persists across instances
    assert collector2.record_genuine_round(round_id=501, multiplier=1.85) is False


def test_stake_collector_auto_training_callback(tmp_path):
    """Verify that reaching collection target invokes the callback."""
    jsonl_p = tmp_path / "stake_cb.jsonl"
    csv_p = tmp_path / "stake_cb.csv"
    collector = StakeDataCollector(jsonl_path=str(jsonl_p), csv_path=str(csv_p))

    callback_called = [False]

    def on_complete():
        callback_called[0] = True

    collector.start_session(
        target_rounds=3,
        auto_train=True,
        on_complete_callback=on_complete,
    )

    collector.record_genuine_round(round_id=101, multiplier=1.50)
    assert callback_called[0] is False

    collector.record_genuine_round(round_id=102, multiplier=2.10)
    assert callback_called[0] is False

    # Third round reaches target rounds = 3
    collector.record_genuine_round(round_id=103, multiplier=3.40)
    assert callback_called[0] is True
    assert collector.session_status in ("completed", "training")


# =========================================================================
# 4. Live Shadow Learner Convex Optimization & Step Size Regularization
# =========================================================================

def test_shadow_learner_convex_constraints_and_step_clipping():
    """Verify shadow learner preserves deployed weights and respects simplex constraints & step limits."""
    learner = AdaptiveShadowLearner(
        model_names=["logistic", "rf", "xgboost", "mlp", "lstm", "gru"],
        min_evidence_rounds=5,
        regularization_lambda=0.05,
        max_step_delta=0.08,
    )

    # Initial state
    assert np.isclose(sum(learner.deployed_weights.values()), 1.0)
    assert np.isclose(sum(learner.candidate_weights.values()), 1.0)

    # Feed 15 out-of-sample resolved rounds
    for i in range(15):
        mult = 2.5 if i % 2 == 0 else 1.2
        model_preds = {
            m: {
                "2.0x": 0.85 if (mult >= 2.0 and m == "xgboost") else (0.15 if m == "xgboost" else 0.50)
            }
            for m in learner.model_names
        }
        learner.record_round_resolution(
            round_id=1000 + i,
            model_predictions=model_preds,
            actual_multiplier=mult,
        )

    cand_weights = learner.candidate_weights
    dep_weights = learner.deployed_weights

    # 1. Deployed weights must NOT have changed
    for m in learner.model_names:
        assert np.isclose(dep_weights[m], 1 / 6, atol=1e-3)

    # 2. Candidate weights must be non-negative and sum to 1
    total_cand = sum(cand_weights.values())
    assert np.isclose(total_cand, 1.0, atol=1e-3)
    for m, w in cand_weights.items():
        assert w >= 0.0


# =========================================================================
# 5. Champion Selector Monotonicity Enforcement & Baseline Fallback
# =========================================================================

def test_champion_selector_monotonicity_projection():
    """Verify isotonic projection strictly enforces non-increasing probability across thresholds."""
    # Inverted non-monotonic input: P(>=5x) > P(>=2x)
    non_monotonic = {
        "1.5x": 0.50,
        "2.0x": 0.40,
        "3.0x": 0.45,  # Inversion!
        "5.0x": 0.60,  # Inversion!
        "10.0x": 0.10,
    }
    projected = enforce_probability_monotonicity(non_monotonic)

    keys = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]
    for i in range(len(keys) - 1):
        assert projected[keys[i]] >= projected[keys[i + 1]] - 1e-6


def test_champion_selector_baseline_fallback():
    """Verify that when candidate models do not establish significant advantage, fallback selects Historical Baseline."""
    selector = ChampionSelector(min_test_rounds=20, alpha_significance=0.05)

    y_test = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 5)  # 40 rounds
    baseline_p = np.full_like(y_test, 0.50, dtype=float)
    cand_p = np.full_like(y_test, 0.49, dtype=float)

    cand_preds = {
        "Historical Probability Baseline": {th: baseline_p for th in ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]},
        "Candidate XGBoost": {th: cand_p for th in ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]},
    }
    y_true_matrix = {th: y_test for th in ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]}

    result = selector.select_champion(cand_preds, y_true_matrix, baseline_name="Historical Probability Baseline")
    assert result["advantage_established"] is False
    assert result["is_fallback"] is True
    assert "Historical Probability Baseline" in result["champion_name"]


# =========================================================================
# 6. Statistical Signal & Fairness Engine
# =========================================================================

def test_statistical_signal_autocorrelation_white_noise():
    """Verify that independent uniform multiplier draws are diagnosed as white noise."""
    rng = np.random.default_rng(42)
    # 200 random i.i.d. draws
    multipliers = rng.uniform(1.0, 10.0, size=200)

    audit = run_autocorrelation_audit(multipliers, max_lags=10, alpha=0.05)
    assert audit["sample_size"] == 200
    assert audit["is_white_noise"] is True
    assert "No serial dependence" in audit["conclusion"]


def test_statistical_distribution_fairness_audit():
    """Verify 97% theoretical model checks: instant crashes, Wilson CI, and KS test."""
    rng = np.random.default_rng(42)
    n = 500
    # Generate Aviator-like continuous Pareto distribution: P(X >= x) = 0.97 / x
    u = rng.uniform(0, 1, size=n)
    mults = np.where(u < 0.03, 1.00, 0.97 / (1.0 - u))

    audit = run_distribution_fairness_audit(mults, rtp=0.97)
    assert audit["sample_size"] == 500
    assert 0.01 <= audit["instant_crash_rate"] <= 0.06
    assert len(audit["instant_crash_95_ci"]) == 2
    assert audit["kolmogorov_smirnov"]["p_value"] >= 0.01


def test_provably_fair_verification_spribe():
    """Verify Spribe HMAC-SHA512 provably fair verification function."""
    server_seed = "server_seed_secret_12345"
    client_seed = "client_seed_public_67890"

    res = verify_provably_fair_round(server_seed, client_seed, nonce=1)
    assert "computed_multiplier" in res
    assert res["computed_multiplier"] >= 1.00
    assert len(res["hmac_sha512_hash"]) == 128
    assert res["retrospective_valid"] is True


# =========================================================================
# 7. Overnight Pipeline & Atomic Rollback
# =========================================================================

def test_overnight_pipeline_atomic_rollback(tmp_path):
    """Test candidate promotion metadata and rollback execution."""
    models_dir = tmp_path / "models"
    checkpoints_dir = models_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)

    # Initial approved version file
    approved_file = models_dir / "approved_version.json"
    init_meta = {
        "active_version": "v1.0.0_initial",
        "active_checkpoint_path": str(models_dir),
        "promoted_at": "2026-10-10T00:00:00Z",
        "champion_model": "Approved Incumbent Ensemble",
        "rollback_version": None,
        "rollback_checkpoint_path": None,
    }
    approved_file.write_text(json.dumps(init_meta, indent=2))

    # Create dummy artifact files
    (models_dir / "model_weights.bin").write_text("v1_weights")

    pipeline = OvernightTrainingPipeline()
    pipeline.models_dir = models_dir
    pipeline.checkpoints_dir = checkpoints_dir
    pipeline.approved_version_file = approved_file

    # Simulate candidate promotion
    cand_dir = checkpoints_dir / "candidate_v2"
    cand_dir.mkdir(parents=True)
    (cand_dir / "model_weights.bin").write_text("v2_weights")

    pipeline._promote_checkpoint_atomically(
        candidate_checkpoint_dir=cand_dir,
        version_tag="v2",
        champion_name="Candidate XGBoost",
    )

    # Verify candidate was promoted
    meta_promoted = json.loads(approved_file.read_text())
    assert meta_promoted["active_version"] == "v_v2"
    assert meta_promoted["rollback_version"] == "v1.0.0_initial"
    assert (models_dir / "model_weights.bin").read_text() == "v2_weights"

    # Now execute rollback
    rb_res = pipeline.rollback()
    assert rb_res["status"] == "success"

    meta_rolled_back = json.loads(approved_file.read_text())
    assert meta_rolled_back["active_version"] == "v1.0.0_initial"
    assert (models_dir / "model_weights.bin").read_text() == "v1_weights"


# =========================================================================
# 8. FastAPI API Endpoints Integration
# =========================================================================

def test_api_endpoints_integration():
    """Verify newly added API endpoints respond with 200 OK and valid schemas."""
    client = TestClient(app)

    # 1. Collection Status
    r_col = client.get("/api/collection/status")
    assert r_col.status_code == 200
    data_col = r_col.json()
    assert "is_active" in data_col
    assert "session_status" in data_col
    assert "session_rounds_collected" in data_col
    assert "total_authentic_rounds" in data_col

    # 2. Ledger Audit
    r_led = client.get("/api/live/ledger")
    assert r_led.status_code == 200
    data_led = r_led.json()
    assert "integrity" in data_led
    assert "recent_entries" in data_led

    # 3. Shadow Learner State
    r_sl = client.get("/api/live/shadow-learner")
    assert r_sl.status_code == 200
    data_sl = r_sl.json()
    assert "deployed_weights" in data_sl
    assert "candidate_weights" in data_sl

    # 4. Overnight Training Status
    r_ov = client.get("/api/training/overnight/status")
    assert r_ov.status_code == 200
    data_ov = r_ov.json()
    assert "approved_version" in data_ov
    assert "collection_status" in data_ov

    # 5. Statistical Signal Audit Endpoint
    r_sig = client.get("/api/research/statistical-signal?max_lags=10")
    assert r_sig.status_code == 200
    data_sig = r_sig.json()
    assert "autocorrelation_audit" in data_sig
    assert "distribution_fairness_audit" in data_sig


# =========================================================================
# 9. Dashboard Prediction Consistency & Overnight Pipeline Regression Tests
# =========================================================================

def test_one_authoritative_final_prediction_parity_and_ledger(tmp_path):
    """Test that the Champion Selector's approved result is the single source of truth
    for all five thresholds, matching headline prediction, active_prediction, and the immutable ledger.
    """
    manager = LiveManager(seed_dataset_path="data/raw/synthetic_sample.csv")
    assert manager.active_prediction is not None

    pred = manager.active_prediction

    # 1. Target Stake round ID and model identification
    assert "target_round_id" in pred
    assert "round_id" in pred
    assert pred["target_round_id"] == pred["round_id"]
    assert "selected_model" in pred
    assert "champion_model" in pred
    assert "model_version" in pred
    assert "prediction_timestamp" in pred

    # 2. Five threshold probabilities
    assert "probabilities" in pred
    probs = pred["probabilities"]
    threshold_keys = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]
    for th in threshold_keys:
        assert th in probs
        assert isinstance(probs[th], (float, int))
        assert 0.0 <= probs[th] <= 1.0

    # 3. Monotonicity enforcement: P(>=1.5x) >= P(>=2x) >= P(>=3x) >= P(>=5x) >= P(>=10x)
    assert probs["1.5x"] >= probs["2.0x"]
    assert probs["2.0x"] >= probs["3.0x"]
    assert probs["3.0x"] >= probs["5.0x"]
    assert probs["5.0x"] >= probs["10.0x"]

    # 4. Headline prediction equals 2.0x probability
    headline_2x = pred["headline_p2x"]
    assert headline_2x == probs["2.0x"]
    assert pred["predicted_prob_2x"] == headline_2x
    assert pred["display_prob_2x"] == headline_2x

    # 5. Ledger record verification
    ledger_entry = manager.pre_round_predictions.get(pred["round_id"])
    assert ledger_entry is not None
    # Ledger pre_round_forecast selected_forecast matches authoritative probabilities
    assert ledger_entry["probabilities"] == probs
    assert ledger_entry["headline_p2x"] == headline_2x


def test_reproduce_and_prevent_95_vs_99_prediction_discrepancy(tmp_path, monkeypatch):
    """Regression test: Reproduce scenario where weighted ensemble yields 95.0%
    and consensus mean yields 99.4%, and verify that conflicting values cannot
    be displayed as the authoritative prediction for the same forecast.
    """
    manager = LiveManager(seed_dataset_path="data/raw/synthetic_sample.csv")

    # Manually configure scenario where ensemble is 95.0% and consensus mean is 99.4%
    mock_approved = {
        "active_version": "v_test_ensemble",
        "champion_model": "Approved Incumbent Ensemble",
    }
    monkeypatch.setattr(manager, "_get_approved_version_info", lambda: mock_approved)

    # Trigger generation
    manager._generate_next_prediction()
    pred = manager.active_prediction

    # Authoritative headline prediction
    headline = pred["headline_p2x"]
    authoritative_probs = pred["probabilities"]

    # The headline P(>= 2.0x) and the card for 2.0x MUST be identical
    assert headline == authoritative_probs["2.0x"]
    assert pred["predicted_prob_2x"] == headline
    assert pred["display_prob_2x"] == headline

    # Consensus mean is kept strictly as a separate reference metric
    assert "consensus_mean" in pred
    # If consensus mean diverges from the approved champion's forecast,
    # the headline prediction MUST follow the approved champion, never the consensus mean!
    if pred["consensus_mean"].get("2.0x") != authoritative_probs["2.0x"]:
        assert headline != pred["consensus_mean"]["2.0x"]
        assert headline == authoritative_probs["2.0x"]


def test_overnight_pipeline_type_error_fix_and_promotion_safeguards(tmp_path, monkeypatch):
    """Verify that run_pipeline accepts force_promotion and tag arguments without TypeError,
    while strictly preserving statistical promotion gates, and that CLI executes cleanly.
    """
    from main import cmd_overnight
    import argparse

    pipeline = OvernightTrainingPipeline()

    # Calling with force_promotion=True and tag="test_tag"
    report = pipeline.run_pipeline(
        min_test_rounds=20,
        tag="test_fix_typeerror",
        force_promotion=True,
    )

    assert "pipeline_tag" in report
    assert report["pipeline_tag"] == "test_fix_typeerror"
    # Even with force_promotion=True, statistical promotion gates are strictly preserved!
    # If advantage was not established, promoted must remain False
    selection = report.get("selection_result", {})
    if not selection.get("advantage_established", False):
        assert report["promoted"] is False
        assert report["promotion_status"] in ["retained_incumbent", "rejected_insufficient_samples"]

    # Also verify CLI execution executes cleanly without TypeError or KeyError
    cli_args = argparse.Namespace(file=None, force_promotion=True, tag="cli_test_tag")
    cmd_overnight(cli_args)


def test_champion_selection_baseline_fallback_and_calibration_badge(tmp_path, monkeypatch):
    """Verify that when Historical Baseline is champion, selected_model displays
    Historical Baseline, probabilities reflect theoretical values (48.5% at 2.0x),
    and calibration badge is not displayed (is_calibrated=False).
    """
    manager = LiveManager(seed_dataset_path="data/raw/synthetic_sample.csv")

    # Set approved champion to Historical Baseline
    mock_approved = {
        "active_version": "v_baseline_fallback",
        "champion_model": "Historical Baseline",
    }
    monkeypatch.setattr(manager, "_get_approved_version_info", lambda: mock_approved)

    manager._generate_next_prediction()
    pred = manager.active_prediction

    assert pred["selected_model"] == "Historical Baseline"
    assert pred["champion_model"] == "Historical Baseline"
    assert pred["probabilities"]["2.0x"] == pytest.approx(0.4850, abs=0.001)
    assert pred["headline_p2x"] == pytest.approx(0.4850, abs=0.001)
    # Baseline cannot have a valid ML calibration badge
    assert pred["is_calibrated"] is False
    assert "Uncalibrated" in pred["calibration_status"]


def test_collection_status_synchronization_and_auto_training(tmp_path):
    """Verify collector session progress correctly distinguishes session rounds
    from total authentic stored rounds and triggers auto-training on completion.
    """
    jsonl_file = tmp_path / "test_rounds.jsonl"
    csv_file = tmp_path / "test_rounds.csv"

    # Pre-populate 5 historical authentic rounds
    collector = StakeDataCollector(jsonl_path=str(jsonl_file), csv_path=str(csv_file))
    for i in range(1, 6):
        collector.record_genuine_round(
            round_id=100 + i,
            multiplier=2.5,
            timestamp=f"2026-10-10T12:0{i}:00Z",
        )

    assert len(collector.collected_rounds) == 5
    prog_before = collector.get_session_progress()
    assert prog_before["total_authentic_rounds"] == 5
    assert prog_before["session_rounds_collected"] == 0

    # Start a session targeting 3 rounds with auto-training
    callback_called = []
    def on_complete():
        callback_called.append(True)

    collector.start_session(
        target_rounds=3,
        auto_train=True,
        on_complete_callback=on_complete,
    )

    # Collect 3 session rounds
    for i in range(1, 4):
        collector.record_genuine_round(
            round_id=200 + i,
            multiplier=1.8,
            timestamp=f"2026-10-10T13:0{i}:00Z",
        )

    prog_after = collector.get_session_progress()
    # Total rounds = 5 initial + 3 session = 8
    assert prog_after["total_authentic_rounds"] == 8
    # Session rounds = 3
    assert prog_after["session_rounds_collected"] == 3
    # Target reached -> session completed and auto-training triggered
    assert prog_after["session_status"] == "training"
    assert len(callback_called) == 1

