import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.collector.collector import collector
from src.collector.ledger import ledger
from src.collector.live_manager import live_manager
from src.collector.parser import ParsedEvent
from src.collector.verifier import CollectionQualitySupervisor
from src.data.dataset_aggregator import DatasetAggregator
from src.models.champion_selector import enforce_probability_monotonicity


# =============================================================================
# 1. RETRAINING DATASET SELECTION & AGGREGATION TESTS (Requirement 1)
# =============================================================================
class TestDatasetSelectionAndAggregation:

    def test_aggregator_discovers_approved_historical_and_sessions(self):
        agg = DatasetAggregator()
        sources = agg.discover_eligible_sources()
        paths = [s["path"] for s in sources]
        labels = [s["source"] for s in sources]

        assert any("real_aviator_historical.csv" in p for p in paths), "Approved historical dataset must be included"
        assert "approved_historical" in labels
        assert any("validated_overnight_session" in l for l in labels), "Frozen overnight sessions must be discovered"

    def test_aggregator_excludes_synthetic_and_quarantined_records(self, tmp_path):
        # Create quarantine log
        q_path = tmp_path / "quarantine.jsonl"
        q_entry = {"quarantined_at": "2026-10-13T00:00:00Z", "round_id": 99999, "rejection_reasons": ["Bad round"]}
        q_path.write_text(json.dumps(q_entry) + "\n", encoding="utf-8")

        # Create session with quarantined and synthetic records
        s_dir = tmp_path / "sessions"
        s_dir.mkdir()
        session_file = s_dir / "test_session_frozen.jsonl"
        records = [
            {"round_id": 99999, "multiplier": 2.50, "timestamp": "2026-10-13T01:00:00Z", "source": "browser_observer", "provenance_hashes": {"pre_round_hash": "a"*32, "resolution_hash": "b"*32}}, # Quarantined
            {"round_id": 100001, "multiplier": 1.80, "timestamp": "2026-10-13T01:01:00Z", "source": "synthetic", "provenance_hashes": {"pre_round_hash": "a"*32, "resolution_hash": "b"*32}}, # Synthetic
            {"round_id": 100002, "multiplier": 3.10, "timestamp": "2026-10-13T01:02:00Z", "source": "browser_observer", "provenance_hashes": {"pre_round_hash": "c"*32, "resolution_hash": "d"*32}}, # Valid
        ]
        with open(session_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        agg = DatasetAggregator(
            historical_file=str(tmp_path / "nonexistent.csv"),
            sessions_dir=str(s_dir),
            live_csv_path=str(tmp_path / "nonexistent.csv"),
            live_jsonl_path=str(tmp_path / "nonexistent.jsonl"),
            quarantine_path=str(q_path),
        )
        df, manifest = agg.aggregate_and_manifest()

        assert 99999 not in df["round_id"].values, "Quarantined round must be excluded"
        assert 100001 not in df["round_id"].values, "Synthetic source must be excluded"
        assert 100002 in df["round_id"].values, "Genuine valid round must be included"
        assert manifest["summary"]["total_rejected_rows"] >= 2

    def test_aggregator_detects_and_quarantines_conflicting_duplicates(self, tmp_path):
        s_dir = tmp_path / "sessions"
        s_dir.mkdir()

        # Session 1: Round 55555 crashed at 1.50x
        s1 = s_dir / "s1_frozen.jsonl"
        s1.write_text(json.dumps({
            "round_id": 55555, "multiplier": 1.50, "timestamp": "2026-10-13T01:00:00Z",
            "source": "browser_observer", "provenance_hashes": {"pre_round_hash": "a"*32, "resolution_hash": "b"*32}
        }) + "\n", encoding="utf-8")

        # Session 2: Round 55555 crashed at 4.20x (Discrepancy of 2.70x!)
        s2 = s_dir / "s2_frozen.jsonl"
        s2.write_text(json.dumps({
            "round_id": 55555, "multiplier": 4.20, "timestamp": "2026-10-13T01:00:00Z",
            "source": "browser_observer", "provenance_hashes": {"pre_round_hash": "c"*32, "resolution_hash": "d"*32}
        }) + "\n", encoding="utf-8")

        agg = DatasetAggregator(
            historical_file=str(tmp_path / "empty.csv"),
            sessions_dir=str(s_dir),
            live_csv_path=str(tmp_path / "empty.csv"),
            live_jsonl_path=str(tmp_path / "empty.jsonl"),
            quarantine_path=str(tmp_path / "quarantine.jsonl"),
        )
        df, manifest = agg.aggregate_and_manifest()

        summary = manifest["summary"]
        assert summary["conflicts_detected_count"] == 1, "Conflict must be detected"
        assert len(summary["conflicts"]) == 1
        assert summary["conflicts"][0]["round_id"] == 55555
        assert 55555 not in df["round_id"].values, "Conflicting duplicate record must NOT enter training"

    def test_aggregator_deduplicates_overlapping_matching_records(self, tmp_path):
        s_dir = tmp_path / "sessions"
        s_dir.mkdir()

        # Both sessions record the identical round outcome
        rec = {
            "round_id": 77777, "multiplier": 2.15, "timestamp": "2026-10-13T01:00:00Z",
            "source": "browser_observer", "provenance_hashes": {"pre_round_hash": "a"*32, "resolution_hash": "b"*32}
        }
        (s_dir / "s1_frozen.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
        (s_dir / "s2_frozen.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")

        agg = DatasetAggregator(
            historical_file=str(tmp_path / "empty.csv"),
            sessions_dir=str(s_dir),
            live_csv_path=str(tmp_path / "empty.csv"),
            live_jsonl_path=str(tmp_path / "empty.jsonl"),
            quarantine_path=str(tmp_path / "quarantine.jsonl"),
        )
        df, manifest = agg.aggregate_and_manifest()

        assert len(df) == 1, "Matching duplicate must be cleanly deduplicated"
        assert manifest["summary"]["duplicates_removed_count"] == 1
        assert df.iloc[0]["round_id"] == 77777


# =============================================================================
# 2. PRE-ROUND PREDICTION TIMING & MONOTONICITY TESTS (Requirement 3)
# =============================================================================
class TestPreRoundPredictionTimingAndMonotonicity:

    def test_monotonicity_projection_enforces_ordered_probabilities(self):
        # Non-monotonic input probabilities
        unordered = {
            "1.5x": 0.45,
            "2.0x": 0.55,  # Violates monotonicity: P(2x) > P(1.5x)
            "3.0x": 0.30,
            "5.0x": 0.35,  # Violates monotonicity: P(5x) > P(3x)
            "10.0x": 0.10,
        }
        ordered = enforce_probability_monotonicity(unordered)
        assert ordered["1.5x"] >= ordered["2.0x"]
        assert ordered["2.0x"] >= ordered["3.0x"]
        assert ordered["3.0x"] >= ordered["5.0x"]
        assert ordered["5.0x"] >= ordered["10.0x"]
        for k in ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]:
            assert 0.0 <= ordered[k] <= 1.0

    def test_verifier_accepts_valid_pre_round_record_with_provenance(self):
        supervisor = CollectionQualitySupervisor()
        valid_record = {
            "round_id": 8048999,
            "actual_multiplier": 2.45,
            "source_provenance": "browser_observer",
            "prediction_generation_timestamp": "2026-10-13T10:00:00Z",
            "round_start_flying_ts": "2026-10-13T10:00:05Z",
            "crash_timestamp": "2026-10-13T10:00:15Z",
            "backend_monotonic_prediction_ts": 100.0,
            "backend_monotonic_flying_ts": 105.0,
            "backend_monotonic_crash_ts": 115.0,
            "model_predictions": {
                m: {"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09}
                for m in ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"]
            },
            "selected_forecast": {"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09},
            "pre_round_hash": "a" * 64,
            "input_feature_hash": "b" * 64,
            "resolution_hash": "c" * 64,
        }
        is_valid, status, reasons = supervisor.verify_round_record(valid_record)
        assert is_valid is True
        assert status == "VALID"
        assert len(reasons) == 0

    def test_verifier_rejects_prediction_after_flying(self):
        supervisor = CollectionQualitySupervisor()
        late_record = {
            "round_id": 8049000,
            "actual_multiplier": 1.75,
            "source_provenance": "browser_observer",
            # Prediction generated AFTER flying started
            "prediction_generation_timestamp": "2026-10-13T10:00:06Z",
            "round_start_flying_ts": "2026-10-13T10:00:05Z",
            "crash_timestamp": "2026-10-13T10:00:15Z",
            "backend_monotonic_prediction_ts": 106.0,
            "backend_monotonic_flying_ts": 105.0,
            "backend_monotonic_crash_ts": 115.0,
            "model_predictions": {
                m: {"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09}
                for m in ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"]
            },
            "selected_forecast": {"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09},
            "pre_round_hash": "a" * 64,
            "input_feature_hash": "b" * 64,
            "resolution_hash": "c" * 64,
        }
        is_valid, status, reasons = supervisor.verify_round_record(late_record)
        assert is_valid is False
        assert status == "REJECTED"
        assert any("AFTER flying" in r for r in reasons)

    def test_verifier_rejects_missing_provenance_hashes(self):
        supervisor = CollectionQualitySupervisor()
        unhashed_record = {
            "round_id": 8049001,
            "actual_multiplier": 3.00,
            "source_provenance": "browser_observer",
            "prediction_generation_timestamp": "2026-10-13T10:00:00Z",
            "round_start_flying_ts": "2026-10-13T10:00:05Z",
            "crash_timestamp": "2026-10-13T10:00:15Z",
            "model_predictions": {
                m: {"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09}
                for m in ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"]
            },
            "selected_forecast": {"1.5x": 0.65, "2.0x": 0.48, "3.0x": 0.32, "5.0x": 0.19, "10.0x": 0.09},
            "pre_round_hash": "",
            "input_feature_hash": "",
            "resolution_hash": "",
        }
        is_valid, status, reasons = supervisor.verify_round_record(unhashed_record)
        assert is_valid is False
        assert status == "REJECTED"
        assert any("pre_round_hash" in r for r in reasons)

    def test_historical_session_1791910025_is_quarantined_due_to_empty_hashes_and_monotonicity(self):
        p = Path("data/live/sessions/stake_session_1791910025_9c6fab79_frozen.jsonl")
        if p.exists():
            supervisor = CollectionQualitySupervisor()
            res = supervisor.audit_dataset(p)
            assert res["is_valid"] is False, "Session with empty hashes and non-monotonic neural predictions must fail audit"
            assert res["status"] == "FAILED"
            assert res["rejected_records"] == 11
            assert res["valid_records"] == 0


# =============================================================================
# 3. NEXT ROUND PREDICTION PAGE TESTS (Requirement 4)
# =============================================================================
class TestNextRoundPredictionPage:

    def test_next_round_snapshot_fallback_when_disconnected(self):
        live_manager.is_connected = False
        live_manager.current_stage = "idle"
        snap = live_manager.get_next_round_prediction_snapshot()

        assert snap["is_valid"] is False
        assert snap["fallback_message"] == "No valid pre-round prediction available"
        assert snap["display_message"] == "No valid pre-round prediction available"
        assert snap["phase"] == "unavailable"
        assert snap["probabilities"] is None

    def test_next_round_snapshot_verified_during_waiting(self):
        live_manager.is_connected = True
        live_manager.source_type = "browser_observer"
        live_manager.observer_frames_received = 10
        live_manager.game_socket_status = "connected"
        live_manager.current_stage = "waiting"
        live_manager.active_prediction_expired = False

        snap = live_manager.get_next_round_prediction_snapshot()

        assert snap["is_valid"] is True
        assert snap["verification_status"] == "VERIFIED"
        assert snap["phase"] == "WAITING"
        assert snap["champion_model"] is not None
        assert snap["model_version"] is not None
        assert snap["probabilities"] is not None

        probs = snap["probabilities"]
        assert probs["1.5x"] >= probs["2.0x"] >= probs["3.0x"] >= probs["5.0x"] >= probs["10.0x"]
        assert len(snap["provenance_hashes"]["pre_round_hash"]) >= 16
        assert len(snap["provenance_hashes"]["input_feature_hash"]) >= 16

    def test_next_round_snapshot_expires_upon_flying_transition(self):
        live_manager.is_connected = True
        live_manager.source_type = "browser_observer"
        live_manager.observer_frames_received = 10
        live_manager.game_socket_status = "connected"
        live_manager.current_stage = "waiting"
        live_manager.active_prediction_expired = False

        # Transition to flying
        ev = ParsedEvent(event_type="stage_change", stage="flying", round_id=live_manager.current_round_id)
        live_manager.process_parsed_event(ev)

        snap = live_manager.get_next_round_prediction_snapshot()
        assert snap["is_valid"] is False
        assert snap["phase"] == "FLYING"
        assert snap["verification_status"] == "EXPIRED"
        assert snap["fallback_message"] == "No valid pre-round prediction available"
        assert snap["display_message"] == "No valid pre-round prediction available"


# =============================================================================
# 4. STAKE VS SPRIBE WEBSOCKET DIAGNOSTICS (Requirement 5)
# =============================================================================
class TestStakeVsSpribeSocketReliability:

    @pytest.mark.asyncio
    async def test_distinguish_stake_platform_from_spribe_game_socket(self):
        # Ingest Stake platform socket error
        await live_manager.ingest_raw({
            "type": "socket_status_change",
            "url": "wss://stake.com/_api/websockets",
            "status": "error",
            "category": "stake_platform",
        })
        # Stake platform error should NOT mark game socket disconnected
        assert live_manager.platform_socket_status == "error"
        assert live_manager.game_socket_status != "error"

        # Ingest Spribe game socket detection
        await live_manager.ingest_raw({
            "type": "socket_detected",
            "url": "wss://aviator-next.spribegaming.com/websocket",
            "category": "spribe_game",
            "ready_state": 1,
        })
        assert live_manager.game_socket_status == "connected"

    @pytest.mark.asyncio
    async def test_duplicate_raw_frame_increment_counter(self):
        orig_count = live_manager.duplicate_frames_detected_count
        payload = '{"type":"game_frame","raw_frame":"duplicate_test_payload_12345"}'

        # First ingest
        await live_manager.ingest_raw(payload)
        # Duplicate ingest immediately
        await live_manager.ingest_raw(payload)

        assert live_manager.duplicate_frames_detected_count >= orig_count + 1


# =============================================================================
# 5. RESOLUTION HASH & FLYING TIMESTAMPS REGRESSION TESTS (Requirement 3 & 4)
# =============================================================================
class TestResolutionHashAndTimingRegression:

    def test_full_round_lifecycle_records_flying_ts_and_resolution_hash(self):
        """Test waiting -> flying -> crash produces valid flying timestamps and resolution hash."""
        live_manager.source_type = "browser_observer"
        test_round_id = 9600000 + int(time.time() * 1000) % 1000000

        now_iso = datetime.now(timezone.utc).isoformat()
        # 1. Waiting event
        ev_wait = ParsedEvent(
            event_type="stage_change",
            round_id=test_round_id,
            stage="waiting",
            timestamp=now_iso,
        )
        live_manager.process_parsed_event(ev_wait)

        # 2. Flying event
        time.sleep(0.01)
        ev_fly = ParsedEvent(
            event_type="stage_change",
            round_id=test_round_id,
            stage="flying",
            timestamp=now_iso,
        )
        live_manager.process_parsed_event(ev_fly)

        # 3. Crash event
        time.sleep(0.01)
        ev_crash = ParsedEvent(
            event_type="crash",
            round_id=test_round_id,
            multiplier=3.14,
            stage="crashed",
            timestamp=now_iso,
        )
        live_manager.process_parsed_event(ev_crash)

        rec = collector.collected_rounds[-1]
        assert rec["round_id"] == test_round_id
        assert rec["round_start_flying_ts"] is not None
        assert rec["backend_monotonic_flying_ts"] is not None
        assert rec["resolution_hash"] is not None
        assert len(rec["resolution_hash"]) == 64
        assert rec["provenance_hashes"]["resolution_hash"] == rec["resolution_hash"]

        # Verify strict monotonic ordering
        pred_mono = rec["backend_monotonic_prediction_ts"]
        flying_mono = rec["backend_monotonic_flying_ts"]
        crash_mono = rec["backend_monotonic_crash_ts"]
        assert pred_mono < flying_mono, f"Prediction {pred_mono} must strictly precede flying {flying_mono}"
        assert flying_mono < crash_mono, f"Flying {flying_mono} must strictly precede crash {crash_mono}"

        # Verify quality supervisor validates the record as frozen
        supervisor = CollectionQualitySupervisor()
        is_val, status, reasons = supervisor.verify_round_record(rec, is_live_ingest=False)
        assert is_val is True
        assert status == "VALID"
        assert reasons == []

    def test_user_session_1791914191_e0c7f2c7_remains_rejected(self):
        """Confirm that user's historical unhashed session remains REJECTED by supervisor."""
        session_path = Path("data/live/sessions/stake_session_1791914191_e0c7f2c7_frozen.jsonl")
        assert session_path.exists(), "User session file must exist for verification"

        supervisor = CollectionQualitySupervisor()
        report = supervisor.audit_dataset(session_path)
        assert report["is_valid"] is False
        assert report["status"] == "FAILED"
        assert report["rejected_records"] == 11
        assert report["valid_records"] == 0
        assert any("resolution_hash" in r for sample in report["rejection_sample"] for r in sample["reasons"])

    def test_next_round_snapshot_contains_latest_resolved_round_and_diagnostics(self):
        """Verify get_next_round_prediction_snapshot includes latest resolved round and diagnostics."""
        snap = live_manager.get_next_round_prediction_snapshot()
        assert "latest_resolved_round" in snap
        assert "socket_diagnostics" in snap

        latest_res = snap["latest_resolved_round"]
        assert latest_res is not None
        assert "actual_multiplier" in latest_res
        assert "round_id" in latest_res
        assert "resolution_timestamp" in latest_res

        diag = snap["socket_diagnostics"]
        assert "game_socket_status" in diag
        assert "platform_socket_status" in diag
        assert "connection_gaps_count" in diag

    def test_resolution_hash_recomputation_from_genuine_resolution_data(self):
        """Verify that resolution_hash is deterministically recomputable from genuine resolution fields."""
        from src.collector.ledger import compute_sha256
        import numpy as np

        test_round_id = 9700000 + int(time.time() * 1000) % 1000000
        mult = 2.45
        crash_ts = datetime.now(timezone.utc).isoformat()
        pre_hash = "f" * 64

        # Verify ledger handles numpy arrays without TypeError: Object of type ndarray is not JSON serializable
        res_entry = ledger.record_resolution(
            round_id=test_round_id,
            actual_multiplier=mult,
            crash_timestamp=crash_ts,
            fallback_pre_forecast={
                "source_provenance": "browser_observer",
                "pre_round_hash": pre_hash,
                "model_predictions": {
                    "Logistic Regression": {"1.5x": np.float64(0.70), "2.0x": np.float64(0.55), "3.0x": 0.35, "5.0x": 0.20, "10.0x": 0.10},
                    "MLP": {"1.5x": np.array([0.68])[0], "2.0x": np.array([0.52])[0], "3.0x": 0.33, "5.0x": 0.18, "10.0x": 0.08},
                },
                "ensemble_weights": {"Logistic Regression": np.float64(0.5), "MLP": np.float64(0.5)},
                "probabilities": {"1.5x": 0.69, "2.0x": 0.53, "3.0x": 0.34, "5.0x": 0.19, "10.0x": 0.09},
                "champion_model": "MLP",
                "model_version": "v_cli_parity_test",
                "input_feature_hash": "e" * 64,
            },
        )
        assert res_entry is not None
        stored_hash = res_entry["resolution_hash"]
        assert len(stored_hash) == 64

        # Recompute directly from genuine res_entry resolution dictionary
        res_dict = {k: v for k, v in res_entry["resolution"].items() if k != "resolution_hash"}
        recomputed_hash = compute_sha256(res_dict)
        assert stored_hash == recomputed_hash, f"Stored {stored_hash} must match recomputed {recomputed_hash}"

    def test_spribe_raw_frame_authoritative_timestamp_extraction(self):
        """Verify that parser extracts authoritative serverTime from Spribe SFS2X binary payloads."""
        import base64
        from src.collector.parser import AviatorEventParser
        parser = AviatorEventParser()

        # Raw Spribe changeState SFS2X binary payload with serverTime: 1791914716159 (18:05:16.159 UTC)
        # and timeLeft: 5000 (5.0s pre-round human betting countdown)
        raw_b64 = "gACTEgADAAFwEgACAAFwEgAGAApuZXdTdGF0ZUlkBAAAAAEABGNvZGUEAAAAyAAPYmV0U3RhdGVFbmRUaW1lBQAAAaE2cL+HAApzZXJ2ZXJUaW1lBQAAAaE2cKv/AAdyb3VuZElkBAB60OUACHRpbWVMZWZ0BAAAE4gAAWMIAAtjaGFuZ2VTdGF0ZQABYQMADQABYwIB"
        raw_bytes = base64.b64decode(raw_b64)

        events = parser.parse_frame(raw_bytes)
        assert len(events) >= 1
        ev = events[0]
        assert ev.event_type == "stage_change"
        assert ev.stage == "waiting"
        assert ev.round_id == 8048869
        # Server time 1791914716159 ms must normalize to 2026-10-13T18:05:16.159000+00:00
        assert ev.timestamp.startswith("2026-10-13T18:05:16")

    def test_timing_semantics_and_burst_event_distinction(self):
        """Verify event ordering and that batch arrival times maintain monotonic causality."""
        test_r = 9800000 + int(time.time() * 1000) % 1000000
        live_manager.source_type = "browser_observer"
        live_manager.is_connected = True

        now_iso = datetime.now(timezone.utc).isoformat()
        ev_w = ParsedEvent(event_type="stage_change", round_id=test_r, stage="waiting", timestamp=now_iso)
        live_manager.process_parsed_event(ev_w)
        snap_w = live_manager.get_next_round_prediction_snapshot()
        assert snap_w["phase"] == "WAITING"

        # Even with compressed burst delivery (<2ms apart, e.g. batch buffer)
        ev_f = ParsedEvent(event_type="stage_change", round_id=test_r, stage="flying", timestamp=now_iso)
        live_manager.process_parsed_event(ev_f)
        snap_f = live_manager.get_next_round_prediction_snapshot()
        assert snap_f["phase"] == "FLYING"
        assert snap_f["verification_status"] == "EXPIRED"

        ev_c = ParsedEvent(event_type="crash", round_id=test_r, multiplier=1.85, stage="crashed", timestamp=now_iso)
        live_manager.process_parsed_event(ev_c)

        rec = collector.collected_rounds[-1]
        assert rec["round_id"] == test_r
        # Monotonic ordering is strictly preserved with strict inequality
        assert rec["backend_monotonic_prediction_ts"] < rec["backend_monotonic_flying_ts"]
        assert rec["backend_monotonic_flying_ts"] < rec["backend_monotonic_crash_ts"]


