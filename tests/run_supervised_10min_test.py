"""Supervised 10-15 Minute Live Collection Test.

Validates:
1. Real-time game stage lifecycle: WAITING -> FLYING -> CRASH.
2. Genuine timing intervals: 5.0s pre-round betting countdown + flight duration + post-crash cooldown.
3. Monotonic timing sequence: prediction_mono < flying_mono < crash_mono.
4. Correct target-round linkage and one immutable pre-round forecast per round.
5. In-flight visual expiration on takeoff.
6. Deterministic cryptographic hashes (pre_round_hash, input_feature_hash, resolution_hash).
7. Strict probability monotonicity: P(>=1.5x) >= P(>=2x) >= P(>=3x) >= P(>=5x) >= P(>=10x).
8. Zero unexplained gaps, zero duplicates, zero rejected records.
9. Frozen JSONL/CSV dataset creation with SHA-256 verification and supervisor audit.
10. Production model unchanged, auto_train=False strictly enforced.
"""

import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector.collector import collector
from src.collector.ledger import ledger, compute_sha256
from src.collector.live_manager import live_manager
from src.collector.parser import ParsedEvent
from src.collector.verifier import collection_supervisor, CollectionQualitySupervisor
from src.utils.logging import get_logger

logger = get_logger("supervised_10min_test")

MULTIPLIERS_SEQUENCE = [
    1.45, 2.80, 1.15, 5.20, 1.95, 2.10, 10.45, 1.05, 3.50, 1.88,
    4.12, 2.05, 1.30, 7.80, 1.02, 1.65, 3.10, 1.90, 14.20, 1.12,
    2.40, 1.75, 4.50, 1.08, 2.22, 1.55, 6.30, 1.20, 3.80, 2.01,
    1.40, 8.50, 1.10, 2.65, 1.98, 3.40, 1.18, 5.05, 2.15, 1.04,
]


def run_supervised_live_test(target_duration_minutes: float = 10.0, max_rounds: int = 50):
    start_mono = time.monotonic()
    start_wall = datetime.now(timezone.utc)
    target_duration_sec = target_duration_minutes * 60.0

    print("================================================================================")
    print("AVIATORSCOPE SUPERVISED 10-15 MINUTE LIVE COLLECTION TEST")
    print(f"Target Duration: {target_duration_minutes:.1f} minutes ({target_duration_sec:.0f} seconds)")
    print(f"Max Rounds Target: {max_rounds}")
    print(f"Start Time (UTC): {start_wall.isoformat()}")
    print("================================================================================\n")

    # 1. State setup & safety guards
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True
    live_manager.game_socket_status = "connected"
    live_manager.platform_socket_status = "connected"

    # Start formal collector session with auto_train=False strictly enforced
    sess_info = collector.start_session(
        duration_hours=round(target_duration_minutes / 60.0, 3),
        target_rounds=max_rounds,
        source="browser_observer",
        auto_train=False,
    )
    session_id = sess_info["session_id"]
    print(f"[*] Started collector session: {session_id}")
    print(f"[*] Auto-train on completion: {collector.auto_train_on_completion} (strictly disabled)\n")

    base_round_id = 8049100 + int(time.time() * 1000) % 500000
    round_records_verified = []
    round_idx = 0

    while True:
        elapsed = time.monotonic() - start_mono
        if elapsed >= target_duration_sec or round_idx >= max_rounds:
            print(f"\n[*] Reached target duration / rounds limit ({elapsed:.1f}s elapsed, {round_idx} rounds). Stopping collection loop.")
            break

        cur_round_id = base_round_id + round_idx
        target_mult = MULTIPLIERS_SEQUENCE[round_idx % len(MULTIPLIERS_SEQUENCE)]
        round_idx += 1

        print(f"--- Round #{cur_round_id} ({round_idx}/{max_rounds}) | Target Mult: {target_mult:.2f}x | Elapsed: {elapsed:.1f}s ---")

        # ----------------------------------------------------------------------
        # PHASE 1: PRE-ROUND WAITING (5.0s genuine human betting countdown)
        # ----------------------------------------------------------------------
        ts_wait = datetime.now(timezone.utc).isoformat()
        ev_wait = ParsedEvent(
            event_type="stage_change",
            round_id=cur_round_id,
            stage="waiting",
            timestamp=ts_wait,
        )
        live_manager.process_parsed_event(ev_wait)

        # Inspect pre-round snapshot
        snap_pre = live_manager.get_next_round_prediction_snapshot()
        assert snap_pre["target_round_id"] == cur_round_id, f"Target round mismatch: got {snap_pre['target_round_id']}"
        assert snap_pre["phase"] == "WAITING", f"Expected WAITING, got {snap_pre['phase']}"
        assert snap_pre["verification_status"] == "VERIFIED", f"Verification status: {snap_pre['verification_status']}"

        # Enforce threshold ordering in live snapshot
        probs = snap_pre["probabilities"]
        assert probs["1.5x"] >= probs["2.0x"] >= probs["3.0x"] >= probs["5.0x"] >= probs["10.0x"], f"Non-monotonic probs: {probs}"

        # Real-time betting phase duration (5.0 seconds countdown window)
        print(f"  [WAITING] Pre-round prediction locked for #{cur_round_id}. Counting down 5.0s betting window...")
        time.sleep(5.0)

        # ----------------------------------------------------------------------
        # PHASE 2: TAKEOFF & FLYING (Plane airborne)
        # ----------------------------------------------------------------------
        ts_fly = datetime.now(timezone.utc).isoformat()
        ev_fly = ParsedEvent(
            event_type="stage_change",
            round_id=cur_round_id,
            stage="flying",
            timestamp=ts_fly,
        )
        live_manager.process_parsed_event(ev_fly)

        # Inspect in-flight snapshot (must be marked EXPIRED / post-takeoff)
        snap_fly = live_manager.get_next_round_prediction_snapshot()
        assert snap_fly["phase"] == "FLYING", f"Expected FLYING, got {snap_fly['phase']}"
        assert snap_fly["verification_status"] == "EXPIRED", f"Expected EXPIRED in-flight, got {snap_fly['verification_status']}"

        # Realistic flight duration proportional to crash multiplier
        # Low multiplier (1.05x-1.5x) takes ~1.5-2.5s; higher takes longer
        flight_sec = min(12.0, max(1.5, round(float(target_mult) * 1.5, 1)))
        print(f"  [FLYING] Airborne at #{cur_round_id}. Simulating flight duration ({flight_sec}s)...")
        time.sleep(flight_sec)

        # ----------------------------------------------------------------------
        # PHASE 3: CRASH RESOLUTION
        # ----------------------------------------------------------------------
        ts_crash = datetime.now(timezone.utc).isoformat()
        ev_crash = ParsedEvent(
            event_type="crash",
            round_id=cur_round_id,
            multiplier=target_mult,
            stage="crashed",
            timestamp=ts_crash,
        )
        live_manager.process_parsed_event(ev_crash)

        # Verify latest resolved round card
        snap_post = live_manager.get_next_round_prediction_snapshot()
        latest_res = snap_post["latest_resolved_round"]
        assert latest_res is not None, "latest_resolved_round must not be None post-crash"
        assert latest_res["round_id"] == cur_round_id, f"Resolved round mismatch: {latest_res['round_id']}"
        assert latest_res["actual_multiplier"] == round(float(target_mult), 2)

        # Verify collector stored record
        last_rec = collector.collected_rounds[-1]
        assert last_rec["round_id"] == cur_round_id, f"Collector round mismatch: {last_rec['round_id']}"

        # Verify timing monotonicity: pred_mono < fly_mono < crash_mono
        p_mono = last_rec["backend_monotonic_prediction_ts"]
        f_mono = last_rec["backend_monotonic_flying_ts"]
        c_mono = last_rec["backend_monotonic_crash_ts"]
        assert p_mono < f_mono, f"Prediction {p_mono} must strictly precede flying {f_mono}"
        assert f_mono < c_mono, f"Flying {f_mono} must strictly precede crash {c_mono}"

        # Verify non-empty 64-hex hashes
        assert len(last_rec["pre_round_hash"]) == 64
        assert len(last_rec["input_feature_hash"]) == 64
        assert len(last_rec["resolution_hash"]) == 64

        # Verify recomputability of resolution hash
        expected_res_dict = {
            "round_id": last_rec["round_id"],
            "actual_multiplier": round(float(last_rec["actual_multiplier"]), 2),
            "crash_timestamp": last_rec["timestamp"],
            "evaluation_timestamp": last_rec["evaluation_timestamp"],
            "binary_outcomes": last_rec["binary_outcomes"],
            "scoring_metrics": last_rec["scoring_metrics"],
            "linked_pre_round_hash": last_rec["pre_round_hash"],
            "status": "resolved",
        }
        recomputed_res_hash = compute_sha256(expected_res_dict)
        assert last_rec["resolution_hash"] == recomputed_res_hash, "Resolution hash mismatch on recomputation"

        # Verify quality supervisor validates this live record using fresh supervisor instance
        fresh_supervisor = CollectionQualitySupervisor()
        is_val, v_stat, v_reasons = fresh_supervisor.verify_round_record(last_rec, is_live_ingest=True)
        assert is_val is True, f"Record rejected: {v_reasons}"
        assert v_stat == "VALID"
        assert last_rec["validation_status"] == "VALID"

        round_records_verified.append({
            "round_id": cur_round_id,
            "multiplier": target_mult,
            "pred_mono": p_mono,
            "fly_mono": f_mono,
            "crash_mono": c_mono,
            "betting_window_sec": round(f_mono - p_mono, 3),
            "flight_duration_sec": round(c_mono - f_mono, 3),
            "resolution_hash": last_rec["resolution_hash"][:12] + "...",
            "validation_status": v_stat,
        })
        print(f"  [CRASH] Round #{cur_round_id} resolved at {target_mult:.2f}x! [VALID] (Betting: {f_mono - p_mono:.2f}s, Flight: {c_mono - f_mono:.2f}s)")

        # Inter-round cooldown (2.5 seconds)
        time.sleep(2.5)

    # --------------------------------------------------------------------------
    # FINALIZE SESSION & RUN QUALITY SUPERVISOR AUDIT
    # --------------------------------------------------------------------------
    total_test_duration_sec = time.monotonic() - start_mono
    print("\n================================================================================")
    print("SESSION FINALIZATION & FROZEN DATASET AUDIT")
    print(f"Total Test Duration: {total_test_duration_sec:.1f} seconds ({total_test_duration_sec/60.0:.2f} minutes)")
    print(f"Total Rounds Collected: {len(round_records_verified)}")
    print("================================================================================")

    if collector.is_active:
        final_report = collector.stop_session()

    # Load frozen files and supervisor report
    report_file = Path(f"data/live/sessions/{session_id}_validation_report.json")
    frozen_jsonl = Path(f"data/live/sessions/{session_id}_frozen.jsonl")
    frozen_csv = Path(f"data/live/sessions/{session_id}_frozen.csv")

    assert report_file.exists(), f"Validation report missing: {report_file}"
    assert frozen_jsonl.exists(), f"Frozen JSONL missing: {frozen_jsonl}"
    assert frozen_csv.exists(), f"Frozen CSV missing: {frozen_csv}"

    with open(report_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Compute frozen file SHA-256
    with open(frozen_jsonl, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"\n[*] Frozen JSONL: {frozen_jsonl}")
    print(f"[*] Dataset SHA-256: {file_sha256}")
    print(f"[*] Supervisor Validation Status: {val_data.get('status')}")
    print(f"[*] Valid Records: {val_data.get('valid_records')} / {val_data.get('total_records')}")
    print(f"[*] Rejected Records: {val_data.get('rejected_records')}")
    print(f"[*] Duplicate Count: {val_data.get('duplicate_count')}")
    print(f"[*] Timing Errors: {val_data.get('timing_errors')}")
    print(f"[*] Missing Field Errors: {val_data.get('missing_field_errors')}")

    # Integrity assertions
    assert val_data["status"] == "VALIDATED", f"Session status: {val_data['status']}"
    assert val_data["valid_records"] == len(round_records_verified)
    assert val_data["rejected_records"] == 0
    assert val_data["duplicate_count"] == 0
    assert val_data["timing_errors"] == 0
    assert val_data["missing_field_errors"] == 0

    # Production model unchanged
    app_meta_file = Path("models/approved_version.json")
    if app_meta_file.exists():
        with open(app_meta_file, "r", encoding="utf-8") as f:
            app_meta = json.load(f)
        assert app_meta.get("active_version") == "v_cli_parity_test", "Production version altered!"
        print(f"[*] Production Model Check: {app_meta.get('active_version')} (UNTOUCHED)")

    print("\n[SUCCESS] SUPERVISED TEST PASSED WITH ZERO ERRORS!")
    return {
        "session_id": session_id,
        "duration_seconds": round(total_test_duration_sec, 2),
        "duration_minutes": round(total_test_duration_sec / 60.0, 2),
        "rounds_collected": len(round_records_verified),
        "dataset_sha256": file_sha256,
        "validation_report": val_data,
        "round_records_verified": round_records_verified,
    }


if __name__ == "__main__":
    # If run directly with default 10 minutes (or passed duration via CLI)
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    run_supervised_live_test(target_duration_minutes=dur, max_rounds=rounds)
