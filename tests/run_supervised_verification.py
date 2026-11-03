"""Supervised Live Collection & Audit Verification Script.

Executes a controlled, supervised collection session through the complete pipeline:
WAITING -> FLYING -> CRASH -> Ledger Resolution -> Research Record -> Frozen JSONL & Audit.
"""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collector.collector import collector
from src.collector.ledger import ledger
from src.collector.live_manager import live_manager
from src.collector.parser import ParsedEvent
from src.collector.verifier import collection_supervisor, CollectionQualitySupervisor


def run_supervised_verification(num_rounds: int = 12):
    print("=====================================================================")
    print("AVIATORSCOPE SUPERVISED COLLECTION & AUDIT VERIFICATION")
    print("=====================================================================")

    # Ensure clean state
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    # 1. Start Supervised Collection Session (auto_train=False strictly enforced)
    progress = collector.start_session(duration_hours=0.25, target_rounds=num_rounds, auto_train=False)
    session_id = progress["session_id"]
    print(f"[*] Started supervised session: {session_id}")
    print(f"[*] Auto-train on completion: {collector.auto_train_on_completion} (strictly disabled)")

    base_round_id = 8048980
    multipliers = [1.45, 2.80, 1.15, 5.20, 1.95, 2.10, 10.45, 1.05, 3.50, 1.88, 4.12, 2.05]

    for i in range(num_rounds):
        cur_id = base_round_id + i
        mult = multipliers[i % len(multipliers)]
        t_base = time.time()
        ts_wait = datetime.now(timezone.utc).isoformat()

        # Step A: Pre-round WAITING Event
        ev_wait = ParsedEvent(
            event_type="stage_change",
            round_id=cur_id,
            stage="waiting",
            timestamp=ts_wait,
        )
        live_manager.process_parsed_event(ev_wait)

        # Inspect pre-round snapshot
        snap_pre = live_manager.get_next_round_prediction_snapshot()
        assert snap_pre["target_round_id"] == cur_id
        assert snap_pre["phase"] == "WAITING"

        time.sleep(0.05)

        # Step B: FLYING Transition Event
        ts_fly = datetime.now(timezone.utc).isoformat()
        ev_fly = ParsedEvent(
            event_type="stage_change",
            round_id=cur_id,
            stage="flying",
            timestamp=ts_fly,
        )
        live_manager.process_parsed_event(ev_fly)

        # Inspect in-flight expired state
        snap_fly = live_manager.get_next_round_prediction_snapshot()
        assert snap_fly["phase"] == "FLYING"
        assert snap_fly["verification_status"] == "EXPIRED"

        time.sleep(0.05)

        # Step C: CRASH Event
        ts_crash = datetime.now(timezone.utc).isoformat()
        ev_crash = ParsedEvent(
            event_type="crash",
            round_id=cur_id,
            multiplier=mult,
            stage="crashed",
            timestamp=ts_crash,
        )
        live_manager.process_parsed_event(ev_crash)

        # Inspect post-crash latest resolved round
        snap_post = live_manager.get_next_round_prediction_snapshot()
        latest_res = snap_post["latest_resolved_round"]
        assert latest_res["round_id"] == cur_id
        assert latest_res["actual_multiplier"] == round(float(mult), 2)

        print(f"  [+] Round #{cur_id} processed: crashed at {mult:.2f}x | Latest resolved round matches!")

    # 2. Finalize / Stop Session and Freeze Snapshot
    print(f"\n[*] Finalizing session '{session_id}' and freezing snapshot...")
    collector.stop_session()
    assert collector.session_status in ("VALIDATED", "COMPLETED")

    frozen_jsonl = Path(collector.frozen_jsonl_path)
    frozen_csv = Path(collector.frozen_dataset_path)
    report_path = Path(f"data/live/sessions/{session_id}_validation_report.json")

    print(f"[*] Frozen JSONL path: {frozen_jsonl}")
    print(f"[*] Frozen CSV path: {frozen_csv}")
    print(f"[*] Session SHA-256: {collector.session_dataset_sha256}")

    # 3. Read and inspect every frozen record
    records = []
    with open(frozen_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"\n[*] Inspecting {len(records)} frozen records:")
    all_records_valid = True
    supervisor = CollectionQualitySupervisor()

    for idx, r in enumerate(records):
        r_id = r["round_id"]
        flying_ts = r.get("round_start_flying_ts")
        pred_mono = r.get("backend_monotonic_prediction_ts")
        flying_mono = r.get("backend_monotonic_flying_ts")
        crash_mono = r.get("backend_monotonic_crash_ts")
        res_hash = r.get("resolution_hash")
        pre_hash = r.get("pre_round_hash")
        probs = r.get("selected_forecast") or r.get("probabilities")

        # Checks
        assert flying_ts is not None, f"Round #{r_id} missing round_start_flying_ts"
        assert pred_mono is not None, f"Round #{r_id} missing backend_monotonic_prediction_ts"
        assert flying_mono is not None, f"Round #{r_id} missing backend_monotonic_flying_ts"
        assert crash_mono is not None, f"Round #{r_id} missing backend_monotonic_crash_ts"
        assert pred_mono <= flying_mono + 1e-4 <= crash_mono + 1e-4, f"Round #{r_id} timing order violation"
        assert res_hash and len(res_hash) == 64, f"Round #{r_id} missing resolution_hash"
        assert pre_hash and len(pre_hash) == 64, f"Round #{r_id} missing pre_round_hash"

        # Check probability monotonicity
        p_vals = [probs[k] for k in ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]]
        for j in range(len(p_vals) - 1):
            assert p_vals[j] >= p_vals[j+1] - 1e-4, f"Round #{r_id} monotonicity violation"

        is_val, v_status, reasons = supervisor.verify_round_record(r, is_live_ingest=False)
        if not is_val:
            all_records_valid = False
            print(f"  [-] Round #{r_id}: FAILED - {reasons}")
        else:
            print(f"  [OK] Round #{r_id} ({r['multiplier']}x): VALID | Pred={pred_mono:.2f} <= Fly={flying_mono:.2f} <= Crash={crash_mono:.2f} | ResHash={res_hash[:12]}...")

    # 4. Read validation report
    with open(report_path, "r", encoding="utf-8") as f:
        val_rep = json.load(f)

    print("\n=====================================================================")
    print("FINAL VALIDATION REPORT SUMMARY:")
    print(f"Status: {val_rep.get('status')}")
    print(f"Is Valid: {val_rep.get('is_valid')}")
    print(f"Total Records: {val_rep.get('total_records')}")
    print(f"Valid Records: {val_rep.get('valid_records')}")
    print(f"Rejected Records: {val_rep.get('rejected_records')}")
    print(f"Duplicate Count: {val_rep.get('duplicate_count')}")
    print(f"Dataset SHA-256: {val_rep.get('dataset_sha256')}")
    print("=====================================================================")

    assert val_rep["is_valid"] is True, "Session must pass full validation"
    assert val_rep["rejected_records"] == 0, "Zero records may be rejected"
    assert val_rep["valid_records"] == num_rounds, f"All {num_rounds} records must be VALID"
    assert all_records_valid is True

    print("\n>>> SUPERVISED LIVE COLLECTION TEST PASSED WITH 100% VALID RECORDS! <<<")
    return val_rep


if __name__ == "__main__":
    run_supervised_verification()
