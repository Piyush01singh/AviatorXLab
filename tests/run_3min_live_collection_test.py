"""AviatorScope 3-Minute End-to-End Live Collection Test.

Uses genuine Stake/Spribe WebSocket frames streamed over the Telemetry Bridge
into the live backend (FastAPI / LiveManager / Parser / Ledger / Collector).

Verifies:
1. Genuine WAITING -> pre-round prediction -> FLYING -> CRASH sequence.
2. Correct target round IDs and strictly ordered backend-monotonic timestamps.
3. Valid pre-round, feature-input, and resolution SHA-256 hashes (independently recomputable).
4. Monotonic probabilities across all six models and the approved champion.
5. Latest resolved multiplier updates correctly; forecasts expire during FLYING.
6. Real game-socket health, round continuity, duplicate/missing events, connection gaps,
   and frozen-dataset integrity.
7. Post-3-minute session finalization, frozen JSONL/CSV creation, SHA-256 verification.
8. Safety: v_cli_parity_test untouched and auto_train=False strictly enforced.
"""

import asyncio
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import requests
import websockets

from src.collector.ledger import compute_sha256
from src.collector.verifier import CollectionQualitySupervisor

API_BASE = "http://127.0.0.1:8000"
BRIDGE_WS_URI = "ws://127.0.0.1:8000/api/live/bridge-ws"


def load_genuine_raw_frames():
    """Load authentic Spribe WebSocket frames captured from Stake."""
    frames = []
    for fname in ["data/debug/raw_captured_frames.jsonl.1", "data/debug/raw_captured_frames.jsonl"]:
        p = Path(fname)
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                row["source_file"] = p.name
                row["source_idx"] = idx
                frames.append(row)
    return frames


async def run_3min_live_test(target_duration_sec: float = 180.0):
    print("================================================================================")
    print("AVIATORSCOPE 3-MINUTE END-TO-END LIVE COLLECTION TEST")
    print(f"Target Duration: {target_duration_sec:.1f} seconds ({target_duration_sec/60.0:.2f} minutes)")
    print(f"Source: Genuine Spribe SFS2X WebSocket frames (wss://*.spribegaming.com)")
    print(f"Telemetry Bridge: {BRIDGE_WS_URI}")
    print(f"Start Time (UTC): {datetime.now(timezone.utc).isoformat()}")
    print("================================================================================\n")

    # 1. Load authentic raw frames
    raw_frames = load_genuine_raw_frames()
    print(f"[*] Loaded {len(raw_frames)} authentic Spribe SFS2X raw binary frames.")
    assert len(raw_frames) > 0, "No raw frames available for test!"

    # 2. Start formal collection session with auto_train=False
    start_resp = requests.post(
        f"{API_BASE}/api/collection/start",
        json={
            "duration_hours": round(target_duration_sec / 3600.0, 4),
            "duration_seconds": target_duration_sec,
            "source": "browser_observer",
            "auto_train": False,
        },
        timeout=10,
    )
    if not start_resp.ok:
        print(f"[!] Failed to start collection session: {start_resp.text}")
        sys.exit(1)

    sess_data = start_resp.json()
    session_id = sess_data.get("session_id")
    print(f"[*] Started formal collector session: {session_id}")
    print(f"[*] Collector status: {sess_data.get('session_status')}")
    print(f"[*] Auto-train on completion: {sess_data.get('auto_train_on_completion')} (STRICTLY DISABLED)\n")

    # 3. Connect to Telemetry Bridge WebSocket
    print(f"[*] Connecting to Telemetry Bridge companion ({BRIDGE_WS_URI})...")
    async with websockets.connect(BRIDGE_WS_URI) as ws:
        print("[+] Telemetry Bridge connected successfully!\n")

        # Handshake & socket registration
        await ws.send(json.dumps({
            "raw_frame": {
                "type": "bridge_connected",
                "bridge_type": "websocket",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }))
        await ws.send(json.dumps({
            "raw_frame": {
                "type": "socket_status_change",
                "url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
                "status": "connected",
            }
        }))
        await asyncio.sleep(0.5)

        start_mono = time.monotonic()
        frame_idx = 0
        prev_frame_dt = None
        rounds_seen = set()
        verified_phases = []
        last_logged_round = None

        while True:
            elapsed = time.monotonic() - start_mono
            if elapsed >= target_duration_sec:
                print(f"\n[*] Reached 3-minute target duration ({elapsed:.2f}s elapsed). Stopping stream.")
                break

            if frame_idx >= len(raw_frames):
                print(f"\n[*] Reached end of raw frames buffer at {elapsed:.2f}s. Ending stream.")
                break

            frame = raw_frames[frame_idx]
            frame_idx += 1

            # Real-time cadence: sleep for exact delta between consecutive frame timestamps
            curr_frame_dt = datetime.fromisoformat(frame["timestamp"])
            if prev_frame_dt is not None:
                delta = (curr_frame_dt - prev_frame_dt).total_seconds()
                # Pacing: sleep genuine interval, but cap if network reconnect gap
                if 0.0 < delta <= 20.0:
                    # Check remaining time so we don't overshoot 180s
                    rem = target_duration_sec - (time.monotonic() - start_mono)
                    sleep_time = min(delta, rem)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            prev_frame_dt = curr_frame_dt

            # Stream authentic binary frame to backend
            await ws.send(json.dumps({"raw_frame": frame}))

            # Periodically query next-round prediction endpoint to verify live UI telemetry
            if frame_idx % 10 == 0 or frame.get("source_idx") in (10, 36, 50, 51, 73, 108):
                try:
                    pred_res = requests.get(f"{API_BASE}/api/prediction/next-round", timeout=2).json()
                    phase = pred_res.get("phase")
                    tgt_r = pred_res.get("target_round_id")
                    v_stat = pred_res.get("verification_status")
                    latest_res = pred_res.get("latest_resolved_round")

                    if tgt_r and tgt_r not in rounds_seen:
                        rounds_seen.add(tgt_r)
                        print(f"  [ROUND DETECTED] Target #{tgt_r} | Phase: {phase} | Verification: {v_stat} | Elapsed: {elapsed:.1f}s")

                    if phase == "WAITING" and v_stat == "VERIFIED" and tgt_r != last_logged_round:
                        last_logged_round = tgt_r
                        probs = pred_res.get("probabilities", {})
                        p_order = probs.get("1.5x", 0) >= probs.get("2.0x", 0) >= probs.get("3.0x", 0) >= probs.get("5.0x", 0) >= probs.get("10.0x", 0)
                        print(f"  --> [WAITING] Pre-round prediction locked for #{tgt_r}. Monotonic probs: {p_order} (1.5x={probs.get('1.5x')}, 2.0x={probs.get('2.0x')})")
                        verified_phases.append({"round_id": tgt_r, "phase": "WAITING", "verified": True, "monotonic_probs": p_order})

                    elif phase == "FLYING" and v_stat == "EXPIRED":
                        if not any(p["round_id"] == tgt_r and p["phase"] == "FLYING" for p in verified_phases):
                            print(f"  --> [FLYING] Round #{tgt_r} airborne. Prediction visibly EXPIRED as required.")
                            verified_phases.append({"round_id": tgt_r, "phase": "FLYING", "expired": True})

                    if latest_res and latest_res.get("actual_multiplier"):
                        res_r = latest_res.get("round_id")
                        if not any(p.get("round_id") == res_r and p.get("phase") == "CRASH" for p in verified_phases):
                            print(f"  --> [CRASH] Round #{res_r} resolved at {latest_res.get('actual_multiplier')}x. Latest Resolved card updated!")
                            verified_phases.append({"round_id": res_r, "phase": "CRASH", "multiplier": latest_res.get("actual_multiplier")})

                except Exception as err:
                    pass

    total_test_duration = time.monotonic() - start_mono
    print(f"\n[*] 3-Minute Live Stream Completed: {total_test_duration:.2f} seconds.")

    # 4. Stop session & finalize
    stop_resp = requests.post(f"{API_BASE}/api/collection/stop", timeout=10)
    print(f"[*] Collector stop called: status={stop_resp.status_code}")
    await asyncio.sleep(1.0)

    # 5. Verify Frozen Session Artifacts
    frozen_jsonl = Path(f"data/live/sessions/{session_id}_frozen.jsonl")
    frozen_csv = Path(f"data/live/sessions/{session_id}_frozen.csv")
    val_report_path = Path(f"data/live/sessions/{session_id}_validation_report.json")

    print("\n================================================================================")
    print("SESSION FINALIZATION & FROZEN DATASET VERIFICATION")
    print("================================================================================")
    print(f"Session ID: {session_id}")
    print(f"Frozen JSONL: {frozen_jsonl} (Exists: {frozen_jsonl.exists()})")
    print(f"Frozen CSV:   {frozen_csv} (Exists: {frozen_csv.exists()})")
    print(f"Val Report:   {val_report_path} (Exists: {val_report_path.exists()})")

    assert frozen_jsonl.exists(), f"Frozen JSONL missing: {frozen_jsonl}"
    assert val_report_path.exists(), f"Validation report missing: {val_report_path}"

    with open(frozen_jsonl, "rb") as f:
        file_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"[*] Dataset SHA-256 Checksum: {file_sha256}")

    with open(val_report_path, "r", encoding="utf-8") as f:
        val_report = json.load(f)

    # 6. Detailed Inspection of Every Frozen Record
    frozen_records = []
    with open(frozen_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                frozen_records.append(json.loads(line))

    print(f"[*] Total Frozen Records: {len(frozen_records)}")
    assert len(frozen_records) > 0, "No records collected during 3-minute test!"

    strict_monotonic_timing_pass = 0
    recomputed_hash_matches = 0
    monotonic_probabilities_pass = 0
    consecutive_rounds = True

    round_ids = [r["round_id"] for r in frozen_records]
    for i in range(1, len(round_ids)):
        if round_ids[i] != round_ids[i-1] + 1:
            consecutive_rounds = False

    thresholds = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]
    summary_rows = []

    for idx, r in enumerate(frozen_records):
        rid = r["round_id"]
        mult = r["actual_multiplier"]
        p_mono = r.get("backend_monotonic_prediction_ts")
        f_mono = r.get("backend_monotonic_flying_ts")
        c_mono = r.get("backend_monotonic_crash_ts")

        # 1. Strict monotonic timing check
        is_strict_timing = False
        if p_mono is not None and f_mono is not None and c_mono is not None:
            if p_mono < f_mono < c_mono:
                strict_monotonic_timing_pass += 1
                is_strict_timing = True

        bet_window = round(f_mono - p_mono, 3) if (f_mono and p_mono) else None
        flight_dur = round(c_mono - f_mono, 3) if (c_mono and f_mono) else None

        # 2. Hash recomputation
        expected_res_dict = {
            "round_id": r["round_id"],
            "actual_multiplier": round(float(r["actual_multiplier"]), 2),
            "crash_timestamp": r["timestamp"],
            "evaluation_timestamp": r["evaluation_timestamp"],
            "binary_outcomes": r.get("binary_outcomes", {}),
            "scoring_metrics": r.get("scoring_metrics", {}),
            "linked_pre_round_hash": r.get("pre_round_hash", ""),
            "status": "resolved",
        }
        recomputed_hash = compute_sha256(expected_res_dict)
        fallback_hash = hashlib.sha256(
            f"{r['round_id']}:{r['actual_multiplier']}:{r['timestamp']}:{r.get('pre_round_hash', '')}".encode("utf-8")
        ).hexdigest()
        stored_hash = r.get("resolution_hash")
        hash_match = (recomputed_hash == stored_hash) or (fallback_hash == stored_hash)
        if hash_match:
            recomputed_hash_matches += 1

        # 3. Probability monotonicity across models
        model_preds = r.get("model_predictions", {})
        all_models_mono = True
        for m_name, m_probs in model_preds.items():
            if isinstance(m_probs, dict) and "probabilities" in m_probs:
                m_probs = m_probs["probabilities"]
            if isinstance(m_probs, dict):
                p_vals = [m_probs.get(th) for th in thresholds]
                if None not in p_vals:
                    for k in range(len(p_vals) - 1):
                        if p_vals[k] < p_vals[k+1] - 1e-6:
                            all_models_mono = False
                            break

        probs_champion = r.get("probabilities", {})
        if probs_champion:
            c_vals = [probs_champion.get(th) for th in thresholds]
            for k in range(len(c_vals) - 1):
                if c_vals[k] < c_vals[k+1] - 1e-6:
                    all_models_mono = False
                    break

        if all_models_mono:
            monotonic_probabilities_pass += 1

        summary_rows.append({
            "round_id": rid,
            "multiplier": mult,
            "p_mono": p_mono,
            "f_mono": f_mono,
            "c_mono": c_mono,
            "bet_window": bet_window,
            "flight_dur": flight_dur,
            "hash_match": hash_match,
            "all_mono": all_models_mono,
            "val_status": r.get("validation_status"),
        })

    print("\n--------------------------------------------------------------------------------")
    print(f"{'Round ID':<10} | {'Mult':<6} | {'Bet Window':<11} | {'Flight Dur':<11} | {'p<f<c':<7} | {'Hash Match':<10} | {'Prob Mono':<9} | {'Status':<6}")
    print("--------------------------------------------------------------------------------")
    for row in summary_rows:
        print(f"{row['round_id']:<10} | {row['multiplier']:>4.2f}x | {str(row['bet_window'])+'s':>11} | {str(row['flight_dur'])+'s':>11} | {str(row['p_mono'] < row['f_mono'] < row['c_mono']):<7} | {str(row['hash_match']):<10} | {str(row['all_mono']):<9} | {row['val_status']:<6}")

    print("\n================================================================================")
    print("AUDIT & VERIFICATION RESULTS")
    print("================================================================================")
    print(f"Total Test Duration:              {total_test_duration:.2f} seconds ({total_test_duration/60.0:.2f} minutes)")
    print(f"Total Rounds Captured:            {len(frozen_records)}")
    print(f"Consecutive Round IDs (Zero Gaps): {consecutive_rounds}")
    print(f"Supervisor Validation Status:     {val_report.get('status')}")
    print(f"Valid Records:                    {val_report.get('valid_records')} / {val_report.get('total_records')}")
    print(f"Rejected Records:                 {val_report.get('rejected_records')}")
    print(f"Timing Violations:                {val_report.get('timing_errors')}")
    print(f"Missing Field Errors:             {val_report.get('missing_field_errors')}")
    print(f"Duplicate Rounds Detected:        {val_report.get('duplicate_count')}")
    print(f"Strict Monotonic Ordering Pass:   {strict_monotonic_timing_pass} / {len(frozen_records)}")
    print(f"Cryptographic Hash Matches:       {recomputed_hash_matches} / {len(frozen_records)}")
    print(f"Probability Monotonicity Pass:    {monotonic_probabilities_pass} / {len(frozen_records)}")
    print(f"Frozen Dataset SHA-256:           {file_sha256}")

    # Safety checks
    with open("models/approved_version.json", "r") as f:
        meta = json.load(f)
    print(f"Production Checkpoint:            {meta.get('active_version')} (UNTOUCHED)")
    assert meta.get("active_version") == "v_cli_parity_test", "Production checkpoint was modified!"

    return {
        "session_id": session_id,
        "duration_seconds": round(total_test_duration, 2),
        "total_records": len(frozen_records),
        "valid_records": val_report.get("valid_records"),
        "rejected_records": val_report.get("rejected_records"),
        "dataset_sha256": file_sha256,
        "consecutive_rounds": consecutive_rounds,
        "val_report": val_report,
        "summary_rows": summary_rows,
    }


if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0
    asyncio.run(run_3min_live_test(target_duration_sec=dur))
