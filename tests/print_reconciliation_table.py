import json
import base64
from datetime import datetime, timezone
from pathlib import Path
from src.collector.parser import AviatorEventParser

parser = AviatorEventParser()

all_frames = []
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
            all_frames.append(row)

# Collect complete rounds
rounds = {}
for g_idx, frame in enumerate(all_frames):
    b64 = frame.get("payload_b64")
    if not b64:
        continue
    try:
        raw_b = base64.b64decode(b64)
        evts = parser.parse_frame(raw_b)
        for ev in evts:
            rid = ev.round_id
            if not rid:
                continue
            if rid not in rounds:
                rounds[rid] = {}
            if ev.event_type == "stage_change" and ev.stage == "waiting":
                rounds[rid]["waiting"] = {
                    "frame_idx": frame["source_idx"],
                    "file": frame["source_file"],
                    "receipt_ts": frame["timestamp"],
                    "server_ts": ev.timestamp,
                }
            elif ev.event_type == "stage_change" and ev.stage == "flying" and "flying" not in rounds[rid]:
                rounds[rid]["flying"] = {
                    "frame_idx": frame["source_idx"],
                    "file": frame["source_file"],
                    "receipt_ts": frame["timestamp"],
                    "server_ts": ev.timestamp,
                }
            elif ev.event_type == "crash":
                rounds[rid]["crash"] = {
                    "frame_idx": frame["source_idx"],
                    "file": frame["source_file"],
                    "receipt_ts": frame["timestamp"],
                    "server_ts": ev.timestamp,
                    "multiplier": ev.multiplier,
                }
    except Exception:
        pass

print("================================================================================================================")
print("GENUINE SPRIBE RAW WEBSOCKET FRAME RECONCILIATION TABLE (Rounds 8048858 to 8048871)")
print("================================================================================================================")
header = f"{'Round ID':<10} | {'WAITING Frame':<18} | {'FLYING Frame':<18} | {'CRASH Frame':<18} | {'Mult':<7} | {'Bet Window':<11} | {'Flight Dur':<11}"
print(header)
print("-" * len(header))

for rid in sorted(rounds.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
    r = rounds[rid]
    w = r.get("waiting")
    fl = r.get("flying")
    cr = r.get("crash")
    if not (w and fl and cr):
        continue
    
    # Calculate intervals from receipt timestamps
    t_w = datetime.fromisoformat(w["receipt_ts"])
    t_fl = datetime.fromisoformat(fl["receipt_ts"])
    t_cr = datetime.fromisoformat(cr["receipt_ts"])
    
    bet_dur = (t_fl - t_w).total_seconds()
    flight_dur = (t_cr - t_fl).total_seconds()
    
    w_str = f"{w['file']}:{w['frame_idx']}"
    fl_str = f"{fl['file']}:{fl['frame_idx']}"
    cr_str = f"{cr['file']}:{cr['frame_idx']}"
    
    print(f"{rid:<10} | {w_str:<18} | {fl_str:<18} | {cr_str:<18} | {cr['multiplier']:>6.2f}x | {bet_dur:>9.3f}s | {flight_dur:>9.3f}s")
