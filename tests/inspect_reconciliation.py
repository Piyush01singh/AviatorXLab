import json
import base64
from pathlib import Path
from src.collector.parser import AviatorEventParser

parser = AviatorEventParser()

def inspect_raw(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"{path_str} does not exist")
        return {}
    
    frames = []
    with open(p, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            b64 = row.get("payload_b64")
            ts = row.get("timestamp")
            parsed = []
            if b64:
                try:
                    raw_b = base64.b64decode(b64)
                    evts = parser.parse_frame(raw_b)
                    for e in evts:
                        parsed.append({
                            "event_type": e.event_type,
                            "round_id": e.round_id,
                            "stage": e.stage,
                            "multiplier": e.multiplier,
                            "timestamp": e.timestamp
                        })
                except Exception as err:
                    pass
            frames.append({"idx": idx, "timestamp": ts, "parsed": parsed})
            
    print(f"\n=======================================================")
    print(f"RAW CAPTURE FILE: {path_str}")
    print(f"Total frames: {len(frames)}")
    if frames:
        print(f"First frame timestamp: {frames[0]['timestamp']}")
        print(f"Last frame timestamp:  {frames[-1]['timestamp']}")
    
    rounds_found = {}
    for f in frames:
        for ev in f["parsed"]:
            rid = ev["round_id"]
            if rid:
                if rid not in rounds_found:
                    rounds_found[rid] = []
                rounds_found[rid].append({
                    "frame_idx": f["idx"],
                    "frame_ts": f["timestamp"],
                    "event_type": ev["event_type"],
                    "stage": ev["stage"],
                    "multiplier": ev["multiplier"],
                    "server_time": ev["timestamp"]
                })
                
    print(f"Rounds detected in {path_str}: {list(rounds_found.keys())}")
    for rid, evs in rounds_found.items():
        print(f"  Round #{rid}: {len(evs)} events")
        for ev in evs:
            print(f"    Frame #{ev['frame_idx']} [{ev['event_type']}:{ev['stage']}] mult={ev['multiplier']} server_ts={ev['server_time']} frame_ts={ev['frame_ts']}")
            
    return rounds_found

def inspect_frozen(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"{path_str} does not exist")
        return []
    
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"\n=======================================================")
    print(f"FROZEN SESSION: {path_str}")
    print(f"Total records: {len(records)}")
    if records:
        r0 = records[0]
        rN = records[-1]
        print(f"First record: Round #{r0['round_id']} | Mult: {r0['actual_multiplier']} | TS: {r0.get('timestamp')}")
        print(f"Last record:  Round #{rN['round_id']} | Mult: {rN['actual_multiplier']} | TS: {rN.get('timestamp')}")
        print(f"Round ID range: {r0['round_id']} to {rN['round_id']}")
    return records

if __name__ == "__main__":
    inspect_raw("data/debug/raw_captured_frames.jsonl")
    inspect_raw("data/debug/raw_captured_frames.jsonl.1")
    inspect_frozen("data/live/sessions/stake_session_1791919131_200274bc_frozen.jsonl")
    inspect_frozen("data/live/sessions/stake_session_1791916077_77d04b4e_frozen.jsonl")
    inspect_frozen("data/live/sessions/stake_session_1791914191_e0c7f2c7_frozen.jsonl")
