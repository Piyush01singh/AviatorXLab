import json
from src.collector.ledger import compute_sha256

def verify_frozen_resolution_hashes():
    file_path = "data/live/sessions/stake_session_1791916077_77d04b4e_frozen.jsonl"
    with open(file_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            res_payload = {
                "round_id": rec["round_id"],
                "actual_multiplier": round(float(rec["actual_multiplier"]), 2),
                "crash_timestamp": rec["timestamp"],
                "evaluation_timestamp": rec["evaluation_timestamp"],
                "binary_outcomes": rec["binary_outcomes"],
                "scoring_metrics": rec["scoring_metrics"],
                "linked_pre_round_hash": rec.get("linked_pre_round_hash") or rec.get("pre_round_hash"),
                "status": "resolved",
            }
            recomputed = compute_sha256(res_payload)
            stored = rec["resolution_hash"]
            match = (stored == recomputed)
            print(f"Frozen Row {idx+1} (Round #{rec['round_id']}): Stored={stored[:12]}... Recomputed={recomputed[:12]}... Match={match}")

if __name__ == "__main__":
    verify_frozen_resolution_hashes()
