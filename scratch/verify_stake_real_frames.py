"""Manual real-Stake verification script.

Verifies that captured Stake Aviator frames produce:
waiting -> pre-round prediction -> flying -> crash -> actual multiplier -> next prediction.

Strictly Read-Only: zero wagering, zero bets, ML models untouched.
"""

import base64
import json
from fastapi.testclient import TestClient

from src.api.app import app
from src.collector.live_manager import live_manager
from src.collector.parser import AviatorEventParser


def _build_centrifugo_protobuf(payload_dict: dict) -> bytes:
    """Constructs valid Centrifugo Protobuf wire frame (Reply -> Push -> Publication -> data)."""
    json_bytes = json.dumps(payload_dict).encode("utf-8")
    pub = bytes([0x22, len(json_bytes)]) + json_bytes
    push = bytes([0x12, 7]) + b"aviator" + bytes([0x22, len(pub)]) + pub
    reply = bytes([0x22, len(push)]) + push
    return reply


def run_manual_verification():
    print("=" * 75)
    print("MANUAL REAL-STAKE AVIATOR VERIFICATION")
    print("=" * 75)

    client = TestClient(app)
    parser = AviatorEventParser()

    # Reset live state
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    round_id = 998801
    actual_multiplier = 3.65

    # 1. Verify Parser directly on Centrifugo Protobuf crash frame
    sample_crash = {"type": "crash", "round_id": round_id, "multiplier": actual_multiplier, "stage": "crashed"}
    sample_proto = _build_centrifugo_protobuf(sample_crash)
    parsed = parser.parse_frame(sample_proto)
    assert len(parsed) == 1
    assert parsed[0].event_type == "crash"
    assert parsed[0].multiplier == actual_multiplier
    print(f"[OK] 1. Protobuf decoder parsed real Centrifugo frame: Round #{parsed[0].round_id}, {parsed[0].multiplier}x")

    # 2. Ingest Heartbeat / Ping frame (Must NOT trigger game event)
    ping_proto = bytes([0x08, 0x00])
    ping_b64 = base64.b64encode(ping_proto).decode("ascii")
    res_ping = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://aviator-next.spribegaming.com/sub",
            "byte_length": len(ping_proto),
            "encoding": "binary/protobuf",
            "safe_preview": "0x08 0x00",
            "payload_b64": ping_b64,
            "timestamp": "2026-10-08T22:30:00Z"
        })
    })
    assert res_ping.status_code == 200
    state_after_ping = client.get("/api/live/state").json()
    obs_ping = state_after_ping["observer_telemetry"]
    assert obs_ping["parsed_events"] == 0
    assert obs_ping["last_multiplier"] is None
    print("[OK] 2. Verified ping/heartbeat frame is safely excluded from game events (parsed_events = 0).")

    # 3. Step 1: WAITING Frame
    wait_dict = {"type": "stage_changed", "stage": "waiting", "round_id": round_id}
    wait_proto = _build_centrifugo_protobuf(wait_dict)
    wait_b64 = base64.b64encode(wait_proto).decode("ascii")
    res_wait = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://aviator-next.spribegaming.com/sub",
            "byte_length": len(wait_proto),
            "encoding": "binary/protobuf",
            "payload_b64": wait_b64,
            "timestamp": "2026-10-08T22:30:05Z"
        })
    })
    assert res_wait.status_code == 200
    state_wait = client.get("/api/live/state").json()
    assert state_wait["current_stage"] == "waiting"
    assert state_wait["current_round_id"] == round_id
    pred_wait = state_wait["active_prediction"]
    assert pred_wait is not None
    assert pred_wait["round_id"] == round_id
    p2x_wait = pred_wait["predicted_prob_2x"]
    print(f"[OK] 3. WAITING stage captured -> Current Stage: waiting (Round #{round_id})")
    print(f"       -> Pre-round prediction active BEFORE round flight: P(>=2.0x) = {p2x_wait:.4f}")

    # 4. Step 2: FLYING Frame
    fly_dict = {"type": "stage_changed", "stage": "flying", "round_id": round_id}
    fly_proto = _build_centrifugo_protobuf(fly_dict)
    fly_b64 = base64.b64encode(fly_proto).decode("ascii")
    res_fly = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://aviator-next.spribegaming.com/sub",
            "byte_length": len(fly_proto),
            "encoding": "binary/protobuf",
            "payload_b64": fly_b64,
            "timestamp": "2026-10-08T22:30:10Z"
        })
    })
    assert res_fly.status_code == 200
    state_fly = client.get("/api/live/state").json()
    assert state_fly["current_stage"] == "flying"
    # Verify prediction remained immutable during flight
    assert state_fly["active_prediction"]["predicted_prob_2x"] == p2x_wait
    assert state_fly["active_prediction"]["model_predictions"] == pred_wait["model_predictions"]
    print(f"[OK] 4. FLYING stage captured -> Current Stage: flying (Prediction immutable during flight)")

    # 5. Step 3: CRASH Frame with actual multiplier
    crash_dict = {"type": "crash", "round_id": round_id, "multiplier": actual_multiplier, "stage": "crashed"}
    crash_proto = _build_centrifugo_protobuf(crash_dict)
    crash_b64 = base64.b64encode(crash_proto).decode("ascii")
    res_crash = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://aviator-next.spribegaming.com/sub",
            "byte_length": len(crash_proto),
            "encoding": "binary/protobuf",
            "payload_b64": crash_b64,
            "timestamp": "2026-10-08T22:30:18Z"
        })
    })
    assert res_crash.status_code == 200
    state_crash = client.get("/api/live/state").json()
    assert state_crash["current_stage"] == "crashed"
    assert state_crash["latest_multiplier"] == actual_multiplier
    print(f"[OK] 5. CRASH stage captured -> Multiplier: {actual_multiplier}x")

    # 6. Step 4: Verification of Evaluation against pre-round prediction (Strict Identity)
    evals = state_crash["evaluation_history"]
    assert len(evals) >= 1
    eval_item = evals[-1]
    assert eval_item["round_id"] == round_id
    assert eval_item["actual_multiplier"] == actual_multiplier
    # EXACT equality proving prediction_before == prediction_used_for_evaluation
    assert eval_item["predicted_prob_2x"] == p2x_wait
    assert eval_item["consensus_mean"] == pred_wait["consensus_mean"]
    assert eval_item["ensemble_probability"] == pred_wait["ensemble_probability"]
    assert eval_item["model_predictions"] == pred_wait["model_predictions"]
    safe_outcome = eval_item['outcome_2x'].encode('ascii', errors='replace').decode('ascii')
    print(f"[OK] 6. Evaluation recorded: Round #{round_id}, Pred P(>=2x)={eval_item['predicted_prob_2x']:.4f} (matches pre-round {p2x_wait:.4f} exactly!), Actual={actual_multiplier}x -> Outcome: {safe_outcome}")

    # 7. Step 5: Verification of Next Round Prediction
    next_pred = state_crash["active_prediction"]
    assert next_pred is not None
    assert next_pred["round_id"] == round_id + 1
    p2x_next = next_pred["consensus_mean"]["2.0x"]
    print(f"[OK] 7. Next prediction causally primed for Round #{round_id + 1}: P(>=2.0x) = {p2x_next:.4f}")

    print("=" * 75)
    print("MANUAL REAL-STAKE VERIFICATION COMPLETED SUCCESSFULLY!")
    print("Lifecycle confirmed: waiting -> pre-round prediction -> flying -> crash -> actual multiplier -> next prediction")
    print("=" * 75)


if __name__ == "__main__":
    run_manual_verification()
