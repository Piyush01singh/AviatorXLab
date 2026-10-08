"""Unit and integration tests for live Aviator telemetry parser, manager, and prediction flow."""

import asyncio
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.collector.live_manager import LiveManager
from src.collector.parser import AviatorEventParser, ParsedEvent


@pytest.fixture
def parser():
    return AviatorEventParser()


def test_parser_sockjs_crash_frame(parser):
    # Authentic Spribe Aviator SockJS crash frame
    raw_sockjs = 'a["{\\"type\\":\\"crash\\",\\"round_id\\":10501,\\"multiplier\\":2.48,\\"timestamp\\":\\"2024-04-13T19:00:00Z\\"}"]'
    events = parser.parse_frame(raw_sockjs)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "crash"
    assert ev.round_id == 10501
    assert ev.multiplier == 2.48
    assert ev.stage == "crashed"


def test_parser_spribe_stage_change_waiting(parser):
    # Stage changed to waiting (betting phase)
    payload = {"type": "stage_changed", "stage": "waiting", "round_id": 10502}
    events = parser.parse_frame(payload)
    assert len(events) == 1
    assert events[0].event_type == "stage_change"
    assert events[0].stage == "waiting"
    assert events[0].round_id == 10502


def test_parser_centrifugo_envelope(parser):
    # Centrifugo envelope pattern
    payload = {
        "pub": {
            "data": {
                "type": "round_finish",
                "game_id": "game-999",
                "rate": 15.42,
                "created_at": "2024-04-13T19:05:00Z",
            }
        }
    }
    events = parser.parse_frame(payload)
    assert len(events) == 1
    assert events[0].event_type == "crash"
    assert events[0].round_id == "game-999"
    assert events[0].multiplier == 15.42


def test_parser_history_array(parser):
    payload = {
        "type": "history",
        "history": [1.25, 2.10, 1.05, 4.50, 18.20],
    }
    events = parser.parse_frame(payload)
    assert len(events) == 1
    assert events[0].event_type == "history"
    assert len(events[0].history) == 5
    assert events[0].history[1]["multiplier"] == 2.10


def test_parser_heartbeat_and_malformed(parser):
    assert parser.parse_frame("h") == []
    assert parser.parse_frame("o") == []
    assert parser.parse_frame("invalid json frame") == []


def test_parser_stake_crash_format(parser):
    # Stake Crash GraphQL subscription event
    stake_payload = {
        "event": "crash",
        "data": json.dumps({"crashpoint": 1.74, "id": "stake-101", "status": "crashed"})
    }
    events = parser.parse_frame(stake_payload)
    assert len(events) == 1
    assert events[0].event_type == "crash"
    assert events[0].multiplier == 1.74
    assert events[0].round_id == "stake-101"


@pytest.mark.asyncio
async def test_live_manager_causal_prediction_flow():
    manager = LiveManager(seed_dataset_path="data/raw/synthetic_sample.csv")
    assert len(manager.history_records) >= 50
    assert manager.active_prediction is not None
    assert "ensemble_probability" in manager.active_prediction
    assert "2.0x" in manager.active_prediction["ensemble_probability"]

    prev_round_id = manager.current_round_id
    pred_prob_2x = manager.active_prediction["ensemble_probability"]["2.0x"]

    # Simulate Round N end (Crash at 2.45x -> HIT)
    crash_event = ParsedEvent(
        event_type="crash",
        round_id=prev_round_id,
        multiplier=2.45,
        stage="crashed",
    )
    manager.process_parsed_event(crash_event)

    # Verify prediction vs actual was recorded
    assert len(manager.evaluation_history) == 1
    eval_rec = manager.evaluation_history[0]
    assert eval_rec["round_id"] == prev_round_id
    assert eval_rec["actual_multiplier"] == 2.45
    assert eval_rec["outcome_2x"] == "HIT (≥2.0x)"

    # Verify running metrics updated
    metrics = manager._calculate_running_metrics()
    assert metrics["total_rounds"] == 1
    assert metrics["actual_hit_rate_2x"] == 1.0

    # Verify that prediction for Round N+1 was generated BEFORE round N+1 outcome
    new_prediction = manager.active_prediction
    assert new_prediction is not None
    assert new_prediction["round_id"] != prev_round_id


@pytest.mark.asyncio
async def test_live_manager_sample_stream_start_and_stop():
    manager = LiveManager(seed_dataset_path="data/raw/synthetic_sample.csv")
    await manager.start_sample_stream(interval_sec=0.2)
    assert manager.is_connected is True
    assert manager.source_type == "sample_stream"

    # Allow worker to run briefly
    await asyncio.sleep(0.5)

    await manager.stop_current_stream()
    assert manager.is_connected is False


def test_api_live_endpoints():
    client = TestClient(app)

    # 1. GET /api/live/state
    res = client.get("/api/live/state")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "active_prediction" in data
    assert "history_strip" in data
    assert "running_metrics" in data
    assert "benchmark_comparison" in data

    # 2. POST /api/live/connect (sample stream)
    res_conn = client.post("/api/live/connect", json={"source": "sample_stream", "interval": 10.0})
    assert res_conn.status_code == 200
    assert res_conn.json()["status"] == "connected"

    # 3. POST /api/live/ingest
    res_ingest = client.post("/api/live/ingest", json={"raw_frame": '{"type":"tick","rate":1.05}'})
    assert res_ingest.status_code == 200
    assert res_ingest.json()["status"] == "ingested"

    # 4. POST /api/live/disconnect
    res_disc = client.post("/api/live/disconnect")
    assert res_disc.status_code == 200
    assert res_disc.json()["status"] == "disconnected"


def test_api_live_websocket():
    client = TestClient(app)
    with client.websocket_connect("/api/live/ws") as ws:
        data = ws.receive_json()
        assert "status" in data
        assert "active_prediction" in data


def test_api_live_observer_script_endpoint():
    client = TestClient(app)
    res = client.get("/api/live/observer.js")
    assert res.status_code == 200
    assert "application/javascript" in res.headers["content-type"]
    assert "Aviator ML Observer" in res.text
    assert "sendAviatorTestFrame" in res.text
    assert "window.sendAviatorTestFrame" in res.text
    assert "connectAviatorBridge" in res.text
    assert "bridge.html" in res.text


def test_observer_connection_indicator_progression_and_diagnostics():
    """Verify Observer Installed -> Bridge Connected -> Frames Received and diagnostics."""
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    client = TestClient(app)

    # 1. State 1: Neither Observer nor Bridge connected
    snap1 = client.get("/api/live/state").json()
    obs1 = snap1["observer_telemetry"]
    assert obs1["status"] == "not_detected"
    assert obs1["installed"] is False
    assert obs1["bridge_connected"] is False
    assert obs1["frames_received"] == 0
    assert obs1["frames_receiving"] is False
    assert obs1["parsed_events"] == 0
    assert snap1["is_live"] is False
    assert "not installed" in obs1["diagnostic_reason"].lower()

    # 2. State 2: Bridge connects (e.g. companion bridge window opens)
    bridge_payload = {
        "raw_frame": json.dumps({
            "type": "bridge_connected",
            "bridge_type": "browser_bridge",
            "timestamp": "2026-10-08T19:29:00Z"
        })
    }
    res_b = client.post("/api/live/ingest", json=bridge_payload)
    assert res_b.status_code == 200

    snap2 = client.get("/api/live/state").json()
    obs2 = snap2["observer_telemetry"]
    assert obs2["bridge_connected"] is True
    assert obs2["frames_received"] == 1
    assert "bridge is connected" in obs2["diagnostic_reason"].lower() or "connected" in obs2["diagnostic_reason"].lower()

    # 3. State 3: Observer checks in via bridge
    handshake_payload = {
        "raw_frame": json.dumps({
            "type": "observer_installed",
            "installed": True,
            "via_bridge": True,
            "context": "iframe",
            "host": "aviator-next.spribegaming.com",
            "timestamp": "2026-10-08T19:30:00Z"
        })
    }
    res_hs = client.post("/api/live/ingest", json=handshake_payload)
    assert res_hs.status_code == 200

    snap3 = client.get("/api/live/state").json()
    obs3 = snap3["observer_telemetry"]
    assert obs3["installed"] is True
    assert obs3["bridge_connected"] is True
    assert obs3["host"] == "aviator-next.spribegaming.com"
    assert obs3["frames_received"] == 2
    assert obs3["parsed_events"] == 0
    assert "installed" in obs3["diagnostic_reason"].lower()

    # 4. State 4: Frames Received (Actual game or test crash event)
    crash_payload = {
        "raw_frame": json.dumps({
            "type": "crash",
            "round_id": 88001,
            "multiplier": 3.75,
            "via_bridge": True,
            "timestamp": "2026-10-08T19:30:15Z"
        })
    }
    res_crash = client.post("/api/live/ingest", json=crash_payload)
    assert res_crash.status_code == 200

    snap4 = client.get("/api/live/state").json()
    obs4 = snap4["observer_telemetry"]
    assert obs4["status"] == "frames_received"
    assert obs4["installed"] is True
    assert obs4["bridge_connected"] is True
    assert obs4["frames_received"] == 3
    assert obs4["frames_receiving"] is True
    assert obs4["parsed_events"] == 1
    assert obs4["last_event_type"] == "crash"
    assert obs4["last_round_id"] == 88001
    assert obs4["last_multiplier"] == 3.75
    assert snap4["is_live"] is True
    assert "active" in obs4["diagnostic_reason"].lower() or "operational" in obs4["diagnostic_reason"].lower()


def test_stake_observer_e2e_pipeline():
    """Verify Stake iframe -> observer -> /api/live/ingest -> parser -> LiveManager -> prediction -> /api/live/ws."""
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    client = TestClient(app)

    with client.websocket_connect("/api/live/ws") as ws:
        # Initial snapshot received on connection
        init_state = ws.receive_json()
        assert init_state["observer_telemetry"]["status"] == "not_detected"

        # 1. Stake iframe observer emits handshake
        hs_res = client.post("/api/live/ingest", json={"raw_frame": '{"type":"observer_connected"}'})
        assert hs_res.status_code == 200
        ws_hs_state = ws.receive_json()
        assert ws_hs_state["observer_telemetry"]["status"] == "connected"

        # 2. Stake iframe observer emits betting / waiting stage change
        target_round = 99120
        wait_frame = json.dumps({"type": "stage_changed", "stage": "waiting", "round_id": target_round})
        client.post("/api/live/ingest", json={"raw_frame": wait_frame})
        ws_wait_state = ws.receive_json()
        assert ws_wait_state["current_stage"] == "waiting"
        assert ws_wait_state["current_round_id"] == target_round
        assert ws_wait_state["active_prediction"] is not None

        # 3. Stake iframe observer emits crash event at 2.50x
        crash_frame = json.dumps({
            "type": "crash",
            "round_id": target_round,
            "multiplier": 2.50,
            "timestamp": "2026-10-08T19:35:00Z"
        })
        client.post("/api/live/ingest", json={"raw_frame": crash_frame})
        ws_crash_state = ws.receive_json()

        # Verify full cycle results
        assert ws_crash_state["current_stage"] == "crashed"
        assert ws_crash_state["latest_multiplier"] == 2.50
        assert ws_crash_state["observer_telemetry"]["status"] == "frames_received"
        assert ws_crash_state["observer_telemetry"]["last_multiplier"] == 2.50
        assert ws_crash_state["observer_telemetry"]["last_round_id"] == target_round

        # Verify prediction evaluated and next round predicted
        assert len(ws_crash_state["evaluation_history"]) >= 1
        eval_item = ws_crash_state["evaluation_history"][-1]
        assert eval_item["round_id"] == target_round
        assert eval_item["actual_multiplier"] == 2.50
        assert eval_item["outcome_2x"] == "HIT (≥2.0x)"

        # Verify next round prediction generated causally before next outcome
        assert ws_crash_state["active_prediction"] is not None
        assert ws_crash_state["active_prediction"]["round_id"] != target_round


def test_telemetry_bridge_endpoints_and_cross_origin_pipeline():
    """Verify Telemetry Bridge endpoints (/bridge, /bridge.html, /api/live/bridge-ws) and relay flow."""
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    client = TestClient(app)

    # 1. Verify GET /bridge and /bridge.html serve companion page
    for route in ("/bridge", "/bridge.html"):
        res = client.get(route)
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Aviator Telemetry Bridge" in res.text
        assert "postMessage" in res.text

    # 2. Verify WebSocket bridge streaming (/api/live/bridge-ws)
    target_round = 77055
    with client.websocket_connect("/api/live/bridge-ws") as bridge_ws:
        # Bridge connects -> state updates bridge_connected
        state = client.get("/api/live/state").json()
        assert state["observer_telemetry"]["bridge_connected"] is True

        # Send test frame from bridge over WebSocket
        bridge_payload = json.dumps({
            "type": "crash",
            "round_id": target_round,
            "multiplier": 2.45,
            "via_bridge": True,
            "timestamp": "2026-10-08T19:40:00Z"
        })
        bridge_ws.send_text(json.dumps({"raw_frame": bridge_payload}))

        # Verify backend ingested frame
        state_after = client.get("/api/live/state").json()
        obs_after = state_after["observer_telemetry"]
        assert obs_after["bridge_connected"] is True
        assert obs_after["installed"] is True
        assert obs_after["frames_received"] >= 1
        assert obs_after["last_multiplier"] == 2.45
        assert state_after["latest_multiplier"] == 2.45
        assert state_after["current_stage"] == "crashed"

        # Verify ML evaluation and next prediction
        assert len(state_after["evaluation_history"]) >= 1
        last_eval = state_after["evaluation_history"][-1]
        assert last_eval["round_id"] == target_round
        assert last_eval["actual_multiplier"] == 2.45
        assert last_eval["outcome_2x"] == "HIT (≥2.0x)"
        assert state_after["active_prediction"] is not None


def test_websocket_diagnostics_and_real_game_frame_handling():
    """Verify diagnostics tracking: sockets detected, URLs, raw game frames, parser attempts, and raw logger."""
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    client = TestClient(app)

    # 1. Observer reports detected WebSocket
    socket_frame = {
        "raw_frame": json.dumps({
            "type": "socket_detected",
            "url": "wss://aviator-next.spribegaming.com/sub?format=json",
            "ready_state": 1,
            "detection_source": "prototype_send",
            "timestamp": "2026-10-08T20:00:00Z"
        })
    }
    res_sock = client.post("/api/live/ingest", json=socket_frame)
    assert res_sock.status_code == 200

    state1 = client.get("/api/live/state").json()
    obs1 = state1["observer_telemetry"]
    assert obs1["detected_sockets_count"] == 1
    assert "wss://aviator-next.spribegaming.com/sub?format=json" in obs1["detected_socket_urls"]
    assert obs1["parsed_events"] == 0  # Socket detection is NOT a game event
    assert obs1["parser_recognized_event_count"] == 0
    assert obs1["raw_frame_count"] == 1

    # 2. Observer ping heartbeat (must NOT count as parsed game event)
    ping_frame = {
        "raw_frame": json.dumps({
            "type": "observer_ping",
            "installed": True,
            "via_bridge": True,
            "sockets_detected": 1,
            "socket_urls": ["wss://aviator-next.spribegaming.com/sub?format=json"],
            "raw_frames_captured": 0,
            "timestamp": "2026-10-08T20:00:05Z"
        })
    }
    res_ping = client.post("/api/live/ingest", json=ping_frame)
    assert res_ping.status_code == 200

    state2 = client.get("/api/live/state").json()
    obs2 = state2["observer_telemetry"]
    assert obs2["parsed_events"] == 0
    assert obs2["raw_game_frames_captured"] == 0
    assert obs2["raw_frame_count"] == 2

    # 3. Real captured Spribe/Centrifugo game frame
    game_round = 105432
    raw_spribe_msg = json.dumps({
        "push": {
            "pub": {
                "data": {
                    "type": "crash",
                    "round_id": game_round,
                    "multiplier": 4.12,
                    "stage": "crashed",
                    "time": "2026-10-08T20:00:10Z"
                }
            }
        }
    })
    game_envelope = {
        "raw_frame": json.dumps({
            "type": "game_frame",
            "raw_frame": raw_spribe_msg,
            "socket_url": "wss://aviator-next.spribegaming.com/sub?format=json",
            "frame_index": 1,
            "timestamp": "2026-10-08T20:00:10Z"
        })
    }
    res_game = client.post("/api/live/ingest", json=game_envelope)
    assert res_game.status_code == 200

    state3 = client.get("/api/live/state").json()
    obs3 = state3["observer_telemetry"]
    assert obs3["raw_game_frames_captured"] == 1
    assert obs3["parser_attempts"] >= 1
    assert obs3["parser_recognized_event_count"] == 1
    assert obs3["parsed_events"] == 1
    assert obs3["last_multiplier"] == 4.12
    assert obs3["last_round_id"] == game_round
    assert obs3["last_parsed_event"] is not None
    assert obs3["last_parsed_event"]["event_type"] == "crash"
    assert obs3["last_parsed_event"]["multiplier"] == 4.12
    assert state3["latest_multiplier"] == 4.12
    assert state3["current_stage"] == "crashed"

    # 4. Verify debug frame was appended to data/debug/raw_captured_frames.jsonl
    debug_log_path = Path("data/debug/raw_captured_frames.jsonl")
    assert debug_log_path.exists()
    content = debug_log_path.read_text(encoding="utf-8")
    assert "4.12" in content
    assert str(game_round) in content

    # 5. Verify GET /api/live/observer.js includes new WebSocket prototype hook and diagnostics
    res_obs = client.get("/api/live/observer.js")
    assert res_obs.status_code == 200
    obs_text = res_obs.text
    assert "WebSocket.prototype.send" in obs_text
    assert "scanForExistingSockets" in obs_text
    assert "getAviatorDiagnostics" in obs_text
    assert "sendAviatorTestFrame" in obs_text


def test_frame_deduplication_and_safe_binary_logging():
    """Verify deduplication across multiple hooks, safe binary logging, and rotation."""
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    client = TestClient(app)

    # 1. Multiple duplicate hooks firing identical game crash frame (5 times)
    target_round = 205881
    crash_payload = json.dumps({
        "type": "crash",
        "round_id": target_round,
        "multiplier": 3.42,
        "timestamp": "2026-10-08T21:00:00Z"
    })
    envelope = {"raw_frame": json.dumps({
        "type": "game_frame",
        "raw_frame": crash_payload,
        "frame_index": 10,
        "socket_url": "wss://aviator-next.spribegaming.com/sub"
    })}

    # Post 5 times in rapid succession (simulating multiple hooks firing simultaneously)
    for _ in range(5):
        res = client.post("/api/live/ingest", json=envelope)
        assert res.status_code == 200

    state = client.get("/api/live/state").json()
    obs = state["observer_telemetry"]

    # Ingest packets counted
    assert obs["raw_frame_count"] == 5
    # But de-duplication guarantees only 1 raw game frame was captured and parsed!
    assert obs["raw_game_frames_captured"] == 1
    assert obs["parser_recognized_event_count"] == 1
    assert obs["parsed_events"] == 1
    assert obs["last_multiplier"] == 3.42
    assert obs["last_round_id"] == target_round

    # Verify exactly ONE evaluation record added for this round (Requirement 10)
    matching_evals = [e for e in state["evaluation_history"] if e["round_id"] == target_round]
    assert len(matching_evals) == 1
    assert matching_evals[0]["actual_multiplier"] == 3.42

    # 2. Safe Binary Frame Logging (Requirement 5 & 6)
    bin_envelope = {"raw_frame": json.dumps({
        "type": "binary_frame",
        "socket_url": "wss://aviator-next.spribegaming.com/sub",
        "byte_length": 128,
        "encoding": "binary/protobuf",
        "safe_preview": "0x08 0x01 0x12 0x04 0x73 0x70 0x72 0x69 ...",
        "timestamp": "2026-10-08T21:00:05Z"
    })}
    res_bin = client.post("/api/live/ingest", json=bin_envelope)
    assert res_bin.status_code == 200

    state_bin = client.get("/api/live/state").json()
    obs_bin = state_bin["observer_telemetry"]
    assert obs_bin["last_raw_frame_type"] == "binary_frame"
    assert "128B" in obs_bin["last_raw_frame_snippet"]
    assert "binary/protobuf" in obs_bin["last_raw_frame_snippet"]
    # Binary frame is not a game crash event, so parsed_events remains 1
    assert obs_bin["parsed_events"] == 1

    # Verify debug log format (Requirement 6 & 8)
    debug_log = Path("data/debug/raw_captured_frames.jsonl")
    assert debug_log.exists()
    lines = debug_log.read_text(encoding="utf-8").strip().splitlines()
    last_line = json.loads(lines[-1])
    assert last_line["type"] == "binary_frame"
    assert last_line["byte_length"] == 128
    assert last_line["encoding"] == "binary/protobuf"
    assert "0x08 0x01" in last_line["safe_preview"]

    # 3. Verify Log Rotation when max_capture_log_bytes is reached (Requirement 8)
    orig_max = live_manager.max_capture_log_bytes
    try:
        live_manager.max_capture_log_bytes = 200  # set low threshold to trigger rotation
        # Write another frame to trigger rotation
        rot_envelope = {"raw_frame": json.dumps({
            "type": "game_frame",
            "raw_frame": json.dumps({"type": "stage_changed", "stage": "waiting", "round_id": 999999}),
            "frame_index": 999
        })}
        client.post("/api/live/ingest", json=rot_envelope)
        backup_log = debug_log.with_suffix(".jsonl.1")
        assert backup_log.exists()
    finally:
        live_manager.max_capture_log_bytes = orig_max


def _build_centrifugo_protobuf_frame(payload_dict: dict) -> bytes:
    """Helper to construct a valid Centrifugo Protobuf wire frame containing a publication payload."""
    json_bytes = json.dumps(payload_dict).encode("utf-8")
    # Publication (field 4: bytes data -> tag (4 << 3) | 2 = 0x22)
    pub = bytes([0x22, len(json_bytes)]) + json_bytes
    # Push (field 2: channel "aviator" -> tag 0x12; field 4: pub -> tag 0x22)
    push = bytes([0x12, 7]) + b"aviator" + bytes([0x22, len(pub)]) + pub
    # Reply (field 4: push -> tag 0x22)
    reply = bytes([0x22, len(push)]) + push
    return reply


def test_protobuf_centrifugo_decoding_and_real_round_lifecycle(parser):
    """Regression test: verify Centrifugo Protobuf frame decoding and complete round lifecycle.

    Lifecycle: waiting -> pre-round prediction -> flying -> crash -> actual multiplier -> next prediction.
    Also verifies ping/heartbeat frames are not treated as game events.
    """
    import base64
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    client = TestClient(app)

    # 1. Direct Parser Test on Raw Protobuf Bytes
    crash_dict = {
        "type": "crash",
        "round_id": 889900,
        "multiplier": 3.85,
        "stage": "crashed",
        "time": "2026-10-08T22:30:00Z"
    }
    proto_bytes = _build_centrifugo_protobuf_frame(crash_dict)
    events = parser.parse_frame(proto_bytes)
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "crash"
    assert ev.round_id == 889900
    assert ev.multiplier == 3.85
    assert ev.stage == "crashed"

    # 2. Direct Parser Test on Ping / Lifecycle Protobuf Frame (must NOT be a game event)
    ping_proto = bytes([0x08, 0x00])  # Command/Reply empty ping
    ping_events = parser.parse_frame(ping_proto)
    assert len(ping_events) == 1
    assert ping_events[0].event_type == "binary_frame"
    assert ping_events[0].multiplier is None

    # 3. Full Round Lifecycle through /api/live/ingest and WebSocket
    round_id = 889901

    with client.websocket_connect("/api/live/ws") as ws:
        init_ws = ws.receive_json()
        assert init_ws["status"] in ("connected", "disconnected")

        # Step 3a: WAITING phase in Protobuf
        wait_frame_dict = {
            "type": "stage_changed",
            "stage": "waiting",
            "round_id": round_id
        }
        wait_proto = _build_centrifugo_protobuf_frame(wait_frame_dict)
        wait_b64 = base64.b64encode(wait_proto).decode("ascii")

        ingest_wait = client.post("/api/live/ingest", json={
            "raw_frame": json.dumps({
                "type": "binary_frame",
                "socket_url": "wss://aviator-next.spribegaming.com/sub",
                "byte_length": len(wait_proto),
                "encoding": "binary/protobuf",
                "safe_preview": "0x22 ...",
                "payload_b64": wait_b64,
                "timestamp": "2026-10-08T22:31:00Z"
            })
        })
        assert ingest_wait.status_code == 200

        ws_wait = ws.receive_json()
        assert ws_wait["current_stage"] == "waiting"
        assert ws_wait["current_round_id"] == round_id
        # Pre-round prediction must exist BEFORE round is flown/crashed
        pred_before = ws_wait["active_prediction"]
        assert pred_before is not None
        assert pred_before["round_id"] == round_id
        p2x_before = pred_before["consensus_mean"]["2.0x"]
        assert 0.0 <= p2x_before <= 1.0

        # Step 3b: FLYING phase in Protobuf
        fly_frame_dict = {
            "type": "stage_changed",
            "stage": "flying",
            "round_id": round_id
        }
        fly_proto = _build_centrifugo_protobuf_frame(fly_frame_dict)
        fly_b64 = base64.b64encode(fly_proto).decode("ascii")

        ingest_fly = client.post("/api/live/ingest", json={
            "raw_frame": json.dumps({
                "type": "binary_frame",
                "socket_url": "wss://aviator-next.spribegaming.com/sub",
                "byte_length": len(fly_proto),
                "encoding": "binary/protobuf",
                "safe_preview": "0x22 ...",
                "payload_b64": fly_b64,
                "timestamp": "2026-10-08T22:31:05Z"
            })
        })
        assert ingest_fly.status_code == 200

        ws_fly = ws.receive_json()
        assert ws_fly["current_stage"] == "flying"

        # Step 3c: CRASH phase in Protobuf with actual multiplier 4.56
        actual_mult = 4.56
        crash_frame_dict = {
            "type": "crash",
            "round_id": round_id,
            "multiplier": actual_mult,
            "stage": "crashed"
        }
        crash_proto = _build_centrifugo_protobuf_frame(crash_frame_dict)
        crash_b64 = base64.b64encode(crash_proto).decode("ascii")

        ingest_crash = client.post("/api/live/ingest", json={
            "raw_frame": json.dumps({
                "type": "binary_frame",
                "socket_url": "wss://aviator-next.spribegaming.com/sub",
                "byte_length": len(crash_proto),
                "encoding": "binary/protobuf",
                "safe_preview": "0x22 ...",
                "payload_b64": crash_b64,
                "timestamp": "2026-10-08T22:31:10Z"
            })
        })
        assert ingest_crash.status_code == 200

        ws_crash = ws.receive_json()
        assert ws_crash["current_stage"] == "crashed"
        assert ws_crash["latest_multiplier"] == actual_mult

        # Step 3d: Verify evaluation history (Pre-round prediction evaluated vs actual multiplier)
        eval_history = ws_crash["evaluation_history"]
        assert len(eval_history) >= 1
        last_eval = eval_history[-1]
        assert last_eval["round_id"] == round_id
        assert last_eval["actual_multiplier"] == actual_mult
        assert "HIT" in last_eval["outcome_2x"]
        assert "brier_error" in last_eval

        # Step 3e: Verify next prediction is causally generated for round_id + 1
        next_pred = ws_crash["active_prediction"]
        assert next_pred is not None
        assert next_pred["round_id"] == round_id + 1

        # Step 3f: Heartbeat / ping must NOT advance round or alter predictions
        ping_b64 = base64.b64encode(ping_proto).decode("ascii")
        res_ping = client.post("/api/live/ingest", json={
            "raw_frame": json.dumps({
                "type": "binary_frame",
                "socket_url": "wss://aviator-next.spribegaming.com/sub",
                "byte_length": len(ping_proto),
                "encoding": "binary/protobuf",
                "safe_preview": "0x08 0x00",
                "payload_b64": ping_b64,
                "timestamp": "2026-10-08T22:31:15Z"
            })
        })
        assert res_ping.status_code == 200

        # Observer telemetry counters: 4 binary frames captured, but only 3 real game events parsed
        state_final = client.get("/api/live/state").json()
        obs_telemetry = state_final["observer_telemetry"]
        assert obs_telemetry["raw_game_frames_captured"] == 4
        assert obs_telemetry["parsed_events"] == 3
        assert obs_telemetry["last_multiplier"] == actual_mult
        assert obs_telemetry["last_round_id"] == round_id


def test_pre_round_prediction_immutability_and_exact_evaluation_match():
    """Regression test proving:
    1. Pre-round prediction is snapshotted immutably when 'waiting' occurs.
    2. Prediction is never recomputed/overwritten during 'flying' or 'crash'.
    3. Post-round evaluation uses the exact stored pre-round snapshot.
    4. Proves prediction_before == prediction_used_for_evaluation across all 6 models, ensemble, and thresholds.
    """
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    client = TestClient(app)
    target_round = 771122

    # Step 1: Ingest WAITING frame for target_round
    wait_frame = json.dumps({"type": "stage_changed", "stage": "waiting", "round_id": target_round})
    res_wait = client.post("/api/live/ingest", json={"raw_frame": wait_frame})
    assert res_wait.status_code == 200

    # Fetch live state BEFORE round flights/crashes
    state_before = client.get("/api/live/state").json()
    assert state_before["current_stage"] == "waiting"
    assert state_before["current_round_id"] == target_round
    pred_before = state_before["active_prediction"]
    assert pred_before is not None
    assert pred_before["round_id"] == target_round

    p_before_2x = pred_before["predicted_prob_2x"]
    ens_before = pred_before["ensemble_probability"]
    mean_before = pred_before["consensus_mean"]
    models_before = pred_before["model_predictions"]

    # Step 2: Ingest FLYING frame -> Must remain strictly immutable!
    fly_frame = json.dumps({"type": "stage_changed", "stage": "flying", "round_id": target_round})
    res_fly = client.post("/api/live/ingest", json={"raw_frame": fly_frame})
    assert res_fly.status_code == 200

    state_flying = client.get("/api/live/state").json()
    assert state_flying["current_stage"] == "flying"
    pred_flying = state_flying["active_prediction"]
    # Verify no recomputation during flight
    assert pred_flying["predicted_prob_2x"] == p_before_2x
    assert pred_flying["ensemble_probability"] == ens_before
    assert pred_flying["consensus_mean"] == mean_before
    assert pred_flying["model_predictions"] == models_before

    # Step 3: Ingest CRASH frame with actual outcome
    actual_multiplier = 4.15
    crash_frame = json.dumps({
        "type": "crash",
        "round_id": target_round,
        "multiplier": actual_multiplier,
        "timestamp": "2026-10-08T22:45:00Z"
    })
    res_crash = client.post("/api/live/ingest", json={"raw_frame": crash_frame})
    assert res_crash.status_code == 200

    state_crashed = client.get("/api/live/state").json()
    assert state_crashed["current_stage"] == "crashed"
    assert state_crashed["latest_multiplier"] == actual_multiplier

    # Step 4: Verify Evaluation Record matches pre-round prediction exactly
    evals = state_crashed["evaluation_history"]
    assert len(evals) >= 1
    eval_item = [e for e in evals if e["round_id"] == target_round][-1]

    # 1. Primary probability match
    assert eval_item["predicted_prob_2x"] == p_before_2x
    assert eval_item["actual_multiplier"] == actual_multiplier
    assert "HIT" in eval_item["outcome_2x"]

    # 2. Ensemble match across all thresholds
    for th in ("1.5x", "2.0x", "3.0x", "5.0x", "10.0x"):
        assert eval_item["ensemble_probability"][th] == ens_before[th]
        assert eval_item["consensus_mean"][th] == mean_before[th]

    # 3. All 6 models match across all thresholds
    for model_name in ("Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"):
        assert model_name in eval_item["model_predictions"]
        for th in ("1.5x", "2.0x", "3.0x", "5.0x", "10.0x"):
            assert eval_item["model_predictions"][model_name][th] == models_before[model_name][th]

    # 4. Brier score accuracy verified
    expected_brier = (1.0 - p_before_2x) ** 2
    assert abs(eval_item["brier_error"] - expected_brier) < 1e-4

    # 5. Verify Next prediction generated strictly for target_round + 1
    next_pred = state_crashed["active_prediction"]
    assert next_pred["round_id"] == target_round + 1


def _build_sfs2x_frame(obj_dict: dict, compressed: bool = True) -> bytes:
    """Build a mock SFS2X / BlueBox binary packet for testing."""
    import struct
    import zlib

    def _encode_val(v):
        if isinstance(v, bool):
            return bytes([1, 1 if v else 0])
        elif isinstance(v, int):
            return bytes([4]) + struct.pack(">i", v)
        elif isinstance(v, float):
            return bytes([7]) + struct.pack(">d", v)
        elif isinstance(v, str):
            utf = v.encode("utf-8")
            return bytes([8]) + struct.pack(">H", len(utf)) + utf
        elif isinstance(v, dict):
            return _encode_obj(v)
        elif isinstance(v, list):
            res = bytearray([17])
            res.extend(struct.pack(">H", len(v)))
            for el in v:
                res.extend(_encode_val(el))
            return bytes(res)
        raise ValueError(f"Unsupported SFS value type: {type(v)}")

    def _encode_obj(d: dict) -> bytes:
        res = bytearray([18])
        res.extend(struct.pack(">H", len(d)))
        for k, v in d.items():
            k_utf = str(k).encode("utf-8")
            res.extend(struct.pack(">H", len(k_utf)))
            res.extend(k_utf)
            res.extend(_encode_val(v))
        return bytes(res)

    sfs_bytes = _encode_obj(obj_dict)
    if compressed:
        comp = zlib.compress(sfs_bytes)
        return bytes([0xa0]) + struct.pack(">H", len(comp)) + comp
    else:
        return bytes([0x80]) + struct.pack(">H", len(sfs_bytes)) + sfs_bytes


def test_sfs2x_smartfox_packet_decoding_and_round_lifecycle(parser):
    """Regression test: verify SmartFoxServer 2X (SFS2X / BlueBox) compressed and uncompressed binary frames.

    Verifies:
    1. Decoder correctly parses SFSObject within ZLIB-compressed (0xa0) and uncompressed (0x80) packets.
    2. Heartbeat frames (controller c=0 or ping command) and bets/stats are recognized as ignored events.
    3. LiveManager / Ingest endpoint correctly processes full lifecycle:
       WAITING -> FLYING -> CRASH with exact multiplier.
    4. Counters: raw_game_frames_captured records every frame, parsed_events increments ONLY for real game events.
    """
    import base64
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True

    client = TestClient(app)
    round_id = 993301
    actual_mult = 5.67

    # 1. Direct Parser Test: Compressed (0xa0) SFS2X Crash Frame
    crash_sfs = {
        "c": 1,
        "a": 13,
        "p": {
            "c": "onGameCrash",
            "round_id": round_id,
            "multiplier": actual_mult,
            "stage": "crashed",
        }
    }
    comp_bytes = _build_sfs2x_frame(crash_sfs, compressed=True)
    events = parser.parse_frame(comp_bytes)
    assert len(events) == 1
    assert events[0].event_type == "crash"
    assert events[0].round_id == round_id
    assert events[0].multiplier == actual_mult

    # 2. Direct Parser Test: Uncompressed (0x80) SFS2X Waiting Frame
    wait_sfs = {
        "c": 1,
        "a": 13,
        "p": {
            "c": "onGameStart",
            "round_id": round_id,
            "stage": "waiting",
        }
    }
    uncomp_bytes = _build_sfs2x_frame(wait_sfs, compressed=False)
    events_wait = parser.parse_frame(uncomp_bytes)
    assert len(events_wait) == 1
    assert events_wait[0].event_type == "stage_change"
    assert events_wait[0].stage == "waiting"
    assert events_wait[0].round_id == round_id

    # 3. Direct Parser Test: SFS2X System Ping (c=0) -> ignored_stream_event
    ping_sfs = {
        "c": 0,
        "a": 1,
        "p": {
            "c": "ping"
        }
    }
    ping_bytes = _build_sfs2x_frame(ping_sfs, compressed=False)
    events_ping = parser.parse_frame(ping_bytes)
    assert len(events_ping) == 1
    assert events_ping[0].event_type == "ignored_stream_event"

    # 4. Direct Parser Test: SFS2X User Bets frame -> ignored_stream_event
    bets_sfs = {
        "c": 1,
        "a": 13,
        "p": {
            "c": "bets",
            "total_bets": 42
        }
    }
    bets_bytes = _build_sfs2x_frame(bets_sfs, compressed=True)
    events_bets = parser.parse_frame(bets_bytes)
    assert len(events_bets) == 1
    assert events_bets[0].event_type == "ignored_stream_event"

    # 5. Full End-to-End Pipeline via POST /api/live/ingest
    # Step A: Ingest SFS WAITING frame
    wait_b64 = base64.b64encode(uncomp_bytes).decode("ascii")
    res_w = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
            "byte_length": len(uncomp_bytes),
            "encoding": "binary/sfs2x",
            "safe_preview": "0x80 ...",
            "payload_b64": wait_b64,
            "timestamp": "2026-10-09T01:00:00Z"
        })
    })
    assert res_w.status_code == 200

    state_w = client.get("/api/live/state").json()
    assert state_w["current_stage"] == "waiting"
    assert state_w["current_round_id"] == round_id
    pred_w = state_w["active_prediction"]
    assert pred_w is not None
    assert pred_w["round_id"] == round_id
    pred_prob_2x_w = pred_w["predicted_prob_2x"]

    # Step B: Ingest SFS FLYING frame
    fly_sfs = {
        "c": 1,
        "a": 13,
        "p": {
            "c": "multiplier_tick",
            "round_id": round_id,
            "stage": "flying",
            "multiplier": 1.95,
        }
    }
    fly_bytes = _build_sfs2x_frame(fly_sfs, compressed=True)
    fly_b64 = base64.b64encode(fly_bytes).decode("ascii")
    res_f = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
            "byte_length": len(fly_bytes),
            "encoding": "binary/sfs2x",
            "safe_preview": "0xa0 ...",
            "payload_b64": fly_b64,
            "timestamp": "2026-10-09T01:00:05Z"
        })
    })
    assert res_f.status_code == 200

    state_f = client.get("/api/live/state").json()
    assert state_f["current_stage"] == "flying"
    assert state_f["active_prediction"]["predicted_prob_2x"] == pred_prob_2x_w

    # Step C: Ingest SFS CRASH frame
    crash_b64 = base64.b64encode(comp_bytes).decode("ascii")
    res_c = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
            "byte_length": len(comp_bytes),
            "encoding": "binary/sfs2x",
            "safe_preview": "0xa0 ...",
            "payload_b64": crash_b64,
            "timestamp": "2026-10-09T01:00:10Z"
        })
    })
    assert res_c.status_code == 200

    state_c = client.get("/api/live/state").json()
    assert state_c["current_stage"] == "crashed"
    assert state_c["latest_multiplier"] == actual_mult

    # Check evaluation snapshot
    evals = state_c["evaluation_history"]
    assert len(evals) >= 1
    eval_item = [e for e in evals if e["round_id"] == round_id][-1]
    assert eval_item["predicted_prob_2x"] == pred_prob_2x_w
    assert eval_item["actual_multiplier"] == actual_mult
    assert eval_item["outcome_2x"] == "HIT (≥2.0x)"

    # Step D: Ingest SFS PING frame (c=0) -> must NOT trigger game event or alter prediction
    ping_b64 = base64.b64encode(ping_bytes).decode("ascii")
    res_p = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
            "byte_length": len(ping_bytes),
            "encoding": "binary/sfs2x",
            "safe_preview": "0x80 ...",
            "payload_b64": ping_b64,
            "timestamp": "2026-10-09T01:00:15Z"
        })
    })
    assert res_p.status_code == 200

    # Diagnostics check: 4 binary frames captured, exactly 3 real game events recognized
    state_fin = client.get("/api/live/state").json()
    obs_telemetry = state_fin["observer_telemetry"]
    assert obs_telemetry["raw_game_frames_captured"] == 4
    assert obs_telemetry["parsed_events"] == 3
    assert obs_telemetry["last_multiplier"] == actual_mult
    assert obs_telemetry["last_round_id"] == round_id


def test_disconnect_live_stream_prevents_auto_reconnect():
    """Verify that disconnecting live stream stops ingestion and prevents background frames from auto-reconnecting."""
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()
    live_manager.source_type = "browser_observer"
    live_manager.is_connected = True
    assert live_manager.is_connected is True

    client = TestClient(app)

    # 1. Post disconnect
    res_disc = client.post("/api/live/disconnect")
    assert res_disc.status_code == 200
    assert live_manager.is_connected is False
    assert live_manager.source_type == "disconnected"
    assert live_manager._user_disconnected is True

    # 2. Ingest background frame while disconnected
    test_frame = json.dumps({"type": "stage_changed", "stage": "waiting", "round_id": 999123})
    res_ingest = client.post("/api/live/ingest", json={"raw_frame": test_frame})
    assert res_ingest.status_code == 200

    # Must remain disconnected and not process game events
    assert live_manager.is_connected is False
    assert live_manager.source_type == "disconnected"
    assert live_manager.current_round_id != 999123

    # 3. Explicitly reconnect via connect endpoint
    res_conn = client.post("/api/live/connect", json={"source_type": "browser_observer"})
    assert res_conn.status_code == 200
    assert live_manager.is_connected is True
    assert live_manager._user_disconnected is False

    # 4. Ingest frame now succeeds and updates round
    res_ingest2 = client.post("/api/live/ingest", json={"raw_frame": test_frame})
    assert res_ingest2.status_code == 200
    state = client.get("/api/live/state").json()
    assert state["current_round_id"] == 999123


@pytest.mark.asyncio
async def test_real_stake_sfs2x_lifecycle_and_dashboard_multiplier_update():
    """Regression test proving:
    1. When Stake Observer frames arrive, source_type switches from sample_stream to browser_observer.
    2. Real Stake SFS2X changeState (newStateId=1) updates Active Round and Game Stage to WAITING.
    3. Real Stake SFS2X changeState (newStateId=2) updates Game Stage to FLYING while keeping pre-round snapshot.
    4. Real Stake SFS2X 'x' tick updates latest_multiplier during flight.
    5. Real Stake SFS2X roundChartInfo CRASH event extracts maxMultiplier and updates Latest Multiplier in dashboard.
    6. Main dashboard returns non-empty multiplier and exact round ID.
    7. Unrelated frames (updateCurrentCashOuts, updateCurrentBets, onlinePlayers, PING_RESPONSE) are safely ignored.
    """
    import base64
    from src.collector.live_manager import live_manager
    live_manager.reset_observer_telemetry()

    # Step 1: Start in sample_stream mode (e.g. initial demo state)
    await live_manager.start_sample_stream(interval_sec=5.0)
    assert live_manager.is_connected is True
    assert live_manager.source_type == "sample_stream"

    client = TestClient(app)
    stake_round_id = 8029133
    real_crash_mult = 4.37

    # Step 2: Real Stake Frame 1 - changeState with newStateId=1 (WAITING)
    wait_sfs = {
        "c": 1,
        "a": 13,
        "p": {
            "c": "changeState",
            "p": {
                "newStateId": 1,
                "code": 200,
                "roundId": stake_round_id,
                "timeLeft": 5000,
            }
        }
    }
    wait_bytes = _build_sfs2x_frame(wait_sfs, compressed=True)
    wait_b64 = base64.b64encode(wait_bytes).decode("ascii")

    res_wait = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
            "byte_length": len(wait_bytes),
            "encoding": "binary/sfs2x",
            "payload_b64": wait_b64,
            "timestamp": "2026-10-09T01:30:00Z"
        })
    })
    assert res_wait.status_code == 200

    # Main dashboard MUST now be browser_observer, NOT sample_stream!
    state_wait = client.get("/api/live/state").json()
    assert state_wait["source_type"] == "browser_observer"
    assert state_wait["current_stage"] == "waiting"
    assert state_wait["current_round_id"] == stake_round_id
    assert state_wait["active_prediction"] is not None
    assert state_wait["active_prediction"]["round_id"] == stake_round_id
    pred_prob_2x = state_wait["active_prediction"]["predicted_prob_2x"]

    # Step 3: Real Stake Frame 2 - changeState with newStateId=2 (FLYING)
    fly_sfs = {
        "c": 1,
        "a": 13,
        "p": {
            "c": "changeState",
            "p": {
                "newStateId": 2,
                "code": 200,
                "roundId": stake_round_id,
            }
        }
    }
    fly_bytes = _build_sfs2x_frame(fly_sfs, compressed=True)
    fly_b64 = base64.b64encode(fly_bytes).decode("ascii")

    res_fly = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
            "byte_length": len(fly_bytes),
            "encoding": "binary/sfs2x",
            "payload_b64": fly_b64,
            "timestamp": "2026-10-09T01:30:05Z"
        })
    })
    assert res_fly.status_code == 200

    state_fly = client.get("/api/live/state").json()
    assert state_fly["current_stage"] == "flying"
    assert state_fly["current_round_id"] == stake_round_id
    assert state_fly["active_prediction"]["predicted_prob_2x"] == pred_prob_2x

    # Step 4: Real Stake Frame 3 - 'x' tick (3.06x)
    tick_sfs = {
        "c": 1,
        "a": 13,
        "p": {
            "c": "x",
            "p": {
                "code": 200,
                "x": 3.06
            }
        }
    }
    tick_bytes = _build_sfs2x_frame(tick_sfs, compressed=False)
    tick_b64 = base64.b64encode(tick_bytes).decode("ascii")

    res_tick = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
            "byte_length": len(tick_bytes),
            "encoding": "binary/sfs2x",
            "payload_b64": tick_b64,
            "timestamp": "2026-10-09T01:30:08Z"
        })
    })
    assert res_tick.status_code == 200

    state_tick = client.get("/api/live/state").json()
    assert state_tick["current_stage"] == "flying"
    assert state_tick["latest_multiplier"] == 3.06

    # Step 5: Real Stake Frame 4 - updateCurrentCashOuts (ignored non-game event)
    cashout_sfs = {
        "c": 1,
        "a": 13,
        "p": {
            "c": "updateCurrentCashOuts",
            "p": {
                "code": 200,
                "totalCashOut": 1217.45,
                "openBetsCount": 697
            }
        }
    }
    cashout_bytes = _build_sfs2x_frame(cashout_sfs, compressed=True)
    cashout_b64 = base64.b64encode(cashout_bytes).decode("ascii")
    res_cashout = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
            "byte_length": len(cashout_bytes),
            "encoding": "binary/sfs2x",
            "payload_b64": cashout_b64,
            "timestamp": "2026-10-09T01:30:10Z"
        })
    })
    assert res_cashout.status_code == 200
    state_after_cashout = client.get("/api/live/state").json()
    assert state_after_cashout["current_stage"] == "flying"

    # Step 6: Real Stake Frame 5 - roundChartInfo (CRASH event with maxMultiplier: 4.37)
    crash_sfs = {
        "c": 1,
        "a": 13,
        "p": {
            "c": "roundChartInfo",
            "p": {
                "code": 200,
                "maxMultiplier": real_crash_mult,
                "roundId": stake_round_id,
            }
        }
    }
    crash_bytes = _build_sfs2x_frame(crash_sfs, compressed=True)
    crash_b64 = base64.b64encode(crash_bytes).decode("ascii")

    res_crash = client.post("/api/live/ingest", json={
        "raw_frame": json.dumps({
            "type": "binary_frame",
            "socket_url": "wss://eu-central-1-game7.spribegaming.com/BlueBox/websocket",
            "byte_length": len(crash_bytes),
            "encoding": "binary/sfs2x",
            "payload_b64": crash_b64,
            "timestamp": "2026-10-09T01:30:12Z"
        })
    })
    assert res_crash.status_code == 200

    # MAIN DASHBOARD VERIFICATION: Multiplier MUST be non-empty and equal to 4.37
    state_final = client.get("/api/live/state").json()
    assert state_final["current_stage"] == "crashed"
    assert state_final["latest_multiplier"] == real_crash_mult
    assert state_final["observer_telemetry"]["last_multiplier"] == real_crash_mult
    assert state_final["observer_telemetry"]["last_round_id"] == stake_round_id
    assert state_final["current_round_id"] == stake_round_id + 1

    # Evaluation verification
    evals = state_final["evaluation_history"]
    assert len(evals) >= 1
    eval_rec = [e for e in evals if e["round_id"] == stake_round_id][-1]
    assert eval_rec["actual_multiplier"] == real_crash_mult
    assert eval_rec["predicted_prob_2x"] == pred_prob_2x
    assert eval_rec["outcome_2x"] == "HIT (≥2.0x)"

    # Clean up
    await live_manager.stop_current_stream()







