"""Parser and normalizer for live Aviator (Spribe) WebSocket and game events.

Strictly Read-Only:
Extracts solely round_id, timestamp, multiplier, and round state.
Zero account interaction, zero betting, zero automated execution.
"""

import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import struct
from typing import Any, Dict, List, Optional, Union
import zlib
from src.utils.logging import get_logger

logger = get_logger("event_parser")


@dataclass
class ParsedEvent:
    """Normalized live Aviator round event."""
    event_type: str  # 'crash', 'stage_change', 'history', 'unknown'
    round_id: Optional[Union[int, str]] = None
    multiplier: Optional[float] = None
    stage: Optional[str] = None  # 'waiting', 'flying', 'crashed'
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    history: List[Dict[str, Any]] = field(default_factory=list)
    raw_payload: Optional[Any] = None


class SFSDecoder:
    """SmartFoxServer 2X (SFS2X / BlueBox) binary object and array deserializer."""

    NULL = 0
    BOOL = 1
    BYTE = 2
    SHORT = 3
    INT = 4
    LONG = 5
    FLOAT = 6
    DOUBLE = 7
    UTF_STRING = 8
    BOOL_ARRAY = 9
    BYTE_ARRAY = 10
    SHORT_ARRAY = 11
    INT_ARRAY = 12
    LONG_ARRAY = 13
    FLOAT_ARRAY = 14
    DOUBLE_ARRAY = 15
    UTF_STRING_ARRAY = 16
    SFS_ARRAY = 17
    SFS_OBJECT = 18
    CLASS = 19
    TEXT = 20

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def has_bytes(self, n: int = 1) -> bool:
        return self.offset + n <= len(self.data)

    def read_byte(self) -> int:
        if not self.has_bytes(1):
            raise EOFError("Unexpected end of SFS buffer")
        val = self.data[self.offset]
        self.offset += 1
        return val

    def read_short(self) -> int:
        if not self.has_bytes(2):
            raise EOFError("Unexpected end of SFS buffer")
        val = struct.unpack_from(">h", self.data, self.offset)[0]
        self.offset += 2
        return val

    def read_ushort(self) -> int:
        if not self.has_bytes(2):
            raise EOFError("Unexpected end of SFS buffer")
        val = struct.unpack_from(">H", self.data, self.offset)[0]
        self.offset += 2
        return val

    def read_int(self) -> int:
        if not self.has_bytes(4):
            raise EOFError("Unexpected end of SFS buffer")
        val = struct.unpack_from(">i", self.data, self.offset)[0]
        self.offset += 4
        return val

    def read_long(self) -> int:
        if not self.has_bytes(8):
            raise EOFError("Unexpected end of SFS buffer")
        val = struct.unpack_from(">q", self.data, self.offset)[0]
        self.offset += 8
        return val

    def read_float(self) -> float:
        if not self.has_bytes(4):
            raise EOFError("Unexpected end of SFS buffer")
        val = struct.unpack_from(">f", self.data, self.offset)[0]
        self.offset += 4
        return val

    def read_double(self) -> float:
        if not self.has_bytes(8):
            raise EOFError("Unexpected end of SFS buffer")
        val = struct.unpack_from(">d", self.data, self.offset)[0]
        self.offset += 8
        return val

    def read_utf_string(self) -> str:
        length = self.read_ushort()
        if not self.has_bytes(length):
            raise EOFError("Unexpected end of SFS string buffer")
        val = self.data[self.offset : self.offset + length].decode("utf-8", errors="replace")
        self.offset += length
        return val

    def decode_value(self, type_id: int) -> Any:
        if type_id == self.NULL:
            return None
        elif type_id == self.BOOL:
            return bool(self.read_byte())
        elif type_id == self.BYTE:
            if not self.has_bytes(1):
                raise EOFError()
            val = struct.unpack_from(">b", self.data, self.offset)[0]
            self.offset += 1
            return val
        elif type_id == self.SHORT:
            return self.read_short()
        elif type_id == self.INT:
            return self.read_int()
        elif type_id == self.LONG:
            return self.read_long()
        elif type_id == self.FLOAT:
            return self.read_float()
        elif type_id == self.DOUBLE:
            return self.read_double()
        elif type_id == self.UTF_STRING:
            return self.read_utf_string()
        elif type_id == self.TEXT:
            length = self.read_int()
            if not self.has_bytes(length):
                raise EOFError()
            val = self.data[self.offset : self.offset + length].decode("utf-8", errors="replace")
            self.offset += length
            return val
        elif type_id == self.BOOL_ARRAY:
            count = self.read_ushort()
            return [bool(self.read_byte()) for _ in range(count)]
        elif type_id == self.BYTE_ARRAY:
            length = self.read_int()
            if not self.has_bytes(length):
                raise EOFError()
            val = self.data[self.offset : self.offset + length]
            self.offset += length
            return val
        elif type_id == self.SHORT_ARRAY:
            count = self.read_ushort()
            return [self.read_short() for _ in range(count)]
        elif type_id == self.INT_ARRAY:
            count = self.read_ushort()
            return [self.read_int() for _ in range(count)]
        elif type_id == self.LONG_ARRAY:
            count = self.read_ushort()
            return [self.read_long() for _ in range(count)]
        elif type_id == self.FLOAT_ARRAY:
            count = self.read_ushort()
            return [self.read_float() for _ in range(count)]
        elif type_id == self.DOUBLE_ARRAY:
            count = self.read_ushort()
            return [self.read_double() for _ in range(count)]
        elif type_id == self.UTF_STRING_ARRAY:
            count = self.read_ushort()
            return [self.read_utf_string() for _ in range(count)]
        elif type_id == self.SFS_ARRAY:
            count = self.read_ushort()
            arr = []
            for _ in range(count):
                elem_type = self.read_byte()
                arr.append(self.decode_value(elem_type))
            return arr
        elif type_id == self.SFS_OBJECT:
            count = self.read_ushort()
            obj = {}
            for _ in range(count):
                key = self.read_utf_string()
                val_type = self.read_byte()
                obj[key] = self.decode_value(val_type)
            return obj
        elif type_id == self.CLASS:
            _cls_name = self.read_utf_string()
            return self.decode_value(self.SFS_OBJECT)
        else:
            raise ValueError(f"Unknown SFSDataType: {type_id}")

    def decode(self) -> Any:
        type_id = self.read_byte()
        return self.decode_value(type_id)


class AviatorEventParser:
    """Parses raw WebSocket messages from Aviator game feeds into normalized schema records."""

    STAGE_MAP = {
        "waiting": "waiting",
        "betting": "waiting",
        "start": "waiting",
        "starting": "waiting",
        "new_round": "waiting",
        "round_start": "waiting",
        "ongamestart": "waiting",
        "game_start": "waiting",
        "flying": "flying",
        "in_flight": "flying",
        "running": "flying",
        "in_progress": "flying",
        "flight": "flying",
        "multiplier_tick": "flying",
        "tick": "flying",
        "crashed": "crashed",
        "crash": "crashed",
        "ended": "crashed",
        "finished": "crashed",
        "over": "crashed",
        "completed": "crashed",
        "round_finish": "crashed",
        "round_finished": "crashed",
        "game_crash": "crashed",
        "ongamecrash": "crashed",
        "end_round": "crashed",
        "roundchartinfo": "crashed",
        "round_chart_info": "crashed",
        "roundchart": "crashed",
        "chartinfo": "crashed",
        "x": "flying",
    }

    @staticmethod
    def _extract_bytes_from_dict(d: Dict[str, Any]) -> Optional[bytes]:
        """Extract raw bytes from binary frame dictionary if present."""
        if not isinstance(d, dict):
            return None
        if "payload_b64" in d and d["payload_b64"]:
            try:
                return base64.b64decode(d["payload_b64"])
            except Exception:
                pass
        if "payload_hex" in d and d["payload_hex"]:
            try:
                clean_hex = str(d["payload_hex"]).replace("0x", "").replace(" ", "").replace("\n", "").strip()
                return bytes.fromhex(clean_hex)
            except Exception:
                pass
        if "raw_bytes" in d and isinstance(d["raw_bytes"], (bytes, bytearray)):
            return bytes(d["raw_bytes"])
        if "raw_bytes" in d and isinstance(d["raw_bytes"], list):
            try:
                return bytes(d["raw_bytes"])
            except Exception:
                pass
        return None

    @staticmethod
    def decode_protobuf_payloads(buf: bytes) -> List[str]:
        """Decode Protobuf wire format (Centrifugo / Spribe Aviator) and extract embedded JSON payloads.

        Centrifugo Protobuf client protocol:
          Reply (field 4: Push) -> Push (field 4: Publication) -> Publication (field 4: bytes data)
        Multiple messages per frame may be varint length-delimited.
        """
        payloads: List[str] = []

        def decode_varint(data: bytes, offset: int):
            val = 0
            shift = 0
            while offset < len(data):
                b = data[offset]
                offset += 1
                val |= (b & 0x7F) << shift
                if not (b & 0x80):
                    return val, offset
                shift += 7
            return None, offset

        def scan_message(msg_bytes: bytes, depth: int = 0):
            if depth > 8 or not msg_bytes:
                return
            stripped = msg_bytes.strip()
            if (stripped.startswith(b"{") and stripped.endswith(b"}")) or (stripped.startswith(b"[") and stripped.endswith(b"]")):
                try:
                    text = stripped.decode("utf-8")
                    json.loads(text)
                    payloads.append(text)
                    return
                except Exception:
                    pass

            offset = 0
            while offset < len(msg_bytes):
                tag, next_offset = decode_varint(msg_bytes, offset)
                if tag is None:
                    break
                offset = next_offset
                wire_type = tag & 0x07

                if wire_type == 0:  # varint
                    _, offset = decode_varint(msg_bytes, offset)
                    if offset is None:
                        break
                elif wire_type == 1:  # 64-bit
                    offset += 8
                elif wire_type == 2:  # length-delimited (string, bytes, embedded message)
                    length, offset = decode_varint(msg_bytes, offset)
                    if length is None or offset + length > len(msg_bytes):
                        break
                    chunk = msg_bytes[offset:offset + length]
                    offset += length

                    c_strip = chunk.strip()
                    if (c_strip.startswith(b"{") and c_strip.endswith(b"}")) or (c_strip.startswith(b"[") and c_strip.endswith(b"]")):
                        try:
                            text = c_strip.decode("utf-8")
                            json.loads(text)
                            payloads.append(text)
                            continue
                        except Exception:
                            pass

                    scan_message(chunk, depth + 1)
                elif wire_type == 5:  # 32-bit
                    offset += 4
                else:
                    break

        scan_message(buf)

        # Multi-message varint-framed scan if direct scan yielded nothing
        if not payloads and len(buf) > 1:
            off = 0
            while off < len(buf):
                msg_len, next_off = decode_varint(buf, off)
                if msg_len is None or next_off + msg_len > len(buf):
                    break
                scan_message(buf[next_off:next_off + msg_len])
                off = next_off + msg_len

        # Fallback heuristic scan for embedded JSON substring
        if not payloads:
            try:
                raw_text = buf.decode("utf-8", errors="ignore")
                first_brace = raw_text.find("{")
                last_brace = raw_text.rfind("}")
                if first_brace != -1 and last_brace > first_brace:
                    candidate = raw_text[first_brace:last_brace + 1]
                    try:
                        json.loads(candidate)
                        payloads.append(candidate)
                    except Exception:
                        pass
            except Exception:
                pass

        return payloads

    @classmethod
    def decode_binary_payloads(cls, buf: bytes) -> List[Union[Dict[str, Any], str]]:
        """Decode binary frames from SmartFoxServer (SFS2X / BlueBox), ZLIB compression, or Protobuf."""
        results: List[Union[Dict[str, Any], str]] = []
        if not buf or len(buf) < 2:
            return results

        # 1. Check for SFS packet framing: 0x80 (uncompressed) or 0xa0 (zlib compressed)
        if buf[0] in (0x80, 0xa0) and len(buf) >= 3:
            flag = buf[0]
            body_len = struct.unpack_from(">H", buf, 1)[0]
            body = buf[3 : 3 + body_len] if 3 + body_len <= len(buf) else buf[3:]
            is_compressed = bool(flag & 0x20)
            if is_compressed:
                for offset in range(min(4, len(body))):
                    if body[offset:offset+2] in (b"\x78\x9c", b"\x78\x01", b"\x78\xda"):
                        try:
                            decomp = zlib.decompress(body[offset:])
                            if decomp.startswith(b"\x12"):
                                try:
                                    sfs_obj = SFSDecoder(decomp).decode()
                                    if isinstance(sfs_obj, dict):
                                        results.append(sfs_obj)
                                except Exception:
                                    pass
                            elif (decomp.strip().startswith(b"{") and decomp.strip().endswith(b"}")) or (decomp.strip().startswith(b"[") and decomp.strip().endswith(b"]")):
                                try:
                                    results.append(decomp.decode("utf-8", errors="replace"))
                                except Exception:
                                    pass
                            break
                        except Exception:
                            pass
            else:
                if body.startswith(b"\x12"):
                    try:
                        sfs_obj = SFSDecoder(body).decode()
                        if isinstance(sfs_obj, dict):
                            results.append(sfs_obj)
                    except Exception:
                        pass
                elif (body.strip().startswith(b"{") and body.strip().endswith(b"}")) or (body.strip().startswith(b"[") and body.strip().endswith(b"]")):
                    try:
                        results.append(body.decode("utf-8", errors="replace"))
                    except Exception:
                        pass

        # 2. Check for any embedded ZLIB DEFLATE stream anywhere in the buffer
        if not results:
            for magic in (b"\x78\x9c", b"\x78\x01", b"\x78\xda"):
                idx = buf.find(magic)
                if idx != -1:
                    try:
                        decomp = zlib.decompress(buf[idx:])
                        if decomp.startswith(b"\x12"):
                            try:
                                sfs_obj = SFSDecoder(decomp).decode()
                                if isinstance(sfs_obj, dict):
                                    results.append(sfs_obj)
                            except Exception:
                                pass
                        elif (decomp.strip().startswith(b"{") and decomp.strip().endswith(b"}")) or (decomp.strip().startswith(b"[") and decomp.strip().endswith(b"]")):
                            try:
                                results.append(decomp.decode("utf-8", errors="replace"))
                            except Exception:
                                pass
                        if results:
                            break
                    except Exception:
                        pass

        # 3. Direct uncompressed SFSObject
        if not results and buf.startswith(b"\x12"):
            try:
                sfs_obj = SFSDecoder(buf).decode()
                if isinstance(sfs_obj, dict):
                    results.append(sfs_obj)
            except Exception:
                pass

        # 4. Protobuf wire format (Centrifugo Reply -> Push -> Publication -> data)
        if not results:
            proto_texts = cls.decode_protobuf_payloads(buf)
            if proto_texts:
                results.extend(proto_texts)

        return results

    def parse_frame(self, raw_data: Union[str, bytes, dict, list]) -> List[ParsedEvent]:
        """Parse raw incoming WebSocket frame (SockJS, Centrifugo, SFS2X/BlueBox, Stake, or direct JSON).

        Args:
            raw_data: String or decoded dict/list from WebSocket message.

        Returns:
            List of ParsedEvent objects found in the frame.
        """
        events: List[ParsedEvent] = []

        if isinstance(raw_data, bytes):
            try:
                text = raw_data.decode("utf-8")
                stripped = text.strip()
                if stripped.startswith("{") or stripped.startswith("[") or stripped.startswith("a["):
                    raw_data = text
            except Exception:
                pass

        if isinstance(raw_data, bytes):
            bin_payloads = self.decode_binary_payloads(raw_data)
            if bin_payloads:
                for p in bin_payloads:
                    try:
                        item = json.loads(p) if isinstance(p, str) else p
                        ev = self._parse_json_message(item)
                        if ev:
                            events.append(ev)
                    except Exception:
                        pass
                if events:
                    return events

            safe_prev = " ".join(f"0x{b:02x}" for b in raw_data[:16]) + ("..." if len(raw_data) > 16 else "")
            return [
                ParsedEvent(
                    event_type="binary_frame",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    raw_payload={
                        "type": "binary_frame",
                        "byte_length": len(raw_data),
                        "encoding": "binary/bytes",
                        "safe_preview": safe_prev,
                    },
                )
            ]

        if isinstance(raw_data, str):
            text = raw_data.strip()
            # Handle SockJS frames
            if text in ("h", "o"):  # SockJS heartbeat / open
                return []
            if text.startswith("a["):  # SockJS array wrapper: a["..."] or a[{...}]
                try:
                    payload_list = json.loads(text[1:])
                    for item in payload_list:
                        if isinstance(item, str):
                            try:
                                item = json.loads(item)
                            except Exception:
                                pass
                        if isinstance(item, dict):
                            ev = self._parse_json_message(item)
                            if ev:
                                events.append(ev)
                    return events
                except Exception as e:
                    logger.debug(f"Failed to unwrap SockJS array: {e}")
            elif text.startswith("{") or text.startswith("["):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict):
                                ev = self._parse_json_message(item)
                                if ev:
                                    events.append(ev)
                        return events
                    elif isinstance(parsed, dict):
                        # Extract any binary payload if present
                        extracted_bytes = self._extract_bytes_from_dict(parsed)
                        if not extracted_bytes and isinstance(parsed.get("raw_frame"), dict):
                            extracted_bytes = self._extract_bytes_from_dict(parsed["raw_frame"])
                        if extracted_bytes:
                            bin_payloads = self.decode_binary_payloads(extracted_bytes)
                            for p in bin_payloads:
                                try:
                                    p_item = json.loads(p) if isinstance(p, str) else p
                                    p_ev = self._parse_json_message(p_item)
                                    if p_ev:
                                        events.append(p_ev)
                                except Exception:
                                    pass
                            if events:
                                return events

                        ev = self._parse_json_message(parsed)
                        if ev:
                            events.append(ev)
                        return events
                except Exception as e:
                    logger.debug(f"JSON parse error: {e}")
                    return []
        elif isinstance(raw_data, dict):
            # Check if this dict contains an encoded binary frame
            extracted_bytes = self._extract_bytes_from_dict(raw_data)
            if not extracted_bytes and isinstance(raw_data.get("raw_frame"), dict):
                extracted_bytes = self._extract_bytes_from_dict(raw_data["raw_frame"])
            if extracted_bytes:
                bin_payloads = self.decode_binary_payloads(extracted_bytes)
                for p in bin_payloads:
                    try:
                        item = json.loads(p) if isinstance(p, str) else p
                        ev = self._parse_json_message(item)
                        if ev:
                            events.append(ev)
                    except Exception:
                        pass
                if events:
                    return events

            ev = self._parse_json_message(raw_data)
            if ev:
                events.append(ev)
        elif isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict):
                    ev = self._parse_json_message(item)
                    if ev:
                        events.append(ev)

        return events

    def _parse_json_message(self, msg: Dict[str, Any]) -> Optional[ParsedEvent]:
        """Extract round_id, multiplier, stage, and history from a single JSON payload."""
        # Iteratively unwrap outer envelopes: Centrifugo, SFS params, raw_frame, etc.
        inner = msg
        parent_cmd = None
        for _ in range(8):
            unwrapped = False
            if isinstance(inner, dict):
                cmd_cand = inner.get("cmd") or inner.get("command") or inner.get("type") or inner.get("event") or inner.get("action")
                if not cmd_cand and isinstance(inner.get("c"), str):
                    cmd_cand = inner["c"]
                if cmd_cand and isinstance(cmd_cand, str) and cmd_cand not in ("1", "0", ""):
                    parent_cmd = cmd_cand
            for envelope_key in ("raw_frame", "push", "pub", "data", "payload", "message", "body", "result", "p", "params"):
                if isinstance(inner, dict) and envelope_key in inner:
                    val = inner[envelope_key]
                    if isinstance(val, dict):
                        inner = val
                        unwrapped = True
                        break
                    elif isinstance(val, str) and (val.strip().startswith("{") or val.strip().startswith("[")):
                        try:
                            parsed_val = json.loads(val.strip())
                            if isinstance(parsed_val, dict):
                                inner = parsed_val
                                unwrapped = True
                                break
                        except Exception:
                            pass
            if not unwrapped or not isinstance(inner, dict):
                break

        # Handle GraphQL subscriptions like Stake crashGame
        if "data" in inner and isinstance(inner["data"], dict) and "crashGame" in inner["data"]:
            inner = inner["data"]["crashGame"]
        elif "crashGame" in inner and isinstance(inner["crashGame"], dict):
            inner = inner["crashGame"]

        raw_cmd = inner.get("type") or inner.get("cmd") or inner.get("command") or inner.get("event") or inner.get("action")
        if not raw_cmd and isinstance(inner.get("c"), str):
            raw_cmd = inner["c"]
        if not raw_cmd and parent_cmd:
            raw_cmd = parent_cmd
        msg_type = str(raw_cmd or "").lower()

        # Check SFS2X Aviator changeState newStateId (1 = waiting, 2 = flying)
        new_state_id = inner.get("newStateId") or inner.get("new_state_id") or inner.get("stateId") or inner.get("state_id")
        if new_state_id == 1:
            stage_raw = "waiting"
        elif new_state_id == 2:
            stage_raw = "flying"
        else:
            stage_raw = str(inner.get("stage") or inner.get("state") or inner.get("status") or inner.get("st") or "").lower()

        normalized_stage = self.STAGE_MAP.get(stage_raw, self.STAGE_MAP.get(msg_type))

        # Check for binary frame with possible embedded payload
        if msg_type == "binary_frame":
            extracted_bytes = self._extract_bytes_from_dict(inner) or self._extract_bytes_from_dict(msg)
            if extracted_bytes:
                bin_payloads = self.decode_binary_payloads(extracted_bytes)
                for p in bin_payloads:
                    try:
                        p_item = json.loads(p) if isinstance(p, str) else p
                        p_ev = self._parse_json_message(p_item)
                        if p_ev and p_ev.event_type in ("crash", "stage_change", "history", "ignored_stream_event"):
                            return p_ev
                    except Exception:
                        pass
            return ParsedEvent(
                event_type="binary_frame",
                timestamp=self._normalize_timestamp(inner.get("timestamp")),
                raw_payload=msg,
            )

        # Check for ping, heartbeat, player stats, bets, chat, online players to safely ignore (Requirement 6)
        if msg_type in (
            "ping",
            "pong",
            "heartbeat",
            "keepalive",
            "player_stats",
            "players_online",
            "chat",
            "chat_message",
            "bet",
            "bets",
            "cashout",
            "cashouts",
            "user_bet",
            "user_cashout",
            "system_ping",
            "updatecurrentcashouts",
            "update_current_cashouts",
            "updatecurrentbets",
            "update_current_bets",
            "onlineplayers",
            "online_players",
            "ping_response",
            "pingresponse",
        ) or (inner.get("c") == 0 and "round_id" not in inner and "roundId" not in inner and "multiplier" not in inner and "maxMultiplier" not in inner):
            return ParsedEvent(
                event_type="ignored_stream_event",
                timestamp=self._normalize_timestamp(inner.get("timestamp")),
                raw_payload=msg,
            )

        # Check for observer, bridge, and socket lifecycle frames
        if msg_type in (
            "observer_connected",
            "observer_installed",
            "observer_ping",
            "observer_heartbeat",
            "handshake",
            "bridge_connected",
            "bridge_heartbeat",
            "bridge_ping",
            "bridge_ready",
            "socket_detected",
        ):
            return ParsedEvent(
                event_type=msg_type,
                timestamp=self._normalize_timestamp(inner.get("timestamp")),
                raw_payload=msg,
            )

        # Check for initial history dump
        history_raw = inner.get("history") or inner.get("recent_rounds") or inner.get("recent_multipliers")
        if history_raw and isinstance(history_raw, list):
            norm_history = self._normalize_history_list(history_raw)
            if norm_history:
                return ParsedEvent(
                    event_type="history",
                    history=norm_history,
                    raw_payload=msg,
                )

        # Extract round_id
        round_id = (
            inner.get("round_id")
            or inner.get("roundId")
            or inner.get("game_id")
            or inner.get("gameId")
            or inner.get("id")
            or inner.get("round")
            or inner.get("game")
            or inner.get("rid")
            or inner.get("r")
        )

        # Extract multiplier
        multiplier = None
        for k in (
            "multiplier",
            "maxMultiplier",
            "max_multiplier",
            "final_multiplier",
            "finalMultiplier",
            "crash",
            "crashpoint",
            "crash_point",
            "val",
            "rate",
            "coefficient",
            "coef",
            "payout",
            "odd",
            "odds",
            "result",
            "m",
            "f",
            "x",
        ):
            val = inner.get(k)
            if val is not None:
                try:
                    fval = float(val)
                    if fval >= 1.0:
                        multiplier = round(fval, 2)
                        break
                except (ValueError, TypeError):
                    pass

        # Extract timestamp
        ts = inner.get("timestamp") or inner.get("time") or inner.get("created_at")
        ts_str = self._normalize_timestamp(ts)

        # Detect CRASH / ROUND_FINISH event
        is_crash = (
            normalized_stage == "crashed"
            or msg_type in (
                "crash",
                "round_finish",
                "round_finished",
                "crashed",
                "end_round",
                "game_over",
                "roundchartinfo",
                "round_chart_info",
                "chartinfo",
                "roundchart",
                "ongamecrash",
                "game_crash",
            )
            or (multiplier is not None and normalized_stage not in ("flying", "waiting"))
        )

        if is_crash and multiplier is not None:
            return ParsedEvent(
                event_type="crash",
                round_id=round_id,
                multiplier=multiplier,
                stage="crashed",
                timestamp=ts_str,
                raw_payload=msg,
            )

        # Detect STAGE CHANGE (waiting / flying)
        if normalized_stage in ("waiting", "flying"):
            return ParsedEvent(
                event_type="stage_change",
                round_id=round_id,
                multiplier=multiplier,  # could be live tick multiplier if flying
                stage=normalized_stage,
                timestamp=ts_str,
                raw_payload=msg,
            )

        # Fallback if round_id and stage present
        if round_id is not None and (msg_type or stage_raw):
            return ParsedEvent(
                event_type="stage_change",
                round_id=round_id,
                stage=normalized_stage or "waiting",
                timestamp=ts_str,
                raw_payload=msg,
            )

        return None

    def _normalize_history_list(self, raw_list: List[Any]) -> List[Dict[str, Any]]:
        """Normalize an array of historical rounds into standard records."""
        history: List[Dict[str, Any]] = []
        base_time = datetime.now(timezone.utc)

        for i, item in enumerate(raw_list):
            if isinstance(item, (int, float)):
                # Just a raw multiplier number
                val = round(float(item), 2)
                if val >= 1.0:
                    history.append({
                        "round_id": i + 1,
                        "multiplier": val,
                        "timestamp": base_time.isoformat(),
                    })
            elif isinstance(item, dict):
                r_id = item.get("round_id") or item.get("id") or (i + 1)
                mult = None
                for k in ("multiplier", "crash", "rate", "coef", "coefficient"):
                    if k in item and item[k] is not None:
                        try:
                            fval = float(item[k])
                            if fval >= 1.0:
                                mult = round(fval, 2)
                                break
                        except (ValueError, TypeError):
                            pass
                if mult is not None:
                    ts = self._normalize_timestamp(item.get("timestamp") or item.get("time"))
                    history.append({
                        "round_id": r_id,
                        "multiplier": mult,
                        "timestamp": ts,
                    })

        return history

    @staticmethod
    def _normalize_timestamp(val: Any) -> str:
        """Normalize timestamp to ISO 8601 string."""
        if val is None:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(val, (int, float)):
            # Millisecond or second epoch
            if val > 1e11:  # milliseconds
                val = val / 1000.0
            try:
                return datetime.fromtimestamp(val, tz=timezone.utc).isoformat()
            except Exception:
                return datetime.now(timezone.utc).isoformat()
        if isinstance(val, str):
            try:
                # Validate or parse
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                return dt.isoformat()
            except Exception:
                return val
        return datetime.now(timezone.utc).isoformat()
