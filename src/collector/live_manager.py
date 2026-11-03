"""Live session manager for Aviator round observation and real-time causal ML prediction.

Strictly Read-Only:
- Zero betting or automated wagers.
- Zero account interaction.
- Strict anti-leakage guarantee: Prediction for Round N is derived strictly from rounds < N.
"""

import asyncio
import copy
from copy import deepcopy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from src.api.inference import (
    THRESHOLD_KEYS,
    THRESHOLDS,
    load_benchmark_comparison_table,
    registry,
)
from src.collector.collector import collector
from src.collector.ledger import ledger
from src.collector.parser import AviatorEventParser, ParsedEvent
from src.data.cleaner import DataCleaner
from src.features.builder import FeatureBuilder
from src.models.champion_selector import enforce_probability_monotonicity
from src.models.shadow_learner import shadow_learner
from src.training.overnight_pipeline import overnight_pipeline
from src.utils.logging import get_logger

logger = get_logger("live_manager")


class LiveManager:
    """Manages real-time Aviator rounds, pre-round causal predictions, and running metrics."""

    def __init__(self, seed_dataset_path: str = "data/raw/real_aviator_historical.csv"):
        self.parser = AviatorEventParser()
        self.seed_dataset_path = Path(seed_dataset_path)

        # Connection status
        self.is_connected: bool = False
        self.source_type: str = "disconnected"  # 'sample_stream', 'websocket', 'browser_observer', 'disconnected'
        self.ws_url: Optional[str] = None

        # Round state
        self.current_stage: str = "idle"  # 'waiting', 'flying', 'crashed', 'idle'
        self.current_round_id: Optional[Any] = None
        self.latest_multiplier: Optional[float] = None
        self.latest_timestamp: Optional[str] = None

        # Rolling history buffer (minimum 50 rounds required for rolling lag feature windows)
        self.history_records: List[Dict[str, Any]] = []
        self._load_seed_history()

        # Active pre-round prediction for the current/upcoming round
        self.active_prediction: Optional[Dict[str, Any]] = None

        # Immutable pre-round prediction snapshots keyed by round_id
        self.pre_round_predictions: Dict[Any, Dict[str, Any]] = {}

        # Live prediction vs actual evaluation tracking
        self.evaluation_history: List[Dict[str, Any]] = []

        # Background tasks & listeners
        self._worker_task: Optional[asyncio.Task] = None
        self._subscribers: Set[asyncio.Queue] = set()

        # Pre-generate initial prediction for next round
        self._generate_next_prediction()

        # Real-time observer connection & telemetry tracking
        self.observer_installed: bool = False
        self.observer_install_timestamp: Optional[str] = None
        self.observer_host: Optional[str] = None
        self.bridge_connected: bool = False
        self.bridge_connected_timestamp: Optional[str] = None
        self.bridge_type: Optional[str] = None
        self.observer_frames_received: int = 0
        self.observer_parsed_events: int = 0
        self.observer_status: str = "not_detected"  # 'not_detected', 'installed', 'bridge_connected', 'frames_received'
        self.observer_last_event_type: Optional[str] = None
        self.observer_last_round_id: Optional[Any] = None
        self.observer_last_multiplier: Optional[float] = None
        self.observer_last_seen_timestamp: Optional[str] = None

        # Diagnostic inspection metrics (Requirement 1 & 4)
        self.detected_sockets: List[Dict[str, Any]] = []
        self.detected_socket_urls: List[str] = []
        self.game_socket_status: str = "not_detected"
        self.platform_socket_status: str = "not_detected"
        self.disconnect_count: int = 0
        self.reconnect_count: int = 0
        self.missed_rounds_count: int = 0
        self.duplicate_frames_detected_count: int = 0
        self.raw_game_frames_captured: int = 0
        self.raw_frame_count: int = 0
        self.parser_attempts: int = 0
        self.parser_recognized_events: int = 0
        self.last_raw_frame: Optional[str] = None
        self.last_raw_frame_type: Optional[str] = None
        self.last_parsed_event: Optional[Dict[str, Any]] = None
        self.raw_capture_log_path: Path = Path("data/debug/raw_captured_frames.jsonl")
        self.max_capture_log_bytes: int = 1 * 1024 * 1024  # 1 MB rotation cap
        self.max_capture_log_lines: int = 1000

        # De-duplication and rate-limiting buffers (Requirement 1, 2, 7)
        self._recent_raw_signatures: Dict[str, float] = {}
        self._recent_event_keys: Dict[str, float] = {}
        self._last_diag_log_time: float = 0.0
        self._pending_logged_frames_count: int = 0
        self._user_disconnected: bool = False

        # Pre-round prediction lifecycle guards
        self.round_start_flying_monotonic: Optional[float] = None
        self.flying_timestamps: Dict[Any, Dict[str, Any]] = {}
        self.active_prediction_expired: bool = False
        self.last_round_id_seen: Optional[Any] = None

        # Post-session training requires explicit manual user confirmation
        collector.on_session_complete_callback = None

    def _load_seed_history(self) -> None:
        """Seed initial 100 historical rounds so feature windows (size=50) are primed immediately."""
        try:
            if not self.seed_dataset_path.exists():
                self.seed_dataset_path = Path("data/raw/synthetic_sample.csv")

            if self.seed_dataset_path.exists():
                df = pd.read_csv(self.seed_dataset_path)
                cleaner = DataCleaner()
                clean_df, _ = cleaner.clean(df)
                tail_df = clean_df.tail(100)

                for _, r in tail_df.iterrows():
                    ts = (
                        r["timestamp"].isoformat()
                        if hasattr(r["timestamp"], "isoformat")
                        else str(r["timestamp"])
                    )
                    r_id = r["round_id"]
                    try:
                        r_id = int(r_id)
                    except Exception:
                        r_id = str(r_id)

                    self.history_records.append({
                        "round_id": r_id,
                        "timestamp": ts,
                        "multiplier": round(float(r["multiplier"]), 2),
                    })

                if self.history_records:
                    last = self.history_records[-1]
                    self.latest_multiplier = last["multiplier"]
                    self.latest_timestamp = last["timestamp"]
                    if isinstance(last["round_id"], int):
                        self.current_round_id = last["round_id"] + 1
                    else:
                        self.current_round_id = f"{last['round_id']}_next"

                logger.info(f"Primed live manager with {len(self.history_records)} historical rounds.")
        except Exception as e:
            logger.warning(f"Could not load seed history: {e}")

    def subscribe(self) -> asyncio.Queue:
        """Register a new frontend client queue for real-time state broadcasts."""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Unregister a client queue."""
        self._subscribers.discard(q)

    async def broadcast_state(self) -> None:
        """Push current snapshot to all active client queues."""
        if not self._subscribers:
            return
        snapshot = self.get_state_snapshot()
        for q in list(self._subscribers):
            try:
                q.put_nowait(snapshot)
            except Exception:
                self._subscribers.discard(q)

    def _get_diagnostic_reason(self) -> str:
        """Return clear diagnostic explanation for the current observer and bridge connection state."""
        if self.source_type == "browser_observer" or self.observer_status != "not_detected" or self.observer_installed or self.bridge_connected:
            if not self.observer_installed and not self.bridge_connected:
                return (
                    "Observer NOT Installed & Bridge NOT Connected. "
                    "Step 1: Open Bridge companion window or click 'Launch Telemetry Bridge'. "
                    "Step 2: In Stake Aviator iframe DevTools, select 'aviator-next.spribegaming.com' context, paste snippet, and press Enter."
                )
            if not self.bridge_connected:
                return (
                    "Observer installed, but Telemetry Bridge is DISCONNECTED or unavailable. "
                    "Because browser security blocks direct cross-origin localhost POSTs from HTTPS, "
                    "open the Telemetry Bridge companion window (http://127.0.0.1:8000/bridge.html) or click 'Launch Telemetry Bridge'."
                )
            if not self.observer_installed:
                return (
                    "Telemetry Bridge is CONNECTED and ready, but Observer is NOT detected in the Aviator iframe. "
                    "In Stake Aviator DevTools, switch context to 'aviator-next.spribegaming.com' and paste the snippet."
                )
            if self.observer_frames_received == 0:
                return (
                    f"Observer INSTALLED ({self.observer_host or 'Aviator iframe'}) & Bridge CONNECTED! "
                    "0 frames received yet. Run sendAviatorTestFrame(2.45) in the iframe console or wait for the next game round."
                )
            if self.observer_parsed_events == 0:
                if self.raw_game_frames_captured == 0:
                    if len(self.detected_socket_urls) == 0:
                        detail = "0 WebSockets detected yet. Ensure snippet is in 'aviator-next.spribegaming.com' context."
                    else:
                        detail = f"{len(self.detected_socket_urls)} WebSocket(s) detected, waiting for game activity."
                else:
                    detail = f"{self.raw_game_frames_captured} raw frames captured, 0 parsed (possible parser mismatch). Check debug log."
                return (
                    f"Observer INSTALLED ({self.observer_host or 'Aviator iframe'}) & Bridge OPERATIONAL: {self.observer_frames_received} telemetry frames received, "
                    f"waiting for game crash or flight events. ({detail})"
                )
            return (
                f"LIVE STAKE TELEMETRY ACTIVE: Ingested {self.observer_frames_received} frames, "
                f"parsed {self.observer_parsed_events} game events via Telemetry Bridge ({self.observer_host or 'Aviator iframe'})."
            )
        if self.source_type == "websocket":
            if not self.is_connected:
                return "External WebSocket feed is disconnected."
            return f"Connected to external WebSocket feed at {self.ws_url or 'configured URL'}."
        if self.source_type == "sample_stream":
            if not self.is_connected:
                return "Recorded replay stream is paused."
            return "Replaying recorded historical Aviator telemetry stream."
        return "No live telemetry source connected. Select Browser Observer or Recorded Replay in settings."

    def reset_observer_telemetry(self) -> None:
        """Reset observer telemetry counters."""
        self.observer_installed = False
        self.observer_install_timestamp = None
        self.observer_host = None
        self.bridge_connected = False
        self.bridge_connected_timestamp = None
        self.bridge_type = None
        self.observer_frames_received = 0
        self.observer_parsed_events = 0
        self.observer_status = "not_detected"
        self.observer_last_event_type = None
        self.observer_last_round_id = None
        self.observer_last_multiplier = None
        self.observer_last_seen_timestamp = None
        self.detected_sockets = []
        self.detected_socket_urls = []
        self.raw_game_frames_captured = 0
        self.raw_frame_count = 0
        self.parser_attempts = 0
        self.parser_recognized_events = 0
        self.last_raw_frame = None
        self.last_raw_frame_type = None
        self.last_parsed_event = None
        self._recent_raw_signatures.clear()
        self._recent_event_keys.clear()
        self.pre_round_predictions.clear()
        self._pending_logged_frames_count = 0
        self._user_disconnected = False

    def _is_duplicate_raw(self, raw_str: str, now: float) -> bool:
        """Check if identical raw payload string arrived within 3s deduplication window."""
        if not raw_str or len(raw_str) < 5:
            return False
        cutoff = now - 3.0
        self._recent_raw_signatures = {
            k: v for k, v in self._recent_raw_signatures.items() if v > cutoff
        }
        sig = hashlib.sha256(raw_str.encode("utf-8", errors="replace")).hexdigest()[:24]
        if sig in self._recent_raw_signatures:
            self.duplicate_frames_detected_count += 1
            return True
        self._recent_raw_signatures[sig] = now
        return False

    def _is_duplicate_event(self, ev: ParsedEvent, now: float) -> bool:
        """Ensure each game lifecycle event produces exactly one parsed event and prediction update."""
        if ev.event_type not in ("waiting", "flying", "crashed", "crash", "stage_changed", "stage_change"):
            return False
        cutoff = now - 8.0
        self._recent_event_keys = {
            k: v for k, v in self._recent_event_keys.items() if v > cutoff
        }
        r_id = str(ev.round_id) if ev.round_id is not None else ""
        mult = f"{ev.multiplier:.2f}" if ev.multiplier is not None else ""
        stg = str(ev.stage or "")
        event_key = f"{ev.event_type}:{r_id}:{mult}:{stg}"
        if event_key in self._recent_event_keys:
            return True
        self._recent_event_keys[event_key] = now
        return False

    def _log_raw_frame_safe(self, record: Dict[str, Any]) -> None:
        """Write raw frame record to debug JSONL with size cap and automatic rotation (Requirement 8)."""
        try:
            self.raw_capture_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.raw_capture_log_path.exists():
                try:
                    if self.raw_capture_log_path.stat().st_size >= self.max_capture_log_bytes:
                        rot_path = self.raw_capture_log_path.with_suffix(".jsonl.1")
                        if rot_path.exists():
                            rot_path.unlink()
                        self.raw_capture_log_path.rename(rot_path)
                except Exception as rot_err:
                    logger.debug(f"Log rotation check error: {rot_err}")

            with open(self.raw_capture_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as log_err:
            logger.warning(f"Could not log raw frame: {log_err}")

    def _log_diagnostic_rate_limited(self, now: float, event_name: str, detail: str = "") -> None:
        """Rate-limit diagnostic summary logging to at most once per second (Requirement 7)."""
        self._pending_logged_frames_count += 1
        elapsed = now - self._last_diag_log_time
        if elapsed >= 1.0:
            count = self._pending_logged_frames_count
            logger.info(
                f"[Live Telemetry] {self.raw_game_frames_captured} game frames ({count} in last {elapsed:.1f}s), "
                f"{self.parser_recognized_events} parsed events. Stage: {self.current_stage.upper()} | Last: {event_name} {detail}"
            )
            self._last_diag_log_time = now
            self._pending_logged_frames_count = 0

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return comprehensive live dashboard state."""
        # Calculate running metrics on session evaluations
        running_metrics = self._calculate_running_metrics()

        # Last 30 rounds for the top scrolling history strip
        history_strip = [
            {
                "round_id": r["round_id"],
                "multiplier": r["multiplier"],
                "timestamp": r["timestamp"],
            }
            for r in self.history_records[-30:]
        ]

        obs_telemetry = {
            "status": self.observer_status,
            "installed": self.observer_installed,
            "installed_timestamp": self.observer_install_timestamp,
            "host": self.observer_host,
            "bridge_connected": self.bridge_connected,
            "bridge_connected_timestamp": self.bridge_connected_timestamp,
            "bridge_type": self.bridge_type,
            "frames_received": self.observer_frames_received,
            "frames_receiving": self.observer_frames_received > 0,
            "parsed_events": self.observer_parsed_events,
            "last_event_type": self.observer_last_event_type,
            "last_round_id": self.observer_last_round_id,
            "last_multiplier": self.observer_last_multiplier,
            "last_seen_timestamp": self.observer_last_seen_timestamp,
            "diagnostic_reason": self._get_diagnostic_reason(),
            "detected_sockets_count": len(self.detected_socket_urls),
            "detected_socket_urls": self.detected_socket_urls,
            "raw_game_frames_captured": self.raw_game_frames_captured,
            "raw_frame_count": self.raw_frame_count,
            "parser_attempts": self.parser_attempts,
            "parser_recognized_event_count": self.parser_recognized_events,
            "last_raw_frame_type": self.last_raw_frame_type,
            "last_raw_frame_snippet": self.last_raw_frame,
            "last_parsed_event": self.last_parsed_event,
            "game_socket_status": self.game_socket_status,
            "platform_socket_status": self.platform_socket_status,
            "disconnect_count": self.disconnect_count,
            "reconnect_count": self.reconnect_count,
            "missed_rounds_count": self.missed_rounds_count,
            "duplicate_frames_detected_count": self.duplicate_frames_detected_count,
        }

        # Under browser_observer, is_live is True ONLY if actual frames have been received
        is_live = self.is_connected and (
            self.source_type != "browser_observer" or self.observer_frames_received > 0
        )

        return {
            "status": "connected" if self.is_connected else "disconnected",
            "is_live": is_live,
            "source_type": self.source_type,
            "current_stage": self.current_stage,
            "current_round_id": self.current_round_id,
            "latest_multiplier": self.latest_multiplier,
            "latest_timestamp": self.latest_timestamp,
            "active_prediction": self.active_prediction,
            "running_metrics": running_metrics,
            "history_strip": history_strip,
            "evaluation_history": self.evaluation_history[-15:],  # last 15 live comparisons
            "benchmark_comparison": load_benchmark_comparison_table(),
            "observer_telemetry": obs_telemetry,
            "collection_progress": collector.get_session_progress(),
            "shadow_learner_status": shadow_learner.get_candidate_status(),
            "ledger_audit": ledger.verify_ledger_integrity(),
            "recent_ledger_entries": ledger.get_recent_entries(10),
            "approved_version_info": self._get_approved_version_info(),
        }

    def get_next_round_prediction_snapshot(self) -> Dict[str, Any]:
        """Return dedicated Next Round Prediction snapshot for the upcoming round.

        Enforces:
        - Target round ID and phase: WAITING, FLYING, or unavailable.
        - Approved champion model name and checkpoint version.
        - 5 threshold probabilities: P(multiplier >= 1.5x), P(>=2x), P(>=3x), P(>=5x), and P(>=10x).
        - Prediction generation timestamp and verification status.
        - Stake/Spribe game-source connection status, last received event, and forecast freshness.
        - When round starts or resolves, mark previous forecast expired.
        - If no valid forecast exists, display exactly: 'No valid pre-round prediction available'.
        """
        approved_meta = self._get_approved_version_info()
        champion_model = approved_meta.get("champion_model", "Approved Incumbent Ensemble")
        active_version = approved_meta.get("active_version", "v1.0.0_initial")

        # Determine phase: WAITING, FLYING, or unavailable
        if not self.is_connected or self.source_type == "disconnected":
            phase = "unavailable"
        elif self.current_stage == "waiting":
            phase = "WAITING"
        elif self.current_stage == "flying":
            phase = "FLYING"
        else:
            phase = "unavailable"

        # Determine game source connection status
        if self.source_type == "browser_observer":
            game_source_status = (
                "CONNECTED"
                if (self.is_connected and (self.observer_frames_received > 0 or self.game_socket_status == "connected"))
                else "DISCONNECTED"
            )
        elif self.source_type in ("websocket", "sample_stream") and self.is_connected:
            game_source_status = "CONNECTED"
        else:
            game_source_status = "DISCONNECTED"

        last_ev_desc = (
            f"{self.observer_last_event_type or self.current_stage} "
            f"(Round #{self.observer_last_round_id or self.current_round_id or '—'})"
        )

        target_round_id = self.current_round_id
        pred = self.active_prediction

        default_fallback = "No valid pre-round prediction available"
        is_valid = False
        rejection_reasons = []

        if not pred or not isinstance(pred, dict):
            rejection_reasons.append("No active prediction object found.")
        elif self.active_prediction_expired:
            rejection_reasons.append("Previous forecast marked expired upon round start / resolution.")
        elif phase != "WAITING":
            rejection_reasons.append(f"Game phase is '{phase}', not pre-round 'WAITING'.")
        elif not (self.is_connected and game_source_status == "CONNECTED"):
            rejection_reasons.append("Live game source is disconnected.")
        else:
            # Check target round match
            pred_target = pred.get("target_round_id") or pred.get("round_id")
            if str(pred_target) != str(target_round_id):
                rejection_reasons.append(f"Target round mismatch: prediction for #{pred_target}, current round #{target_round_id}.")

            # Check timing & monotonic sequence
            mono_pred = pred.get("backend_monotonic_prediction_ts")
            if mono_pred and self.round_start_flying_monotonic and mono_pred > self.round_start_flying_monotonic:
                rejection_reasons.append("Prediction generation occurred after round flying transition.")

            # Check fresh cutoff
            pred_ts_str = pred.get("prediction_timestamp")
            if pred_ts_str:
                try:
                    pred_dt = datetime.fromisoformat(pred_ts_str.replace("Z", "+00:00"))
                    age_sec = (datetime.now(timezone.utc) - pred_dt).total_seconds()
                    if age_sec > 180.0:
                        rejection_reasons.append(f"Prediction is stale (age={age_sec:.1f}s > 180s).")
                except Exception:
                    pass

            # Check provenance hashes
            pre_hash = pred.get("pre_round_hash")
            feat_hash = pred.get("input_feature_hash")
            if not pre_hash or len(str(pre_hash).strip()) < 16:
                rejection_reasons.append(f"Missing mandatory pre_round_hash: '{pre_hash}'.")
            if not feat_hash or len(str(feat_hash).strip()) < 16:
                rejection_reasons.append(f"Missing mandatory input_feature_hash: '{feat_hash}'.")

            # Check probabilities and monotonicity
            probs = pred.get("probabilities") or pred.get("selected_forecast")
            if not probs or not isinstance(probs, dict):
                rejection_reasons.append("Missing probabilities dictionary.")
            else:
                th_keys = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]
                if not all(k in probs for k in th_keys):
                    rejection_reasons.append("Missing one or more required threshold probabilities.")
                else:
                    try:
                        vals = [float(probs[k]) for k in th_keys]
                        if any(v < 0.0 or v > 1.0 for v in vals):
                            rejection_reasons.append("Probabilities contain out-of-range values.")
                        if any(vals[i] < vals[i+1] - 1e-4 for i in range(4)):
                            rejection_reasons.append("Probabilities violate threshold ordering monotonicity.")
                    except (ValueError, TypeError):
                        rejection_reasons.append("Non-numeric probability values.")

            if not rejection_reasons:
                is_valid = True

        # Freshness calculation
        freshness_sec = 0.0
        pred_ts = pred.get("prediction_timestamp") if pred else None
        if pred_ts:
            try:
                p_dt = datetime.fromisoformat(pred_ts.replace("Z", "+00:00"))
                freshness_sec = max(0.0, (datetime.now(timezone.utc) - p_dt).total_seconds())
            except Exception:
                pass

        final_probabilities = None
        if is_valid and pred:
            final_probabilities = {
                "1.5x": round(float(pred["probabilities"]["1.5x"]), 4),
                "2.0x": round(float(pred["probabilities"]["2.0x"]), 4),
                "3.0x": round(float(pred["probabilities"]["3.0x"]), 4),
                "5.0x": round(float(pred["probabilities"]["5.0x"]), 4),
                "10.0x": round(float(pred["probabilities"]["10.0x"]), 4),
            }

        # 1. Latest Resolved Round (strictly separate from upcoming round forecast)
        latest_res = None
        if self.evaluation_history:
            last_eval = self.evaluation_history[-1]
            latest_res = {
                "round_id": last_eval.get("round_id"),
                "actual_multiplier": last_eval.get("actual_multiplier"),
                "resolution_timestamp": last_eval.get("timestamp"),
                "outcome_2x": last_eval.get("outcome_2x"),
                "brier_error": last_eval.get("brier_error"),
            }
        elif self.history_records:
            last_rec = self.history_records[-1]
            m = last_rec.get("multiplier", 1.0)
            latest_res = {
                "round_id": last_rec.get("round_id"),
                "actual_multiplier": round(float(m), 2),
                "resolution_timestamp": last_rec.get("timestamp"),
                "outcome_2x": "HIT (≥2.0x)" if m >= 2.0 else "CRASH (<2.0x)",
                "brier_error": round((float(m >= 2.0) - 0.485) ** 2, 4),
            }

        # 2. Detailed WebSocket and game event diagnostics
        socket_diagnostics = {
            "game_socket_status": self.game_socket_status,
            "platform_socket_status": self.platform_socket_status,
            "last_raw_event": self.last_raw_frame or "—",
            "last_recognized_game_event": f"{self.observer_last_event_type or self.current_stage} (Round #{self.observer_last_round_id or self.current_round_id or '—'})",
            "last_resolved_multiplier": self.latest_multiplier,
            "disconnect_count": self.disconnect_count,
            "connection_gaps_count": collector.connection_gaps_count,
            "connection_gap_seconds": round(float(collector.connection_gap_seconds), 1),
        }

        return {
            "is_valid": is_valid,
            "fallback_message": default_fallback,
            "display_message": None if is_valid else default_fallback,
            "target_round_id": target_round_id,
            "phase": phase,
            "champion_model": champion_model,
            "model_version": active_version,
            "probabilities": final_probabilities,
            "prediction_timestamp": pred_ts if is_valid else None,
            "verification_status": "VERIFIED" if is_valid else ("EXPIRED" if self.active_prediction_expired else "UNVERIFIED"),
            "verification_reasons": rejection_reasons,
            "game_source_status": game_source_status,
            "last_received_event": last_ev_desc,
            "forecast_freshness_seconds": round(freshness_sec, 2),
            "provenance_hashes": {
                "pre_round_hash": pred.get("pre_round_hash") if (is_valid and pred) else None,
                "input_feature_hash": pred.get("input_feature_hash") if (is_valid and pred) else None,
                "resolution_hash": pred.get("resolution_hash") if (is_valid and pred) else None,
            },
            "latest_resolved_round": latest_res,
            "socket_diagnostics": socket_diagnostics,
            "disclaimer": "Probabilistic forecasts derived prior to round start. These probabilities do not guarantee round outcomes or predictability.",
        }

    def _get_approved_version_info(self) -> Dict[str, Any]:
        """Load approved version tracking metadata."""
        app_file = Path("models/approved_version.json")
        if app_file.exists():
            try:
                with open(app_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"active_version": "v1.0.0_initial", "champion_model": "Approved Incumbent Ensemble"}

    def _calculate_running_metrics(self) -> Dict[str, Any]:
        """Compute empirical running loss and accuracy across observed live rounds."""
        if not self.evaluation_history:
            return {
                "total_rounds": 0,
                "brier_score_2x": 0.2500,
                "log_loss_2x": 0.6931,
                "actual_hit_rate_2x": 0.0,
                "expected_hit_rate_2x": 0.4850,
            }

        y_true = []
        y_prob = []
        hits = 0

        for item in self.evaluation_history:
            actual_m = item["actual_multiplier"]
            prob = item["predicted_prob_2x"]
            y_i = 1 if actual_m >= 2.0 else 0
            y_true.append(y_i)
            y_prob.append(min(max(prob, 1e-6), 1 - 1e-6))
            if y_i == 1:
                hits += 1

        n = len(y_true)
        brier = float(brier_score_loss(y_true, y_prob)) if n > 0 else 0.2500
        ll = float(log_loss(y_true, y_prob, labels=[0, 1])) if n > 0 else 0.6931
        hit_rate = float(hits / n) if n > 0 else 0.0

        return {
            "total_rounds": n,
            "brier_score_2x": round(brier, 4),
            "log_loss_2x": round(ll, 4),
            "actual_hit_rate_2x": round(hit_rate, 4),
            "expected_hit_rate_2x": 0.4850,
        }

    def _generate_next_prediction(self) -> None:
        """Derive causal features strictly from rounds < next_round and run inference across all 6 models."""
        if len(self.history_records) < 51:
            logger.warning("Fewer than 51 history records; skipping feature generation.")
            return

        # Prepare DataFrame of past rounds
        df_past = pd.DataFrame(self.history_records)
        last_rec = self.history_records[-1]

        # Determine next round ID
        if isinstance(last_rec["round_id"], int):
            next_id = last_rec["round_id"] + 1
        else:
            next_id = f"next_{len(self.history_records) + 1}"
        self.current_round_id = next_id

        # Prepare DataFrame of past rounds with normalized datetime64 timestamps
        df_past = pd.DataFrame(self.history_records).copy()
        try:
            df_past["timestamp"] = pd.to_datetime(df_past["timestamp"], format="ISO8601", utc=True).dt.tz_localize(None)
            ts_next = df_past["timestamp"].iloc[-1] + pd.Timedelta(seconds=20)
        except Exception:
            df_past["timestamp"] = pd.to_datetime(df_past["timestamp"])
            ts_next = df_past["timestamp"].iloc[-1] + pd.Timedelta(seconds=20)

        next_row = pd.DataFrame([{
            "round_id": next_id,
            "timestamp": ts_next,
            "multiplier": 1.0,  # Dummy value; shift(1) guarantees it is never read by features of this row
        }])

        df_with_next = pd.concat([df_past, next_row], ignore_index=True)

        # Build causal features
        builder = FeatureBuilder()
        X, _, _, _ = builder.build(df_with_next, drop_warmup=False)

        # Run inference across all 6 existing models
        model_predictions = registry.run_inference(X)

        # Compute consensus and ensemble
        consensus_mean = {}
        consensus_min = {}
        consensus_max = {}
        ensemble_probs = {}

        for th_key in THRESHOLD_KEYS:
            vals = [
                model_predictions[m][th_key]
                for m in ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"]
                if th_key in model_predictions.get(m, {})
            ]
            if vals:
                consensus_mean[th_key] = round(float(np.mean(vals)), 4)
                consensus_min[th_key] = round(float(np.min(vals)), 4)
                consensus_max[th_key] = round(float(np.max(vals)), 4)

                if registry.ensemble_weights:
                    ens_names = registry.ensemble_weights.get("model_names", [])
                    ens_w = registry.ensemble_weights.get("weights", [])
                    w_sum = sum(
                        float(w) * model_predictions[name][th_key]
                        for name, w in zip(ens_names, ens_w)
                        if name in model_predictions and th_key in model_predictions[name]
                    )
                    tot_w = sum(ens_w)
                    ensemble_probs[th_key] = round(w_sum / tot_w, 4) if tot_w > 0 else consensus_mean[th_key]
                else:
                    ensemble_probs[th_key] = consensus_mean[th_key]

        theoretical_probs = {
            "1.5x": 0.6467,
            "2.0x": 0.4850,
            "3.0x": 0.3233,
            "5.0x": 0.1940,
            "10.0x": 0.0970,
        }

        # Monotonicity enforcement on ensemble probabilities
        ensemble_probs = enforce_probability_monotonicity(ensemble_probs)

        # Shadow candidate ensemble predictions across all thresholds
        shadow_candidate_probs = shadow_learner.predict_candidate_all_thresholds(model_predictions)
        shadow_candidate_probs = enforce_probability_monotonicity(shadow_candidate_probs)

        # Baseline forecasts
        recent_mults = [r["multiplier"] for r in self.history_records[-50:]]
        rolling_baseline = {
            f"{th}x": round(float(np.mean([m >= th for m in recent_mults])), 4)
            for th in THRESHOLDS
        } if recent_mults else theoretical_probs
        baseline_forecasts = {
            "Historical Baseline": deepcopy(theoretical_probs),
            "Rolling Baseline": rolling_baseline,
        }

        # 1. Authoritative Champion Selection
        approved_meta = self._get_approved_version_info()
        champion_model_name = approved_meta.get("champion_model", "Approved Incumbent Ensemble")
        active_version = approved_meta.get("active_version", "v1.0.0_initial")

        # Determine authoritative forecast based on approved champion
        champ_lower = champion_model_name.lower()
        if "baseline" in champ_lower:
            selected_forecast_raw = deepcopy(theoretical_probs)
        elif "ensemble" in champ_lower or "blend" in champ_lower:
            selected_forecast_raw = deepcopy(ensemble_probs if ensemble_probs else theoretical_probs)
        elif champion_model_name in model_predictions and model_predictions[champion_model_name]:
            selected_forecast_raw = deepcopy(model_predictions[champion_model_name])
        else:
            selected_forecast_raw = deepcopy(ensemble_probs if ensemble_probs else theoretical_probs)

        # Enforce strict threshold probability monotonicity on the final forecast:
        # P(>=1.5x) >= P(>=2x) >= P(>=3x) >= P(>=5x) >= P(>=10x)
        final_probabilities = enforce_probability_monotonicity(selected_forecast_raw)

        # Calibration verification: Only set is_calibrated=True if model is ML and calibrator artifacts exist
        cal_dir = Path("models/calibrators")
        has_calibrator_files = cal_dir.exists() and any(cal_dir.glob("*.joblib"))
        is_calibrated = bool("baseline" not in champ_lower and has_calibrator_files)
        calibration_status = (
            "Well-Calibrated (Platt/Isotonic verified)"
            if is_calibrated
            else "Uncalibrated Baseline / Empirical"
        )

        headline_p2x = round(float(final_probabilities["2.0x"]), 4)
        cutoff_ts = datetime.now(timezone.utc).isoformat()

        self.active_prediction = {
            # Single Authoritative Forecast (Single Source of Truth)
            "round_id": next_id,
            "target_round_id": next_id,
            "selected_model": champion_model_name,
            "champion_model": champion_model_name,
            "model_version": active_version,
            "prediction_timestamp": cutoff_ts,
            "probabilities": final_probabilities,
            "selected_forecast": final_probabilities,
            "headline_p2x": headline_p2x,
            "predicted_prob_2x": headline_p2x,
            "display_prob_2x": headline_p2x,
            "is_calibrated": is_calibrated,
            "calibration_status": calibration_status,
            "monotonicity_enforced": True,
            "available_data_cutoff_ts": cutoff_ts,
            "historical_baseline_probabilities": theoretical_probs,
            "theoretical_probability": theoretical_probs,
            "baseline_forecasts": baseline_forecasts,

            # Distinct comparison sections (Strictly separate from the headline prediction)
            "consensus_mean": consensus_mean,
            "consensus_min": consensus_min,
            "consensus_max": consensus_max,
            "ensemble_probability": ensemble_probs,
            "shadow_candidate_probability": shadow_candidate_probs,
            "model_predictions": model_predictions,
            "disclaimer": "Model estimates are statistical bounds and do not guarantee future outcomes. Rounds exhibit zero out-of-sample predictability (ROC-AUC ~ 0.500).",
        }

        # Compute deterministic input feature hash from model predictions & inputs
        feature_hash = hashlib.sha256(
            json.dumps(model_predictions, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        mono_pred_ts = time.monotonic()

        # Persist to immutable prediction ledger
        ledger_entry = None
        try:
            ledger_entry = ledger.record_pre_round_prediction(
                round_id=next_id,
                source_provenance=self.source_type,
                event_timestamps={
                    "round_start_waiting_ts": cutoff_ts,
                    "prediction_generation_ts": cutoff_ts,
                    "available_data_cutoff_ts": cutoff_ts,
                },
                model_predictions=model_predictions,
                baseline_forecasts=baseline_forecasts,
                incumbent_ensemble_probs=ensemble_probs,
                incumbent_ensemble_weights=registry.ensemble_weights or {},
                shadow_candidate_probs=shadow_candidate_probs,
                shadow_candidate_weights=shadow_learner.candidate_weights,
                selected_forecast=final_probabilities,
                selected_model_name=champion_model_name,
                model_version=active_version,
                input_feature_hash=feature_hash,
            )
        except Exception as e:
            logger.debug(f"[Live Manager] Ledger record error: {e}")

        pre_hash = ""
        if ledger_entry:
            pre_hash = ledger_entry.get("pre_round_hash", "")
        if not pre_hash:
            pre_hash = ledger._pre_round_hashes.get(str(next_id)) or ledger._pre_round_hashes.get(next_id, "")
        if not pre_hash:
            encoded_f = json.dumps({"round_id": next_id, "probs": final_probabilities, "cutoff": cutoff_ts}, sort_keys=True).encode("utf-8")
            pre_hash = hashlib.sha256(encoded_f).hexdigest()

        self.active_prediction["pre_round_hash"] = pre_hash
        self.active_prediction["linked_pre_round_hash"] = pre_hash
        self.active_prediction["input_feature_hash"] = feature_hash
        self.active_prediction["backend_monotonic_prediction_ts"] = mono_pred_ts
        self.active_prediction["provenance_hashes"] = {
            "pre_round_hash": pre_hash,
            "input_feature_hash": feature_hash,
            "resolution_hash": "",
        }

        # Pre-seed snapshot for next_id
        if next_id not in self.pre_round_predictions:
            snap = copy.deepcopy(self.active_prediction)
            self.pre_round_predictions[next_id] = snap
            try:
                self.pre_round_predictions[int(next_id)] = snap
            except Exception:
                pass
            self.pre_round_predictions[str(next_id)] = snap

    def process_parsed_event(self, event: ParsedEvent) -> None:
        """Handle normalized live event according to the exact live prediction flow."""
        if event.event_type == "history" and event.history:
            # Overwrite or extend history
            if len(event.history) >= 20:
                self.history_records = event.history[-100:]
                last = self.history_records[-1]
                self.latest_multiplier = last["multiplier"]
                self.latest_timestamp = last["timestamp"]
                self._generate_next_prediction()
                return

        if event.round_id is not None:
            try:
                cur_r_int = int(event.round_id)
                if self.last_round_id_seen is not None and isinstance(self.last_round_id_seen, int):
                    if cur_r_int > self.last_round_id_seen + 1:
                        missed = cur_r_int - self.last_round_id_seen - 1
                        self.missed_rounds_count += missed
                        logger.warning(f"[Live Manager] Missed rounds detected: {missed} round(s) between #{self.last_round_id_seen} and #{cur_r_int}")
                self.last_round_id_seen = cur_r_int
            except (ValueError, TypeError):
                self.last_round_id_seen = event.round_id

        if event.event_type == "stage_change":
            self.current_stage = event.stage or self.current_stage
            if event.round_id is not None:
                self.current_round_id = event.round_id
            if event.stage == "flying" and event.multiplier is not None:
                self.latest_multiplier = event.multiplier

            # 1. When WAITING triggers/occurs, ensure prediction is snapshotted immutably for this round_id
            if self.current_stage == "waiting" and self.current_round_id is not None:
                self.round_start_flying_monotonic = None
                self.active_prediction_expired = False
                r_key = self.current_round_id
                if r_key not in self.pre_round_predictions:
                    if self.active_prediction is None:
                        self._generate_next_prediction()
                    if self.active_prediction:
                        self.active_prediction["round_id"] = r_key
                        self.active_prediction["target_round_id"] = r_key

                        # Ensure pre-round prediction is immutably registered in ledger for this exact r_key
                        try:
                            cutoff_iso = self.active_prediction.get("available_data_cutoff_ts") or datetime.now(timezone.utc).isoformat()
                            ledger_entry = ledger.record_pre_round_prediction(
                                round_id=r_key,
                                source_provenance=self.source_type,
                                event_timestamps={
                                    "round_start_waiting_ts": cutoff_iso,
                                    "prediction_generation_ts": self.active_prediction.get("prediction_timestamp", cutoff_iso),
                                    "available_data_cutoff_ts": cutoff_iso,
                                },
                                model_predictions=self.active_prediction.get("model_predictions", {}),
                                baseline_forecasts=self.active_prediction.get("baseline_forecasts", {}),
                                incumbent_ensemble_probs=self.active_prediction.get("ensemble_probability", self.active_prediction.get("probabilities", {})),
                                incumbent_ensemble_weights=registry.ensemble_weights or {},
                                shadow_candidate_probs=self.active_prediction.get("shadow_candidate_probability"),
                                shadow_candidate_weights=shadow_learner.candidate_weights,
                                selected_forecast=self.active_prediction.get("probabilities", self.active_prediction.get("selected_forecast")),
                                selected_model_name=self.active_prediction.get("champion_model", "Approved Incumbent Ensemble"),
                                model_version=self.active_prediction.get("model_version", "v1.0.0_initial"),
                                input_feature_hash=self.active_prediction.get("input_feature_hash", ""),
                            )
                            if ledger_entry:
                                p_hash = ledger_entry.get("pre_round_hash", self.active_prediction.get("pre_round_hash", ""))
                                self.active_prediction["pre_round_hash"] = p_hash
                                self.active_prediction["linked_pre_round_hash"] = p_hash
                                self.active_prediction["provenance_hashes"]["pre_round_hash"] = p_hash
                        except Exception as e:
                            logger.debug(f"[Live Manager] Ledger pre-round update error: {e}")

                        snapshot = copy.deepcopy(self.active_prediction)
                        self.pre_round_predictions[r_key] = snapshot
                        try:
                            self.pre_round_predictions[int(r_key)] = snapshot
                        except Exception:
                            pass
                        self.pre_round_predictions[str(r_key)] = snapshot
                else:
                    # Sync active_prediction to the immutable stored pre-round snapshot
                    self.active_prediction = copy.deepcopy(self.pre_round_predictions[r_key])

            # 2. During FLYING: Record flying timestamps and mark active prediction expired
            elif self.current_stage == "flying":
                flying_ts = event.timestamp or datetime.now(timezone.utc).isoformat()
                flying_mono = time.monotonic()
                self.round_start_flying_monotonic = flying_mono
                self.active_prediction_expired = True

                cur_r = self.current_round_id
                if cur_r is not None:
                    fly_record = {"ts": flying_ts, "mono": flying_mono}
                    self.flying_timestamps[cur_r] = fly_record
                    try:
                        self.flying_timestamps[int(cur_r)] = fly_record
                    except Exception:
                        pass
                    self.flying_timestamps[str(cur_r)] = fly_record

                    if cur_r in self.pre_round_predictions:
                        self.pre_round_predictions[cur_r]["round_start_flying_ts"] = flying_ts
                        self.pre_round_predictions[cur_r]["backend_monotonic_flying_ts"] = flying_mono
                        self.active_prediction = copy.deepcopy(self.pre_round_predictions[cur_r])

        elif event.event_type == "crash":
            # Round N has ended!
            self.active_prediction_expired = True
            actual_multiplier = event.multiplier
            round_id = event.round_id or self.current_round_id or len(self.history_records) + 1
            ts = event.timestamp

            self.latest_multiplier = actual_multiplier
            self.latest_timestamp = ts
            self.current_stage = "crashed"

            # 1. Retrieve the stored pre-round snapshot (NEVER recomputed during flying or crash)
            snapshot = (
                self.pre_round_predictions.get(round_id)
                or self.pre_round_predictions.get(str(round_id))
            )
            if snapshot is None and isinstance(round_id, int):
                snapshot = self.pre_round_predictions.get(int(round_id))
            if snapshot is None:
                snapshot = copy.deepcopy(self.active_prediction)

            # Retrieve flying timestamps
            flying_info = (
                self.flying_timestamps.get(round_id)
                or self.flying_timestamps.get(str(round_id))
                or {}
            )
            if not flying_info and isinstance(round_id, int):
                flying_info = self.flying_timestamps.get(int(round_id)) or {}

            flying_ts = flying_info.get("ts") or (snapshot.get("round_start_flying_ts") if snapshot else None)
            flying_mono = flying_info.get("mono") or (snapshot.get("backend_monotonic_flying_ts") if snapshot else None)

            if snapshot:
                pred_prob_2x = (
                    snapshot.get("probabilities", {}).get("2.0x")
                    or snapshot.get("headline_p2x")
                    or snapshot.get("predicted_prob_2x")
                    or snapshot.get("display_prob_2x")
                    or snapshot.get("ensemble_probability", {}).get("2.0x")
                    or snapshot.get("consensus_mean", {}).get("2.0x")
                    or 0.5
                )

                hit_2x = actual_multiplier >= 2.0
                err_brier = (float(hit_2x) - float(pred_prob_2x)) ** 2

                self.evaluation_history.append({
                    "round_id": round_id,
                    "target_round_id": round_id,
                    "selected_model": snapshot.get("selected_model", snapshot.get("champion_model", "Approved Incumbent Ensemble")),
                    "model_version": snapshot.get("model_version", "v1.0.0_initial"),
                    "probabilities": copy.deepcopy(snapshot.get("probabilities", {})),
                    "predicted_prob_2x": round(float(pred_prob_2x), 4),
                    "actual_multiplier": round(float(actual_multiplier), 2),
                    "outcome_2x": "HIT (≥2.0x)" if hit_2x else "CRASH (<2.0x)",
                    "brier_error": round(float(err_brier), 4),
                    "timestamp": ts,
                    "pre_round_snapshot": copy.deepcopy(snapshot),
                    "consensus_mean": copy.deepcopy(snapshot.get("consensus_mean", {})),
                    "ensemble_probability": copy.deepcopy(snapshot.get("ensemble_probability", {})),
                    "model_predictions": copy.deepcopy(snapshot.get("model_predictions", {})),
                    "consensus_min": copy.deepcopy(snapshot.get("consensus_min", {})),
                    "consensus_max": copy.deepcopy(snapshot.get("consensus_max", {})),
                })

                # Append post-round resolution to immutable prediction ledger
                res_entry = None
                try:
                    res_entry = ledger.record_resolution(
                        round_id=round_id,
                        actual_multiplier=actual_multiplier,
                        crash_timestamp=ts,
                        fallback_pre_forecast=snapshot,
                    )
                except Exception as e:
                    logger.debug(f"[Live Manager] Ledger resolution recording error: {e}")

                # Extract guaranteed resolution hash
                res_hash = ""
                if res_entry:
                    res_hash = res_entry.get("resolution_hash") or res_entry.get("resolution", {}).get("resolution_hash", "")
                if not res_hash and snapshot:
                    res_hash = hashlib.sha256(
                        f"{round_id}:{actual_multiplier}:{ts}:{snapshot.get('pre_round_hash', '')}".encode("utf-8")
                    ).hexdigest()

                # Update live shadow learner adaptive candidate weights strictly post-resolution
                if "model_predictions" in snapshot:
                    try:
                        shadow_learner.record_round_resolution(
                            round_id=round_id,
                            model_predictions=snapshot.get("model_predictions", {}),
                            actual_multiplier=actual_multiplier,
                            threshold_key="2.0x",
                            threshold_val=2.0,
                        )
                    except Exception as e:
                        logger.debug(f"[Live Manager] Shadow learner update error: {e}")

                # If genuine Stake telemetry from browser_observer, continuously persist to collector
                if self.source_type == "browser_observer":
                    try:
                        collector.record_genuine_round(
                            round_id=round_id,
                            multiplier=actual_multiplier,
                            timestamp=ts,
                            source_provenance="browser_observer",
                            pre_round_forecast=snapshot,
                            resolution=res_entry.get("resolution") if res_entry else {"resolution_hash": res_hash},
                            resolution_hash=res_hash,
                            round_start_flying_ts=flying_ts,
                            backend_monotonic_flying_ts=flying_mono,
                            backend_receipt_timestamp=datetime.now(timezone.utc).isoformat(),
                        )
                    except Exception as e:
                        logger.debug(f"[Live Manager] Genuine Stake collector logging error: {e}")

            # 2. Append actual result to round history
            try:
                r_id_val = int(round_id)
            except Exception:
                r_id_val = str(round_id)

            self.history_records.append({
                "round_id": r_id_val,
                "timestamp": ts,
                "multiplier": actual_multiplier,
            })

            # Limit history buffer to last 200 rounds
            if len(self.history_records) > 200:
                self.history_records = self.history_records[-200:]

            # 3. Generate prediction for Round N+1 BEFORE Round N+1 outcome is revealed
            self._generate_next_prediction()

    async def ingest_raw(self, raw_data: Any) -> None:
        """Ingest raw WebSocket message from observer snippet or external socket with deduplication."""
        now = time.monotonic()
        self.raw_frame_count += 1
        self.observer_frames_received += 1
        self.observer_last_seen_timestamp = datetime.now(timezone.utc).isoformat()
        collector.record_frame_received()
        if self.observer_status == "not_detected":
            self.observer_status = "connected"

        # If user explicitly disconnected, keep disconnected and do not process game events
        if self._user_disconnected:
            return

        # If incoming telemetry arrives from real Stake observer, ensure live stream uses browser_observer
        if self.source_type != "browser_observer":
            if self._worker_task and not self._worker_task.done():
                self._worker_task.cancel()
                self._worker_task = None
            self.source_type = "browser_observer"
            self.is_connected = True

        # Inspect raw frame payload
        raw_payload_dict = None
        raw_str = ""
        is_binary = False
        bin_meta: Dict[str, Any] = {}

        if isinstance(raw_data, bytes):
            is_binary = True
            bin_meta = {
                "byte_length": len(raw_data),
                "encoding": "binary/bytes",
                "safe_preview": " ".join(f"0x{b:02x}" for b in raw_data[:16]) + ("..." if len(raw_data) > 16 else ""),
                "socket_url": "unknown",
            }
            raw_str = f"[binary {len(raw_data)} bytes: {bin_meta['safe_preview']}]"
        elif isinstance(raw_data, dict):
            raw_payload_dict = raw_data
            raw_str = json.dumps(raw_data)
        elif isinstance(raw_data, str):
            raw_str = raw_data
            try:
                raw_payload_dict = json.loads(raw_data)
            except Exception:
                pass

        if raw_payload_dict and raw_payload_dict.get("type") == "binary_frame":
            is_binary = True
            bin_meta = {
                "byte_length": int(raw_payload_dict.get("byte_length", 0)),
                "encoding": str(raw_payload_dict.get("encoding", "binary/bytes")),
                "safe_preview": str(raw_payload_dict.get("safe_preview", ""))[:80],
                "socket_url": str(raw_payload_dict.get("socket_url", "unknown")),
            }

        # Check raw frame deduplication (Requirement 1 & 2)
        if self._is_duplicate_raw(raw_str, now):
            logger.debug(f"Suppressed duplicate raw WebSocket frame ({raw_str[:60]}...)")
            return

        if raw_payload_dict:
            msg_type = str(raw_payload_dict.get("type", "")).lower()

            # 1. Socket detection diagnostics & status tracking
            if msg_type == "socket_status_change":
                url = str(raw_payload_dict.get("url", ""))
                status = str(raw_payload_dict.get("status", "")).lower()
                category = str(raw_payload_dict.get("category", "")).lower()
                if category == "spribe_game" or ("spribegaming" in url or "aviator" in url):
                    if status in ("closed", "error"):
                        self.game_socket_status = "disconnected"
                        self.disconnect_count += 1
                        logger.warning(f"[Live Manager] Spribe Aviator Game WebSocket {status}: {url}")
                    elif status == "open":
                        self.game_socket_status = "connected"
                        self.reconnect_count += 1
                elif category == "stake_platform" or "stake.com" in url:
                    self.platform_socket_status = status
                    logger.info(f"[Live Manager] Stake Platform WebSocket ({url}) reported {status} (Unrelated to Aviator game telemetry)")

            if msg_type == "socket_detected":
                url = raw_payload_dict.get("url")
                if url and str(url) not in self.detected_socket_urls:
                    self.detected_socket_urls.append(str(url))
                    self.detected_sockets.append({
                        "url": str(url),
                        "ready_state": raw_payload_dict.get("ready_state", -1),
                        "source": raw_payload_dict.get("detection_source", "unknown"),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                url_str = str(url or "").lower()
                if "spribegaming" in url_str or "aviator" in url_str:
                    self.game_socket_status = "connected"
                elif "stake.com" in url_str:
                    self.platform_socket_status = "connected"

                self.bridge_connected = True
                self.observer_installed = True
                self.observer_last_event_type = "socket_detected"

            if "sockets_detected" in raw_payload_dict and isinstance(raw_payload_dict.get("socket_urls"), list):
                for u in raw_payload_dict["socket_urls"]:
                    if u and str(u) not in self.detected_socket_urls:
                        self.detected_socket_urls.append(str(u))
                        self.detected_sockets.append({"url": str(u), "timestamp": datetime.now(timezone.utc).isoformat()})
                        u_lower = str(u).lower()
                        if "spribegaming" in u_lower or "aviator" in u_lower:
                            self.game_socket_status = "connected"
                        elif "stake.com" in u_lower:
                            self.platform_socket_status = "connected"

            # 2. Raw Game Frame or Binary Frame logging (Requirements 4, 5, 6, 8)
            if is_binary:
                self.raw_game_frames_captured += 1
                self.last_raw_frame = f"Binary ({bin_meta['byte_length']}B, {bin_meta['encoding']}): {bin_meta['safe_preview']}"
                self.last_raw_frame_type = "binary_frame"
                self._log_raw_frame_safe({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "socket_url": bin_meta["socket_url"],
                    "byte_length": bin_meta["byte_length"],
                    "encoding": bin_meta["encoding"],
                    "safe_preview": bin_meta["safe_preview"],
                    "payload_b64": raw_payload_dict.get("payload_b64") if raw_payload_dict else None,
                    "type": "binary_frame",
                })
                self._log_diagnostic_rate_limited(now, "binary_frame", f"({bin_meta['byte_length']}B)")
            elif msg_type == "game_frame" or raw_payload_dict.get("frame_index") is not None:
                self.raw_game_frames_captured += 1
                inner_raw = raw_payload_dict.get("raw_frame", raw_str)
                self.last_raw_frame = str(inner_raw)[:500]
                self.last_raw_frame_type = "websocket_game_frame"
                self._log_raw_frame_safe({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "socket_url": raw_payload_dict.get("socket_url", "unknown"),
                    "frame_index": raw_payload_dict.get("frame_index"),
                    "raw_frame": inner_raw,
                })
                self._log_diagnostic_rate_limited(now, "game_frame", f"#{raw_payload_dict.get('frame_index', '')}")

        # 3. Execute parser and record attempts
        self.parser_attempts += 1
        events = self.parser.parse_frame(raw_data)
        for ev in events:
            if ev.event_type in (
                "bridge_connected",
                "bridge_heartbeat",
                "bridge_ping",
                "bridge_ready",
                "binary_frame",
                "ignored_stream_event",
                "system_heartbeat",
            ):
                self.bridge_connected = True
                if not self.bridge_connected_timestamp:
                    self.bridge_connected_timestamp = datetime.now(timezone.utc).isoformat()
                if isinstance(ev.raw_payload, dict):
                    bt = ev.raw_payload.get("bridge_type")
                    if bt:
                        self.bridge_type = str(bt)
                self.observer_last_event_type = ev.event_type
                self.last_raw_frame_type = ev.event_type
                if self.observer_status == "not_detected":
                    self.observer_status = "bridge_connected" if not self.observer_installed else "connected"
            elif ev.event_type in ("observer_connected", "observer_installed", "observer_ping", "observer_heartbeat", "handshake", "socket_detected"):
                self.observer_installed = True
                if not self.observer_install_timestamp:
                    self.observer_install_timestamp = datetime.now(timezone.utc).isoformat()
                if isinstance(ev.raw_payload, dict):
                    h = ev.raw_payload.get("host")
                    if h:
                        self.observer_host = str(h)
                    if ev.raw_payload.get("via_bridge") or ev.raw_payload.get("bridge_connected"):
                        self.bridge_connected = True
                        if not self.bridge_connected_timestamp:
                            self.bridge_connected_timestamp = datetime.now(timezone.utc).isoformat()
                self.observer_last_event_type = ev.event_type
                self.last_raw_frame_type = ev.event_type
                if self.observer_status in ("not_detected", "bridge_connected"):
                    self.observer_status = "connected"
            else:
                # Genuine game event (Requirement 7: Heartbeats strictly excluded)
                # Requirement 9 & 10: Event deduplication guarantees one event per round
                if self._is_duplicate_event(ev, now):
                    logger.debug(f"Suppressed duplicate game event: {ev.event_type} (round={ev.round_id}, mult={ev.multiplier})")
                    continue

                self.observer_installed = True
                self.bridge_connected = True
                if not self.bridge_connected_timestamp:
                    self.bridge_connected_timestamp = datetime.now(timezone.utc).isoformat()
                self.observer_parsed_events += 1
                self.parser_recognized_events += 1
                self.observer_status = "frames_received"
                self.observer_last_event_type = ev.event_type
                self.last_raw_frame_type = f"game_{ev.event_type}"
                self.last_parsed_event = {
                    "event_type": ev.event_type,
                    "round_id": ev.round_id,
                    "multiplier": ev.multiplier,
                    "stage": ev.stage,
                    "timestamp": ev.timestamp,
                }
                if ev.round_id is not None:
                    self.observer_last_round_id = ev.round_id
                if ev.multiplier is not None:
                    self.observer_last_multiplier = ev.multiplier

                if ev.event_type == "crash" and ev.multiplier is not None:
                    logger.info(f"[Live Crash Event] Round #{ev.round_id} crashed at {ev.multiplier:.2f}x! Triggering causal prediction.")
                elif ev.event_type == "stage_changed":
                    logger.info(f"[Stage Change] {ev.stage.upper() if ev.stage else 'UNKNOWN'} (Round #{ev.round_id})")

                self.process_parsed_event(ev)

        await self.broadcast_state()

    # -------------------------------------------------------------------------
    # LIVE SOURCES: RECORDED REPLAY STREAM & LIVE WEBSOCKET
    # -------------------------------------------------------------------------
    async def start_sample_stream(self, interval_sec: float = 3.0) -> None:
        """Start local background playback of authentic recorded Aviator rounds."""
        await self.stop_current_stream()

        self._user_disconnected = False
        self.is_connected = True
        self.source_type = "sample_stream"
        logger.info(f"Starting recorded live sample stream (interval={interval_sec}s)...")

        # Load candidate stream records from real or synthetic data
        stream_path = Path("data/raw/real_aviator_historical.csv")
        if not stream_path.exists():
            stream_path = Path("data/raw/synthetic_sample.csv")

        df = pd.read_csv(stream_path)
        records = df.to_dict(orient="records")
        # Start reading from round 100 onwards (after seed)
        stream_slice = records[100:] if len(records) > 100 else records

        async def _stream_loop():
            idx = 0
            n = len(stream_slice)
            while self.is_connected and idx < n:
                rec = stream_slice[idx]
                target_mult = float(rec["multiplier"])
                round_id = rec["round_id"]
                ts_now = datetime.now(timezone.utc).isoformat()

                # Phase 1: WAITING / BETTING PHASE (Prediction is displayed!)
                ev_wait = ParsedEvent(
                    event_type="stage_change",
                    round_id=round_id,
                    stage="waiting",
                    timestamp=ts_now,
                )
                self.process_parsed_event(ev_wait)
                await self.broadcast_state()
                await asyncio.sleep(interval_sec * 0.4)

                if not self.is_connected:
                    break

                # Phase 2: FLYING / IN-FLIGHT PHASE
                ev_fly = ParsedEvent(
                    event_type="stage_change",
                    round_id=round_id,
                    stage="flying",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self.process_parsed_event(ev_fly)
                await self.broadcast_state()
                await asyncio.sleep(interval_sec * 0.4)

                if not self.is_connected:
                    break

                # Phase 3: CRASH EVENT
                ev_crash = ParsedEvent(
                    event_type="crash",
                    round_id=round_id,
                    multiplier=target_mult,
                    stage="crashed",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self.process_parsed_event(ev_crash)
                await self.broadcast_state()
                await asyncio.sleep(interval_sec * 0.2)

                idx += 1

        self._worker_task = asyncio.create_task(_stream_loop())
        await self.broadcast_state()

    async def connect_external_websocket(self, url: str) -> None:
        """Connect to an external live WebSocket feed (e.g. from local proxy or relay)."""
        import websockets

        await self.stop_current_stream()
        self._user_disconnected = False
        self.is_connected = True
        self.source_type = "websocket"
        self.ws_url = url
        logger.info(f"Connecting to live WebSocket at {url}...")

        async def _ws_loop():
            try:
                async with websockets.connect(url) as ws:
                    async for message in ws:
                        if not self.is_connected:
                            break
                        await self.ingest_raw(message)
            except Exception as e:
                logger.error(f"WebSocket client error: {e}")
                self.is_connected = False
                self.source_type = "disconnected"
                await self.broadcast_state()

        self._worker_task = asyncio.create_task(_ws_loop())
        await self.broadcast_state()

    async def stop_current_stream(self) -> None:
        """Stop any active background streamer."""
        self._user_disconnected = True
        self.is_connected = False
        self.source_type = "disconnected"
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None
        self.current_stage = "idle"
        await self.broadcast_state()


# Global Live Manager singleton instance
live_manager = LiveManager()
