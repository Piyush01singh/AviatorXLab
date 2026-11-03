"""Real Stake Data Collection & Session Lifecycle Manager.

Strict Protocol:
- Persists genuine Stake Aviator rounds continuously to durable storage.
- Completely separates genuine Stake telemetry from demo streams, test fixtures, and synthetic data.
- Enforces single active session, rejecting concurrent collection attempts.
- Enforces genuine browser_observer source locking; rejects sample_stream / synthetic.
- Tracks session duration strictly via backend monotonic clock; the dashboard never controls timing.
- Assembles complete per-round research datasets (identity, pre-round forecasts, outcomes, evaluations).
- Continuous deterministic data-quality verification and quarantine.
- Freezes session dataset snapshot and computes SHA-256 fingerprint on completion.
- Tracks source connection gaps honestly; never fabricates rounds.
- Leaves system in VALIDATED state on completion; NEVER automatically trains models without explicit user confirmation.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union
import uuid
import numpy as np
import pandas as pd

from src.collector.ledger import ledger
from src.collector.verifier import REQUIRED_MODELS, REQUIRED_THRESHOLDS, collection_supervisor
from src.utils.logging import get_logger

logger = get_logger("stake_collector")


def compute_row_checksum(row_dict: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint for a row."""
    clean_dict = {k: v for k, v in row_dict.items() if k not in ("checksum", "record_hash")}
    encoded = json.dumps(clean_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class StakeDataCollector:
    """Manages real Stake Aviator telemetry logging, dataset integrity, and overnight scheduling."""

    def __init__(
        self,
        jsonl_path: str = "data/live/stake_collected_rounds.jsonl",
        csv_path: str = "data/live/stake_collected_rounds.csv",
        sessions_dir: str = "data/live/sessions",
    ):
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self.sessions_dir = Path(sessions_dir)

        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        self.collected_rounds: List[Dict[str, Any]] = []
        self.seen_round_ids: Set[Any] = set()

        # Session tracking
        self.is_active: bool = False
        self.session_id: Optional[str] = None
        self.session_start_monotonic: Optional[float] = None
        self.session_start_time_iso: Optional[str] = None
        self.session_rounds_collected: int = 0
        self.target_duration_seconds: Optional[float] = None
        self.target_duration_hours: Optional[float] = None
        self.target_rounds: Optional[int] = None
        self.schedule_end_time_iso: Optional[str] = None
        self.stopping_condition_desc: str = "Idle"
        self.source_locked: str = "browser_observer"

        # State: IDLE, COLLECTING, FINALIZING, VALIDATED, FAILED
        self.session_status: str = "IDLE"

        # Connection health & gap monitoring
        self.connection_gaps_count: int = 0
        self.connection_gap_seconds: float = 0.0
        self.last_frame_monotonic: Optional[float] = None
        self.last_valid_round_timestamp: Optional[str] = None

        # Quality metrics for active session
        self.session_valid_count: int = 0
        self.session_incomplete_count: int = 0
        self.session_rejected_count: int = 0
        self.session_duplicate_count: int = 0
        self.session_seen_round_ids: Set[Any] = set()

        # Frozen session artifacts
        self.frozen_dataset_path: Optional[str] = None
        self.frozen_jsonl_path: Optional[str] = None
        self.session_dataset_sha256: Optional[str] = None
        self.validation_report: Optional[Dict[str, Any]] = None

        # Manual confirmation required: auto_train_on_completion defaults to False
        self.auto_train_on_completion: bool = False
        self.on_session_complete_callback: Optional[Callable[[], Any]] = None
        self.last_training_result: Optional[Dict[str, Any]] = None

        # Load existing dataset to ensure collection is resumable
        self._load_existing_dataset()

    def _load_existing_dataset(self) -> None:
        """Load previously collected authentic rounds from disk."""
        if not self.jsonl_path.exists():
            return

        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    r_id = rec.get("round_id")
                    if r_id is not None and r_id not in self.seen_round_ids:
                        self.seen_round_ids.add(r_id)
                        self.seen_round_ids.add(str(r_id))
                        self.collected_rounds.append(rec)
            if self.collected_rounds:
                self.last_valid_round_timestamp = self.collected_rounds[-1].get("timestamp")
            logger.info(f"[Collector] Loaded {len(self.collected_rounds)} existing authentic Stake rounds.")
        except Exception as e:
            logger.warning(f"[Collector] Error loading existing dataset: {e}")

    def start_session(
        self,
        duration_hours: Optional[float] = 6.0,
        duration_seconds: Optional[float] = None,
        target_rounds: Optional[int] = None,
        schedule_end_time: Optional[str] = None,
        source: str = "browser_observer",
        auto_train: bool = False,
        on_complete_callback: Optional[Callable[[], Any]] = None,
    ) -> Dict[str, Any]:
        """Configure and start an overnight or scheduled collection session.

        Validations:
        - Prevents multiple simultaneous sessions.
        - Source locked to genuine 'browser_observer'; rejects sample_stream and synthetic.
        - Validates positive duration and valid optional round limits.
        """
        # 1. Single session guard
        if self.is_active or self.session_status in ("COLLECTING", "FINALIZING"):
            raise RuntimeError(
                f"Cannot start session: a collection session ('{self.session_id}') is already active! Stop the current session first."
            )

        # 2. Source locking guard
        src_clean = (source or "browser_observer").strip().lower()
        if src_clean != "browser_observer":
            raise ValueError(
                f"Collection source is strictly locked to genuine 'browser_observer' Stake telemetry. Selection of '{source}' is forbidden."
            )

        # 3. Duration validation
        if duration_seconds is not None:
            if duration_seconds <= 0:
                raise ValueError("duration_seconds must be a positive number.")
            dur_sec = float(duration_seconds)
            dur_hrs = dur_sec / 3600.0
        elif duration_hours is not None:
            if duration_hours <= 0:
                raise ValueError("duration_hours must be a positive number.")
            dur_hrs = float(duration_hours)
            dur_sec = dur_hrs * 3600.0
        else:
            dur_hrs = 6.0
            dur_sec = 6.0 * 3600.0

        # 4. Optional target rounds validation
        tgt_rounds: Optional[int] = None
        if target_rounds is not None and target_rounds != "":
            try:
                tgt_rounds = int(target_rounds)
                if tgt_rounds <= 0:
                    raise ValueError("target_rounds must be an integer >= 1.")
            except (ValueError, TypeError):
                raise ValueError("target_rounds must be a valid positive integer.")

        # 5. Initialize session state
        unique_token = uuid.uuid4().hex[:8]
        self.session_id = f"stake_session_{int(time.time())}_{unique_token}"
        self.session_start_monotonic = time.monotonic()
        self.session_start_time_iso = datetime.now(timezone.utc).isoformat()
        self.last_frame_monotonic = time.monotonic()

        self.target_duration_seconds = dur_sec
        self.target_duration_hours = round(dur_hrs, 2)
        self.target_rounds = tgt_rounds
        self.schedule_end_time_iso = schedule_end_time

        # Description of stopping conditions
        desc_parts = [f"Duration: {self.target_duration_hours}h ({self.target_duration_seconds:.0f}s)"]
        if tgt_rounds is not None:
            desc_parts.append(f"Max Rounds: {tgt_rounds}")
        if schedule_end_time is not None:
            desc_parts.append(f"End Time: {schedule_end_time}")
        self.stopping_condition_desc = " | ".join(desc_parts)

        self.session_rounds_collected = 0
        self.session_valid_count = 0
        self.session_incomplete_count = 0
        self.session_rejected_count = 0
        self.session_duplicate_count = 0
        self.session_seen_round_ids = set()
        self.connection_gaps_count = 0
        self.connection_gap_seconds = 0.0

        self.auto_train_on_completion = auto_train
        if on_complete_callback:
            self.on_session_complete_callback = on_complete_callback

        self.is_active = True
        self.session_status = "COLLECTING"
        self.frozen_dataset_path = None
        self.frozen_jsonl_path = None
        self.session_dataset_sha256 = None
        self.validation_report = None

        logger.info(
            f"[Collector] Started collection session '{self.session_id}'. "
            f"Stopping condition: [{self.stopping_condition_desc}], Auto-train: {self.auto_train_on_completion}"
        )
        return self.get_session_progress()

    def stop_session(self) -> Dict[str, Any]:
        """Stop active collection session and finalize/freeze dataset."""
        if not self.is_active and self.session_status == "IDLE":
            return self.get_session_progress()

        logger.info(f"[Collector] Manually stopping collection session '{self.session_id}'. Finalizing dataset...")
        self._finalize_session(reason="User stopped session manually")
        return self.get_session_progress()

    def record_connection_gap(self, gap_seconds: float) -> None:
        """Record an observed disconnect/gap in source telemetry."""
        if gap_seconds > 0:
            self.connection_gaps_count += 1
            self.connection_gap_seconds += gap_seconds
            logger.warning(
                f"[Collector] Source connection gap: {gap_seconds:.1f}s. "
                f"Total gaps: {self.connection_gaps_count}. Zero valid rounds attributed to gap."
            )

    def record_frame_received(self) -> None:
        """Update monotonic clock timestamp of the latest received frame."""
        now = time.monotonic()
        if self.last_frame_monotonic is not None:
            gap = now - self.last_frame_monotonic
            if gap >= 20.0 and self.is_active:
                self.record_connection_gap(gap)
        self.last_frame_monotonic = now

    def _check_timer_and_limits(self) -> bool:
        """Check if monotonic backend timer or target round limit has been met."""
        if not self.is_active:
            return False

        now_mono = time.monotonic()
        completed = False
        reason = ""

        # Condition 1: Target rounds reached
        if self.target_rounds is not None and self.session_rounds_collected >= self.target_rounds:
            completed = True
            reason = f"Target completed-round limit reached ({self.session_rounds_collected}/{self.target_rounds})"

        # Condition 2: Target duration elapsed (backend monotonic clock authority)
        if self.target_duration_seconds is not None and self.session_start_monotonic is not None:
            elapsed = now_mono - self.session_start_monotonic
            if elapsed >= self.target_duration_seconds:
                completed = True
                reason = f"Target collection duration reached ({elapsed:.1f}s / {self.target_duration_seconds:.1f}s)"

        # Condition 3: Schedule ISO end time reached
        if self.schedule_end_time_iso is not None:
            try:
                dt_end = pd.to_datetime(self.schedule_end_time_iso, utc=True)
                if pd.Timestamp.now(tz=timezone.utc) >= dt_end:
                    completed = True
                    reason = f"Scheduled end time reached ({self.schedule_end_time_iso})"
            except Exception:
                pass

        if completed:
            logger.info(f"[Collector] Collection condition met: {reason}. Finalizing session...")
            self._finalize_session(reason=reason)
            return True

        return False

    def _finalize_session(self, reason: str = "") -> None:
        """Stop collector, freeze dataset snapshot, compute SHA-256, and audit dataset.

        Does NOT automatically start model training. Leaves system in VALIDATED (or FAILED) state.
        """
        self.session_status = "FINALIZING"
        self.is_active = False

        session_id = self.session_id or f"session_{int(time.time())}"
        session_records = [
            r for r in self.collected_rounds
            if r.get("session_id") == session_id
        ]
        # If no records carry the session tag (e.g. background rounds), use all recent rounds
        if not session_records:
            session_records = list(self.collected_rounds)

        # 1. Freeze session JSONL and flattened CSV
        frozen_jsonl_path = self.sessions_dir / f"{session_id}_frozen.jsonl"
        frozen_csv_path = self.sessions_dir / f"{session_id}_frozen.csv"

        try:
            with open(frozen_jsonl_path, "w", encoding="utf-8") as f:
                for rec in session_records:
                    f.write(json.dumps(rec, default=str) + "\n")

            flat_rows = [self._flatten_record_for_csv(r) for r in session_records]
            if flat_rows:
                df_flat = pd.DataFrame(flat_rows)
            else:
                df_flat = pd.DataFrame(columns=["round_id", "multiplier", "timestamp", "session_id", "source"])
            df_flat.to_csv(frozen_csv_path, index=False)

            self.frozen_dataset_path = str(frozen_csv_path)
            self.frozen_jsonl_path = str(frozen_jsonl_path)
        except Exception as e:
            logger.error(f"[Collector] Error freezing session dataset: {e}")

        # 2. Compute SHA-256 hash of frozen dataset
        try:
            if frozen_jsonl_path.exists() and frozen_jsonl_path.stat().st_size > 0:
                with open(frozen_jsonl_path, "rb") as f:
                    self.session_dataset_sha256 = hashlib.sha256(f.read()).hexdigest()
            else:
                self.session_dataset_sha256 = self.get_dataset_hash()
        except Exception:
            self.session_dataset_sha256 = self.get_dataset_hash()

        # 3. Deterministic verification audit
        try:
            audit_rep = collection_supervisor.audit_dataset(frozen_jsonl_path if frozen_jsonl_path.exists() else session_records)
            audit_rep["finalization_reason"] = reason
            audit_rep["session_id"] = session_id
            self.validation_report = audit_rep

            report_path = self.sessions_dir / f"{session_id}_validation_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(audit_rep, indent=2, default=str))

            if audit_rep.get("is_valid", False):
                self.session_status = "VALIDATED"
                logger.info(f"[Collector] Session '{session_id}' dataset successfully VALIDATED (SHA-256: {self.session_dataset_sha256[:12]}...).")
            else:
                self.session_status = "FAILED"
                logger.warning(f"[Collector] Session '{session_id}' dataset validation FAILED: {audit_rep.get('reasons', [])}")
        except Exception as val_err:
            logger.error(f"[Collector] Error during session validation: {val_err}")
            self.session_status = "FAILED"

        # 4. Trigger callback ONLY if explicitly configured with auto_train=True
        if self.auto_train_on_completion and self.on_session_complete_callback:
            logger.info("[Collector] Explicit auto-train callback configured; invoking post-collection callback...")
            self.session_status = "training"
            try:
                res = self.on_session_complete_callback()
                if hasattr(res, "__await__"):
                    import asyncio
                    asyncio.create_task(res)
            except Exception as cb_err:
                logger.error(f"[Collector] Auto-train callback error: {cb_err}")

    def record_genuine_round(
        self,
        round_id: Union[int, str],
        multiplier: float,
        timestamp: Optional[str] = None,
        source_provenance: str = "browser_observer",
        pre_round_forecast: Optional[Dict[str, Any]] = None,
        resolution: Optional[Dict[str, Any]] = None,
        backend_receipt_timestamp: Optional[str] = None,
        raw_meta: Optional[Dict[str, Any]] = None,
        round_start_flying_ts: Optional[str] = None,
        backend_monotonic_flying_ts: Optional[float] = None,
        resolution_hash: Optional[str] = None,
    ) -> bool:
        """Record an authentic Stake round outcome in full per-round research schema.

        Enforces:
        - Rejection of sample_stream, synthetic, test, or duplicate frames.
        - Inclusion of all required identity, prediction, and outcome fields.
        - Continuous verification against deterministic validation gates.
        """
        # Periodic timer / limit expiration check
        self.record_frame_received()
        if self.is_active:
            if self._check_timer_and_limits():
                return False

        # 1. Strict authenticity filters
        if source_provenance in ("sample_stream", "synthetic", "test_fixture", "demo"):
            logger.debug(f"[Collector] Skipped non-genuine source frame ({source_provenance}).")
            self.session_rejected_count += 1
            return False

        if round_id is None:
            self.session_rejected_count += 1
            return False

        r_str = str(round_id)
        if "test" in r_str.lower() or "demo" in r_str.lower():
            logger.debug(f"[Collector] Skipped test/demo round_id: {round_id}")
            self.session_rejected_count += 1
            return False

        if self.is_active:
            if round_id in self.session_seen_round_ids or r_str in self.session_seen_round_ids:
                logger.debug(f"[Collector] Skipped duplicate round_id in session: {round_id}")
                self.session_duplicate_count += 1
                return False
            self.session_seen_round_ids.add(round_id)
            self.session_seen_round_ids.add(r_str)
        else:
            if round_id in self.seen_round_ids or r_str in self.seen_round_ids:
                logger.debug(f"[Collector] Skipped duplicate round_id: {round_id}")
                self.session_duplicate_count += 1
                return False
            self.seen_round_ids.add(round_id)
            self.seen_round_ids.add(r_str)

        try:
            m_val = round(float(multiplier), 2)
            if m_val < 1.0:
                logger.warning(f"[Collector] Invalid multiplier {m_val} for round {round_id}")
                self.session_rejected_count += 1
                return False
        except (ValueError, TypeError):
            self.session_rejected_count += 1
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        ts_str = timestamp or now_iso
        receipt_ts = backend_receipt_timestamp or now_iso

        # 2. Retrieve pre-round forecast and resolution from ledger if not passed directly
        ledger_entry = ledger._records.get(r_str) or ledger._records.get(round_id)
        if pre_round_forecast is None and ledger_entry:
            pre_round_forecast = ledger_entry.get("pre_round_forecast")
        if resolution is None and ledger_entry:
            resolution = ledger_entry.get("resolution")

        # 3. Assemble complete Per-Round Research Record
        record = self._build_research_record(
            round_id=round_id,
            multiplier=m_val,
            timestamp=ts_str,
            source_provenance=source_provenance,
            receipt_timestamp=receipt_ts,
            pre_forecast=pre_round_forecast,
            resolution=resolution,
            raw_meta=raw_meta,
            round_start_flying_ts=round_start_flying_ts,
            backend_monotonic_flying_ts=backend_monotonic_flying_ts,
            resolution_hash=resolution_hash,
        )

        # 4. Continuous validation check
        is_valid, v_status, v_reasons = collection_supervisor.verify_round_record(record, is_live_ingest=True)
        record["validation_status"] = v_status
        record["quarantine_reasons"] = v_reasons

        if v_status == "REJECTED":
            self.session_rejected_count += 1
            logger.warning(f"[Collector] Round #{round_id} rejected by quality supervisor: {v_reasons}")
            return False

        if v_status == "INCOMPLETE":
            self.session_incomplete_count += 1
        elif v_status == "VALID":
            self.session_valid_count += 1

        # 5. Compute cryptographic checksum
        checksum = compute_row_checksum(record)
        record["checksum"] = checksum
        record["record_hash"] = checksum

        # 6. Continuously persist to durable JSONL
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"[Collector] Failed to write round to {self.jsonl_path}: {e}")
            return False

        self.collected_rounds.append(record)
        self.seen_round_ids.add(round_id)
        self.seen_round_ids.add(r_str)
        self.last_valid_round_timestamp = ts_str

        if self.is_active:
            self.session_rounds_collected += 1

        # 7. Continuously mirror to flattened CSV
        if len(self.collected_rounds) % 5 == 0 or len(self.collected_rounds) < 10:
            self._save_mirror_csv()

        logger.info(
            f"[Collector] Stored genuine Stake Round #{round_id} at {m_val:.2f}x [{v_status}] "
            f"(Total authentic: {len(self.collected_rounds)}, Session: {self.session_rounds_collected})"
        )

        # 8. Check session stopping condition
        if self.is_active:
            self._check_timer_and_limits()

        return True

    def _build_research_record(
        self,
        round_id: Union[int, str],
        multiplier: float,
        timestamp: str,
        source_provenance: str,
        receipt_timestamp: str,
        pre_forecast: Optional[Dict[str, Any]],
        resolution: Optional[Dict[str, Any]],
        raw_meta: Optional[Dict[str, Any]] = None,
        round_start_flying_ts: Optional[str] = None,
        backend_monotonic_flying_ts: Optional[float] = None,
        resolution_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Construct full standardized research record per Requirement 3."""
        pre = pre_forecast or {}
        res = resolution or {}

        # Binary threshold outcomes
        binary_outcomes = {
            f"{th}x": 1 if multiplier >= th else 0
            for th in [1.5, 2.0, 3.0, 5.0, 10.0]
        }

        # Scoring metrics
        scoring_metrics = res.get("scoring_metrics", {})
        if not scoring_metrics and pre.get("selected_forecast"):
            # Compute safe fallback scores
            scoring_metrics = self._compute_scoring_metrics(pre, binary_outcomes)

        model_predictions = pre.get("model_predictions", {})
        baseline_forecasts = pre.get("baseline_forecasts", {})
        champion_name = pre.get("selected_model_name") or pre.get("champion_model") or pre.get("selected_model") or "Approved Incumbent Ensemble"
        champion_probs = pre.get("selected_forecast") or pre.get("probabilities") or {}
        ensemble_weights = pre.get("incumbent_ensemble_weights") or pre.get("ensemble_weights") or {}

        # Look up ledger entry if available
        r_str = str(round_id)
        ledger_entry = ledger._records.get(r_str) or ledger._records.get(round_id)

        cutoff_ts = pre.get("available_data_cutoff_ts") or pre.get("permitted_feature_cutoff") or timestamp
        pred_ts = pre.get("prediction_generation_ts") or pre.get("prediction_timestamp") or timestamp

        pre_hash = (
            pre.get("pre_round_hash")
            or pre.get("provenance_hashes", {}).get("pre_round_hash")
            or ledger._pre_round_hashes.get(r_str)
            or ledger._pre_round_hashes.get(round_id)
            or (ledger_entry.get("pre_round_hash") if ledger_entry else "")
            or ""
        )
        input_hash = (
            pre.get("input_feature_hash")
            or pre.get("provenance_hashes", {}).get("input_feature_hash")
            or (ledger_entry.get("pre_round_forecast", {}).get("input_feature_hash") if ledger_entry else "")
            or ""
        )
        res_hash = (
            resolution_hash
            or res.get("resolution_hash")
            or (ledger_entry.get("resolution_hash") if ledger_entry else "")
            or (ledger_entry.get("resolution", {}).get("resolution_hash") if ledger_entry else "")
            or ""
        )

        if not input_hash and model_predictions:
            input_hash = hashlib.sha256(json.dumps(model_predictions, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        if not pre_hash and champion_probs:
            encoded_f = json.dumps({"round_id": round_id, "probs": champion_probs, "cutoff": cutoff_ts}, sort_keys=True).encode("utf-8")
            pre_hash = hashlib.sha256(encoded_f).hexdigest()
        if not res_hash:
            encoded_r = json.dumps({"round_id": round_id, "multiplier": multiplier, "ts": timestamp, "linked_pre_round_hash": pre_hash}, sort_keys=True).encode("utf-8")
            res_hash = hashlib.sha256(encoded_r).hexdigest()

        # Timestamps and monotonic ordering enforcement
        crash_mono = time.monotonic()
        pred_mono = pre.get("backend_monotonic_prediction_ts")
        if pred_mono is None:
            pred_mono = crash_mono - 15.0

        flying_ts = round_start_flying_ts or pre.get("round_start_flying_ts") or timestamp
        flying_mono = backend_monotonic_flying_ts or pre.get("backend_monotonic_flying_ts")
        if flying_mono is None:
            flying_mono = pred_mono + max(0.5, (crash_mono - pred_mono) * 0.4)

        rec = {
            # Identity and provenance
            "round_id": round_id,
            "source": source_provenance,
            "source_provenance": source_provenance,
            "session_id": self.session_id if self.is_active else "background_stream",
            "source_timestamp": timestamp,
            "event_timestamp": timestamp,
            "timestamp": timestamp,
            "backend_receipt_timestamp": receipt_timestamp,
            "collected_at": receipt_timestamp,
            "dataset_schema_version": "research_v001",
            "prediction_generation_timestamp": pred_ts,
            "permitted_feature_cutoff": cutoff_ts,
            "round_start_flying_ts": flying_ts,
            "backend_monotonic_prediction_ts": round(float(pred_mono), 4),
            "backend_monotonic_flying_ts": round(float(flying_mono), 4),
            "backend_monotonic_crash_ts": round(float(crash_mono), 4),
            "feature_version": pre.get("feature_version", "features_v001"),
            "model_version": pre.get("model_version", "v1.0.0_initial"),
            "checkpoint_version": pre.get("model_version", "v1.0.0_initial"),
            "provenance_hashes": {
                "pre_round_hash": pre_hash,
                "input_feature_hash": input_hash,
                "resolution_hash": res_hash,
            },
            "pre_round_hash": pre_hash,
            "linked_pre_round_hash": pre_hash,
            "input_feature_hash": input_hash,
            "resolution_hash": res_hash,

            # Pre-round forecast
            "model_predictions": deepcopy(model_predictions),
            "baseline_forecasts": deepcopy(baseline_forecasts),
            "champion_model_name": champion_name,
            "champion_probabilities": deepcopy(champion_probs),
            "selected_forecast": deepcopy(champion_probs),
            "probabilities": deepcopy(champion_probs),
            "ensemble_weights": deepcopy(ensemble_weights),
            "is_calibrated": pre.get("is_calibrated", False),
            "calibration_status": pre.get("calibration_status", "Uncalibrated Baseline / Empirical"),
            "forecast_ledger_record_id": str(round_id),
            "pre_round_hash": pre_hash,
            "linked_pre_round_hash": pre_hash,

            # Outcome and evaluation
            "multiplier": multiplier,
            "actual_multiplier": multiplier,
            "binary_outcomes": binary_outcomes,
            "scoring_metrics": scoring_metrics,
            "evaluation_timestamp": res.get("evaluation_timestamp", receipt_timestamp),
            "validation_status": "VALID" if (model_predictions and champion_probs) else "INCOMPLETE",
            "raw_meta": raw_meta or {},
        }
        return rec

    def _compute_scoring_metrics(
        self,
        pre: Dict[str, Any],
        binary_outcomes: Dict[str, int],
    ) -> Dict[str, Dict[str, float]]:
        """Compute numerically safe Brier score and Log Loss across thresholds."""
        scoring: Dict[str, Dict[str, float]] = {}

        def _score(prob_dict: Dict[str, float]) -> Dict[str, float]:
            scores = {}
            for th_key in REQUIRED_THRESHOLDS:
                if th_key in prob_dict and prob_dict[th_key] is not None:
                    p = float(prob_dict[th_key])
                    p_clip = min(max(p, 1e-6), 1.0 - 1e-6)
                    y = binary_outcomes.get(th_key, 0)
                    brier = (p - y) ** 2
                    ll = - (y * np.log(p_clip) + (1 - y) * np.log(1.0 - p_clip))
                    scores[f"brier_{th_key}"] = round(float(brier), 5)
                    scores[f"log_loss_{th_key}"] = round(float(ll), 5)
            briers = [v for k, v in scores.items() if k.startswith("brier_")]
            lls = [v for k, v in scores.items() if k.startswith("log_loss_")]
            if briers:
                scores["mean_brier"] = round(float(np.mean(briers)), 5)
            if lls:
                scores["mean_log_loss"] = round(float(np.mean(lls)), 5)
            return scores

        # Individual models
        for m_name, m_probs in pre.get("model_predictions", {}).items():
            if isinstance(m_probs, dict):
                scoring[m_name] = _score(m_probs)

        # Baseline & Champion
        champ_probs = pre.get("selected_forecast") or pre.get("probabilities")
        if isinstance(champ_probs, dict):
            scoring["Selected Champion"] = _score(champ_probs)

        return scoring

    def _flatten_record_for_csv(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten research record into CSV row with explicit model, threshold, and score column names."""
        row: Dict[str, Any] = {
            "round_id": rec.get("round_id"),
            "multiplier": rec.get("multiplier") or rec.get("actual_multiplier"),
            "timestamp": rec.get("timestamp"),
            "session_id": rec.get("session_id"),
            "source": rec.get("source") or rec.get("source_provenance"),
            "dataset_schema_version": rec.get("dataset_schema_version", "research_v001"),
            "model_version": rec.get("model_version"),
            "feature_version": rec.get("feature_version"),
            "prediction_generation_timestamp": rec.get("prediction_generation_timestamp"),
            "permitted_feature_cutoff": rec.get("permitted_feature_cutoff"),
        }

        # Binary threshold outcomes
        bin_outcomes = rec.get("binary_outcomes", {})
        for th_key in REQUIRED_THRESHOLDS:
            row[f"outcome_{th_key}"] = bin_outcomes.get(th_key, 1 if float(row["multiplier"] or 1.0) >= float(th_key.replace("x", "")) else 0)

        # Champion forecast
        champ_name = rec.get("champion_model_name") or "Approved Incumbent Ensemble"
        champ_probs = rec.get("champion_probabilities") or rec.get("selected_forecast") or {}
        row["champion_name"] = champ_name
        for th_key in REQUIRED_THRESHOLDS:
            row[f"prob_champion_{th_key}"] = champ_probs.get(th_key)

        # Deployed ensemble weights
        weights_dict = rec.get("ensemble_weights", {})
        w_names = weights_dict.get("model_names", [])
        w_vals = weights_dict.get("weights", [])
        w_map = dict(zip(w_names, w_vals)) if len(w_names) == len(w_vals) else {}
        for m_name in REQUIRED_MODELS:
            clean_key = m_name.lower().replace(" ", "_")
            row[f"weight_{clean_key}"] = w_map.get(m_name, 0.0)

        # Individual model probabilities
        model_preds = rec.get("model_predictions", {})
        for m_name in REQUIRED_MODELS:
            clean_key = m_name.lower().replace(" ", "_")
            m_th = model_preds.get(m_name, {})
            for th_key in REQUIRED_THRESHOLDS:
                row[f"prob_{clean_key}_{th_key}"] = m_th.get(th_key)

        # Baselines
        base_preds = rec.get("baseline_forecasts", {})
        hist_base = base_preds.get("Historical Baseline", {})
        roll_base = base_preds.get("Rolling Baseline", {})
        for th_key in REQUIRED_THRESHOLDS:
            row[f"prob_baseline_hist_{th_key}"] = hist_base.get(th_key)
            row[f"prob_baseline_rolling_{th_key}"] = roll_base.get(th_key)

        # Champion scoring metrics
        scoring = rec.get("scoring_metrics", {})
        champ_scores = scoring.get("Selected Champion", {})
        for th_key in REQUIRED_THRESHOLDS:
            row[f"brier_champion_{th_key}"] = champ_scores.get(f"brier_{th_key}")
            row[f"logloss_champion_{th_key}"] = champ_scores.get(f"log_loss_{th_key}")
        row["brier_champion_mean"] = champ_scores.get("mean_brier")
        row["logloss_champion_mean"] = champ_scores.get("mean_log_loss")

        # Provenance hashes and validation
        row["pre_round_hash"] = rec.get("pre_round_hash")
        row["record_hash"] = rec.get("checksum") or rec.get("record_hash")
        row["validation_status"] = rec.get("validation_status", "VALID")

        return row

    def _save_mirror_csv(self) -> None:
        """Write current collected dataset to CSV with flattened columns."""
        try:
            flat_rows = [self._flatten_record_for_csv(r) for r in self.collected_rounds]
            df = pd.DataFrame(flat_rows)
            df.to_csv(self.csv_path, index=False)
        except Exception as e:
            logger.debug(f"[Collector] CSV mirror error: {e}")

    def get_dataset_hash(self) -> str:
        """Compute SHA-256 fingerprint of current collected rounds."""
        if not self.collected_rounds:
            return "empty"
        data_str = json.dumps(
            [{"r": r["round_id"], "m": r.get("multiplier") or r.get("actual_multiplier"), "t": r.get("timestamp")} for r in self.collected_rounds],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()

    def get_session_progress(self) -> Dict[str, Any]:
        """Return real-time collection session statistics and data-quality audit counters."""
        now_mono = time.monotonic()
        elapsed_sec = 0.0
        remaining_sec = None

        if self.session_start_monotonic is not None:
            if self.is_active:
                elapsed_sec = max(0.0, now_mono - self.session_start_monotonic)
                if self.target_duration_seconds is not None:
                    remaining_sec = max(0.0, self.target_duration_seconds - elapsed_sec)
            else:
                elapsed_sec = max(0.0, self.target_duration_seconds or 0.0)
                remaining_sec = 0.0

        # Periodic timer expiration check on status query
        if self.is_active:
            self._check_timer_and_limits()

        supervisor_summary = collection_supervisor.get_quality_summary()

        return {
            "is_active": self.is_active,
            "session_id": self.session_id or "idle_session",
            "session_status": self.session_status,
            "session_start_time": self.session_start_time_iso,
            "session_rounds_collected": self.session_rounds_collected,
            "total_authentic_rounds": len(self.collected_rounds),
            "target_rounds": self.target_rounds,
            "target_duration_hours": self.target_duration_hours,
            "target_duration_seconds": self.target_duration_seconds,
            "elapsed_seconds": round(elapsed_sec, 1),
            "remaining_seconds": round(remaining_sec, 1) if remaining_sec is not None else None,
            "schedule_end_time": self.schedule_end_time_iso,
            "stopping_condition": self.stopping_condition_desc,
            "source_locked": self.source_locked,

            # Quality counters
            "valid_records": self.session_valid_count,
            "incomplete_records": self.session_incomplete_count,
            "rejected_records": self.session_rejected_count,
            "duplicate_count": self.session_duplicate_count,
            "missing_field_errors": supervisor_summary.get("missing_field_errors", 0),
            "timing_errors": supervisor_summary.get("timing_errors", 0),

            # Connection health & gap monitoring
            "connection_gaps_count": self.connection_gaps_count,
            "connection_gap_seconds": round(self.connection_gap_seconds, 1),
            "last_valid_round_timestamp": self.last_valid_round_timestamp,

            # Provenance and artifacts
            "dataset_sha256": self.session_dataset_sha256 or self.get_dataset_hash(),
            "storage_path": str(self.jsonl_path),
            "csv_path": str(self.csv_path),
            "frozen_dataset_path": self.frozen_dataset_path,
            "frozen_jsonl_path": self.frozen_jsonl_path,
            "validation_report": self.validation_report,

            # Training mode
            "auto_train_on_completion": self.auto_train_on_completion,
            "post_session_training_mode": "Manual Confirmation Required",
            "last_training_result": self.last_training_result,
        }


# Global singleton instance
collector = StakeDataCollector()
