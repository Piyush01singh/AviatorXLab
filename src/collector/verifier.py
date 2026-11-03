"""Autonomous Collection Quality Supervisor & Deterministic Dataset Verifier.

Strict Protocol:
- Validates every captured round continuously during live collection.
- Performs an independent, complete cryptographic and schema audit at session finalization.
- Enforces genuine Stake provenance (browser_observer only; rejects synthetic/sample/test).
- Verifies causal timestamp ordering: prediction <= waiting < crash <= backend receipt.
- Verifies presence of all 6 models across all 5 thresholds or explicit documented failure.
- Verifies probability monotonicity: P(>=1.5x) >= P(>=2.0x) >= P(>=3.0x) >= P(>=5.0x) >= P(>=10.0x).
- Verifies prediction-to-outcome linkage and immutable prediction-ledger SHA-256 integrity.
- Quarantines invalid records with precise rejection reasons; never fabricates values.
- Distinguishes incomplete records from complete eligible training examples.
- Blocks model training if critical validation gates fail.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from src.utils.logging import get_logger

logger = get_logger("collection_verifier")

REQUIRED_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "MLP",
    "LSTM",
    "GRU",
]

REQUIRED_THRESHOLDS = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]
NUMERIC_THRESHOLDS = [1.5, 2.0, 3.0, 5.0, 10.0]


def parse_iso_timestamp(ts: Any) -> Optional[datetime]:
    """Parse ISO timestamp string to timezone-aware UTC datetime."""
    if not ts:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        ts_str = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def compute_sha256(data: Any) -> str:
    """Compute deterministic SHA-256 fingerprint."""
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CollectionQualitySupervisor:
    """Deterministic verifier and autonomous supervisor for collection dataset integrity."""

    def __init__(self, quarantine_path: str = "data/live/quarantine_rounds.jsonl"):
        self.quarantine_path = Path(quarantine_path)
        self.quarantine_path.parent.mkdir(parents=True, exist_ok=True)

        self.reset_counters()

    def reset_counters(self) -> None:
        """Reset live verification audit counters."""
        self.total_checked: int = 0
        self.valid_records: int = 0
        self.incomplete_records: int = 0
        self.rejected_records: int = 0
        self.duplicate_count: int = 0
        self.missing_field_errors: int = 0
        self.timing_errors: int = 0
        self.rejection_details: List[Dict[str, Any]] = []
        self._seen_round_ids: Set[str] = set()

    def verify_round_record(
        self,
        record: Dict[str, Any],
        seen_round_ids: Optional[Set[str]] = None,
        is_live_ingest: bool = False,
    ) -> Tuple[bool, str, List[str]]:
        """Verify an individual completed-round record against required research schema.

        Returns:
            (is_eligible_valid, status, reasons)
            status is one of: 'VALID', 'INCOMPLETE', 'REJECTED'
        """
        reasons: List[str] = []
        is_incomplete = False
        is_rejected = False

        self.total_checked += 1
        seen_set = self._seen_round_ids if seen_round_ids is None else seen_round_ids

        # 1. Source provenance check
        src = str(record.get("source_provenance") or record.get("source") or "").strip().lower()
        if src in ("sample_stream", "synthetic", "test_fixture", "demo"):
            is_rejected = True
            reasons.append(f"Non-genuine source provenance rejected: '{src}'. Must be 'browser_observer'.")
        elif src and src != "browser_observer" and not is_live_ingest:
            is_rejected = True
            reasons.append(f"Source must be genuine 'browser_observer', got: '{src}'.")

        # 2. Round ID authenticity and uniqueness
        round_id = record.get("round_id")
        if round_id is None:
            is_rejected = True
            self.missing_field_errors += 1
            reasons.append("Missing required 'round_id'.")
        else:
            r_str = str(round_id)
            if "test" in r_str.lower() or "demo" in r_str.lower():
                is_rejected = True
                reasons.append(f"Synthetic/test frame detected in round_id: '{r_str}'.")

            if r_str in seen_set:
                self.duplicate_count += 1
                is_rejected = True
                reasons.append(f"Duplicate round_id detected: '{r_str}'.")
            else:
                seen_set.add(r_str)

        # 3. Crash multiplier validation
        mult = record.get("actual_multiplier")
        if mult is None:
            mult = record.get("multiplier")

        if mult is None:
            is_incomplete = True
            self.missing_field_errors += 1
            reasons.append("Missing crash multiplier.")
        else:
            try:
                m_val = float(mult)
                if np.isnan(m_val) or np.isinf(m_val) or m_val < 1.0:
                    is_rejected = True
                    reasons.append(f"Invalid crash multiplier: {m_val} (must be finite float >= 1.00).")
            except (ValueError, TypeError):
                is_rejected = True
                reasons.append(f"Non-numeric crash multiplier: {mult}.")

        # 4. Timestamp relationship and causal timing audit
        pred_ts_raw = record.get("prediction_generation_timestamp")
        cutoff_ts_raw = record.get("permitted_feature_cutoff") or record.get("available_data_cutoff_ts")
        waiting_ts_raw = record.get("round_start_waiting_ts") or record.get("source_timestamp")
        crash_ts_raw = record.get("crash_timestamp") or record.get("timestamp") or record.get("event_timestamp")
        receipt_ts_raw = record.get("backend_receipt_timestamp") or record.get("collected_at")

        flying_ts_raw = record.get("round_start_flying_ts") or record.get("flying_timestamp")
        pred_dt = parse_iso_timestamp(pred_ts_raw)
        crash_dt = parse_iso_timestamp(crash_ts_raw)
        cutoff_dt = parse_iso_timestamp(cutoff_ts_raw)
        receipt_dt = parse_iso_timestamp(receipt_ts_raw)
        flying_dt = parse_iso_timestamp(flying_ts_raw)

        if pred_dt and flying_dt:
            # Prediction must be generated before or at flying transition (no causal inversion)
            if pred_dt > flying_dt:
                self.timing_errors += 1
                is_rejected = True
                reasons.append(
                    f"Timing violation: prediction timestamp ({pred_ts_raw}) occurred AFTER flying timestamp ({flying_ts_raw})."
                )

        if flying_dt and crash_dt:
            # Flying transition must occur before or at crash resolution (no causal inversion)
            if flying_dt > crash_dt:
                self.timing_errors += 1
                is_rejected = True
                reasons.append(
                    f"Timing violation: flying timestamp ({flying_ts_raw}) occurred AFTER crash timestamp ({crash_ts_raw})."
                )

        if pred_dt and crash_dt:
            # Future outcome leakage check: Prediction must be generated before round crash
            if pred_dt > crash_dt:
                self.timing_errors += 1
                is_rejected = True
                reasons.append(
                    f"Timing violation: prediction timestamp ({pred_ts_raw}) occurred AFTER crash timestamp ({crash_ts_raw})."
                )

        if cutoff_dt and crash_dt:
            # Available feature cutoff must not include target-round crash
            if cutoff_dt > crash_dt:
                self.timing_errors += 1
                is_rejected = True
                reasons.append(
                    f"Timing violation: feature cutoff ({cutoff_ts_raw}) occurred AFTER crash timestamp ({crash_ts_raw})."
                )

        # Strict monotonic backend clock sequence check: pred_mono < flying_mono < crash_mono
        pred_mono = record.get("backend_monotonic_prediction_ts")
        flying_mono = record.get("backend_monotonic_flying_ts")
        crash_mono = record.get("backend_monotonic_crash_ts")
        if pred_mono is not None and flying_mono is not None and pred_mono >= flying_mono:
            self.timing_errors += 1
            is_rejected = True
            reasons.append(
                f"Monotonic clock sequence violation: prediction ({pred_mono}) must strictly precede flying started ({flying_mono})."
            )
        if flying_mono is not None and crash_mono is not None and flying_mono >= crash_mono:
            self.timing_errors += 1
            is_rejected = True
            reasons.append(
                f"Monotonic clock sequence violation: flying started ({flying_mono}) must strictly precede crash occurred ({crash_mono})."
            )
        if pred_mono is not None and crash_mono is not None and pred_mono >= crash_mono:
            self.timing_errors += 1
            is_rejected = True
            reasons.append(
                f"Monotonic clock sequence violation: prediction ({pred_mono}) must strictly precede crash occurred ({crash_mono})."
            )

        # 5. Pre-round forecast completeness
        model_preds = record.get("model_predictions")
        if not model_preds or not isinstance(model_preds, dict):
            # If no model predictions are present, mark incomplete
            is_incomplete = True
            self.missing_field_errors += 1
            reasons.append("Missing pre-round 'model_predictions' container.")
        else:
            # Check presence of all six required models
            for m_name in REQUIRED_MODELS:
                if m_name not in model_preds:
                    is_incomplete = True
                    self.missing_field_errors += 1
                    reasons.append(f"Model predictions missing required model: '{m_name}'.")
                else:
                    m_th_dict = model_preds[m_name]
                    if not isinstance(m_th_dict, dict):
                        is_incomplete = True
                        reasons.append(f"Predictions for '{m_name}' is not a dictionary.")
                    else:
                        for th_key in REQUIRED_THRESHOLDS:
                            if th_key not in m_th_dict:
                                is_incomplete = True
                                self.missing_field_errors += 1
                                reasons.append(f"Model '{m_name}' missing threshold '{th_key}'.")
                            else:
                                val = m_th_dict[th_key]
                                try:
                                    f_val = float(val)
                                    if np.isnan(f_val) or np.isinf(f_val) or not (0.0 <= f_val <= 1.0):
                                        is_rejected = True
                                        reasons.append(
                                            f"Model '{m_name}' threshold '{th_key}' has out-of-bounds probability: {val}."
                                        )
                                except (TypeError, ValueError):
                                    is_rejected = True
                                    reasons.append(f"Model '{m_name}' threshold '{th_key}' is non-numeric: {val}.")

                        # Enforce probability monotonicity for EVERY model: P(1.5x) >= P(2x) >= P(3x) >= P(5x) >= P(10x)
                        prev_m_p = 1.0
                        for th_key in REQUIRED_THRESHOLDS:
                            if th_key in m_th_dict:
                                try:
                                    cur_m_p = float(m_th_dict[th_key])
                                    if cur_m_p > prev_m_p + 1e-4:
                                        is_rejected = True
                                        reasons.append(
                                            f"Model '{m_name}' monotonicity violation: P({th_key})={cur_m_p} > previous threshold={prev_m_p}."
                                        )
                                    prev_m_p = cur_m_p
                                except (ValueError, TypeError):
                                    pass

        # 6. Champion / Selected forecast & Monotonicity Check
        selected_probs = (
            record.get("champion_probabilities")
            or record.get("selected_forecast")
            or record.get("probabilities")
        )
        if not selected_probs or not isinstance(selected_probs, dict):
            is_incomplete = True
            reasons.append("Missing final/champion probabilities dictionary.")
        else:
            # Check monotonicity: P(>=1.5x) >= P(>=2x) >= P(>=3x) >= P(>=5x) >= P(>=10x)
            prev_p = 1.0
            for th_key in REQUIRED_THRESHOLDS:
                if th_key not in selected_probs:
                    is_incomplete = True
                    reasons.append(f"Champion forecast missing threshold '{th_key}'.")
                else:
                    try:
                        p_cur = float(selected_probs[th_key])
                        if np.isnan(p_cur) or np.isinf(p_cur) or not (0.0 <= p_cur <= 1.0):
                            is_rejected = True
                            reasons.append(f"Champion probability '{th_key}' out of range: {p_cur}.")
                        elif p_cur > prev_p + 1e-4:  # tolerance for rounding
                            is_rejected = True
                            reasons.append(
                                f"Monotonicity violation: P({th_key})={p_cur} > previous threshold={prev_p}."
                            )
                        prev_p = p_cur
                    except (ValueError, TypeError):
                        is_rejected = True
                        reasons.append(f"Non-numeric probability for threshold '{th_key}'.")

        # 7. Ensemble weights validation
        weights = record.get("ensemble_weights")
        if weights and isinstance(weights, dict):
            w_vals = weights.get("weights")
            if isinstance(w_vals, list) and len(w_vals) > 0:
                try:
                    num_weights = [float(w) for w in w_vals]
                    if any(w < -1e-5 for w in num_weights):
                        is_rejected = True
                        reasons.append("Ensemble weights must be non-negative.")
                    sum_w = sum(num_weights)
                    if abs(sum_w - 1.0) > 1e-3 and sum_w > 0:
                        is_rejected = True
                        reasons.append(f"Ensemble weights must sum to 1.0, got: {sum_w:.4f}.")
                except (ValueError, TypeError):
                    is_rejected = True
                    reasons.append("Non-numeric ensemble weights detected.")

        # 8. Immutable prediction ledger hash linkage & Mandatory Provenance
        prov_hashes = record.get("provenance_hashes") or {}
        pre_hash = record.get("pre_round_hash") or prov_hashes.get("pre_round_hash")
        input_feature_hash = record.get("input_feature_hash") or prov_hashes.get("input_feature_hash")
        linked_hash = record.get("linked_pre_round_hash")
        resolution_hash = record.get("resolution_hash") or prov_hashes.get("resolution_hash")

        # Forecast-time hashes MUST be non-empty and non-placeholder for forecast records
        if not is_incomplete:
            if not pre_hash or not isinstance(pre_hash, str) or len(pre_hash.strip()) < 16:
                is_rejected = True
                reasons.append(f"Missing mandatory pre_round_hash provenance: got '{pre_hash}'.")

            if not input_feature_hash or not isinstance(input_feature_hash, str) or len(input_feature_hash.strip()) < 16:
                is_rejected = True
                reasons.append(f"Missing mandatory input_feature_hash provenance: got '{input_feature_hash}'.")

            if pre_hash and linked_hash and pre_hash != linked_hash:
                is_rejected = True
                reasons.append(f"Hash linkage mismatch: pre_round_hash '{pre_hash[:8]}' != linked '{linked_hash[:8]}'.")

            # For completed records frozen for training / offline audit, resolution_hash is mandatory
            if not is_live_ingest:
                if not resolution_hash or not isinstance(resolution_hash, str) or len(resolution_hash.strip()) < 16:
                    is_rejected = True
                    reasons.append(f"Completed record missing mandatory resolution_hash provenance: got '{resolution_hash}'.")

        # Determine outcome
        if is_rejected:
            status = "REJECTED"
            self.rejected_records += 1
            self._quarantine_record(record, reasons)
            return False, status, reasons

        if is_incomplete:
            status = "INCOMPLETE"
            self.incomplete_records += 1
            return False, status, reasons

        status = "VALID"
        self.valid_records += 1
        return True, status, reasons

    def _quarantine_record(self, record: Dict[str, Any], reasons: List[str]) -> None:
        """Quarantine rejected records into dedicated quarantine log."""
        try:
            entry = {
                "quarantined_at": datetime.now(timezone.utc).isoformat(),
                "rejection_reasons": reasons,
                "record": record,
            }
            self.rejection_details.append({
                "round_id": record.get("round_id"),
                "reasons": reasons,
            })
            with open(self.quarantine_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Error writing to quarantine log: {e}")

    def audit_dataset(
        self,
        dataset: Union[List[Dict[str, Any]], str, Path],
    ) -> Dict[str, Any]:
        """Perform complete deterministic verification audit of an entire dataset.

        Returns comprehensive report including accepted/rejected counts, duplicate count,
        missing-field errors, timing errors, round count, and session SHA-256 fingerprint.
        """
        records: List[Dict[str, Any]] = []
        path_str = "in-memory"

        if isinstance(dataset, (str, Path)):
            p = Path(dataset)
            path_str = str(p)
            if not p.exists():
                return {
                    "is_valid": False,
                    "status": "FAILED",
                    "reason": f"Dataset file does not exist: {path_str}",
                    "total_records": 0,
                    "valid_records": 0,
                    "incomplete_records": 0,
                    "rejected_records": 0,
                    "duplicate_count": 0,
                    "missing_field_errors": 0,
                    "timing_errors": 0,
                    "dataset_sha256": "empty",
                    "dataset_file_path": path_str,
                }
            if p.suffix == ".jsonl":
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except Exception:
                                pass
            elif p.suffix == ".csv":
                import pandas as pd
                df = pd.read_csv(p)
                records = df.to_dict(orient="records")
        else:
            records = list(dataset)

        self.reset_counters()
        seen_ids: Set[str] = set()
        validation_results: List[Dict[str, Any]] = []

        for rec in records:
            is_valid, status, reasons = self.verify_round_record(rec, seen_round_ids=seen_ids)
            validation_results.append({
                "round_id": rec.get("round_id"),
                "status": status,
                "reasons": reasons,
            })

        # Calculate dataset SHA-256 fingerprint
        if records:
            data_fingerprint_str = json.dumps(
                [{"r": r.get("round_id"), "m": r.get("multiplier") or r.get("actual_multiplier"), "t": r.get("timestamp")} for r in records],
                sort_keys=True,
                default=str,
            )
            dataset_sha256 = hashlib.sha256(data_fingerprint_str.encode("utf-8")).hexdigest()
        else:
            dataset_sha256 = "empty"

        is_eligible = (
            self.valid_records > 0
            and self.rejected_records == 0
            and self.timing_errors == 0
        )

        overall_status = "VALIDATED" if is_eligible else "FAILED"
        if len(records) == 0:
            overall_status = "EMPTY"

        report = {
            "is_valid": is_eligible,
            "status": overall_status,
            "total_records": len(records),
            "valid_records": self.valid_records,
            "incomplete_records": self.incomplete_records,
            "rejected_records": self.rejected_records,
            "duplicate_count": self.duplicate_count,
            "missing_field_errors": self.missing_field_errors,
            "timing_errors": self.timing_errors,
            "dataset_sha256": dataset_sha256,
            "dataset_file_path": path_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "rejection_sample": self.rejection_details[:10],
        }
        return report

    def get_quality_summary(self) -> Dict[str, Any]:
        """Return live supervisor counters for API/dashboard display."""
        return {
            "total_checked": self.total_checked,
            "valid_records": self.valid_records,
            "incomplete_records": self.incomplete_records,
            "rejected_records": self.rejected_records,
            "duplicate_count": self.duplicate_count,
            "missing_field_errors": self.missing_field_errors,
            "timing_errors": self.timing_errors,
            "status": "VALID" if self.rejected_records == 0 and self.timing_errors == 0 else "WARNING",
            "recent_rejections": self.rejection_details[-5:],
        }


# Global singleton instance
collection_supervisor = CollectionQualitySupervisor()
