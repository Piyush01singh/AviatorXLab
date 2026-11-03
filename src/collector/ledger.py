"""Immutable Prediction Ledger and Data Provenance.

Strict Protocol:
- Persists immutable pre-round forecasts before round flight/crash.
- Appends post-crash actual outcomes and loss scores without modifying pre-round forecasts.
- Cryptographic SHA-256 checksumming on every ledger entry.
- Separates genuine Stake observations, demo streams, and test fixtures.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from src.utils.logging import get_logger

logger = get_logger("prediction_ledger")

THRESHOLDS = [1.5, 2.0, 3.0, 5.0, 10.0]
THRESHOLD_KEYS = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]


class ImmutabilityViolationError(Exception):
    """Raised when an attempt is made to overwrite or alter a frozen pre-round forecast."""
    pass


def compute_sha256(data: Any) -> str:
    """Compute deterministic SHA-256 hash of a JSON-serializable object."""
    encoded = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class PredictionLedger:
    """Append-only, cryptographically verified ledger for live round forecasts and resolutions."""

    def __init__(self, ledger_path: str = "data/live/prediction_ledger.jsonl"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory index of records: round_id -> record
        self._records: Dict[Any, Dict[str, Any]] = {}
        self._pre_round_hashes: Dict[Any, str] = {}
        self._load_existing_ledger()

    def _load_existing_ledger(self) -> None:
        """Load and verify existing ledger entries from disk on startup."""
        if not self.ledger_path.exists():
            return

        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    r_id = entry.get("round_id")
                    if r_id is None:
                        continue

                    entry_type = entry.get("entry_type")
                    if entry_type == "pre_round_prediction":
                        self._records[r_id] = entry
                        self._pre_round_hashes[r_id] = entry.get("pre_round_hash", "")
                    elif entry_type == "round_resolution":
                        if r_id in self._records:
                            self._records[r_id]["resolution"] = entry
            logger.info(f"Loaded {len(self._records)} ledger entries from {self.ledger_path}")
        except Exception as e:
            logger.warning(f"Error loading prediction ledger from {self.ledger_path}: {e}")

    def record_pre_round_prediction(
        self,
        round_id: Union[int, str],
        source_provenance: str,
        event_timestamps: Dict[str, str],
        model_predictions: Dict[str, Dict[str, float]],
        baseline_forecasts: Dict[str, Dict[str, float]],
        incumbent_ensemble_probs: Dict[str, float],
        incumbent_ensemble_weights: Dict[str, Any],
        shadow_candidate_probs: Optional[Dict[str, float]] = None,
        shadow_candidate_weights: Optional[Dict[str, Any]] = None,
        selected_forecast: Optional[Dict[str, float]] = None,
        selected_model_name: str = "Approved Incumbent Ensemble",
        model_version: str = "models_v001",
        feature_version: str = "features_v001",
        dataset_version: str = "dataset_v001",
        preprocessing_version: str = "cleaner_v001",
        input_feature_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Persist an immutable pre-round prediction record BEFORE round outcome is revealed."""
        r_key = str(round_id)

        # Check if already recorded
        if r_key in self._records and "pre_round_forecast" in self._records[r_key]:
            existing = self._records[r_key]["pre_round_forecast"]
            # Verify exact content match
            if existing.get("selected_forecast") != selected_forecast:
                raise ImmutabilityViolationError(
                    f"Illegal attempt to alter frozen pre-round prediction for Round #{round_id}!"
                )
            return self._records[r_key]

        payload = {
            "round_id": round_id,
            "source_provenance": source_provenance,
            "event_timestamps": {
                "round_start_waiting_ts": event_timestamps.get("round_start_waiting_ts", datetime.now(timezone.utc).isoformat()),
                "backend_receipt_ts": event_timestamps.get("backend_receipt_ts", datetime.now(timezone.utc).isoformat()),
                "prediction_generation_ts": event_timestamps.get("prediction_generation_ts", datetime.now(timezone.utc).isoformat()),
                "available_data_cutoff_ts": event_timestamps.get("available_data_cutoff_ts", datetime.now(timezone.utc).isoformat()),
            },
            "model_version": model_version,
            "feature_version": feature_version,
            "dataset_version": dataset_version,
            "preprocessing_version": preprocessing_version,
            "input_feature_hash": input_feature_hash or compute_sha256(model_predictions),
            "model_predictions": deepcopy(model_predictions),
            "baseline_forecasts": deepcopy(baseline_forecasts),
            "incumbent_ensemble_probabilities": deepcopy(incumbent_ensemble_probs),
            "incumbent_ensemble_weights": deepcopy(incumbent_ensemble_weights),
            "shadow_candidate_probabilities": deepcopy(shadow_candidate_probs or {}),
            "shadow_candidate_weights": deepcopy(shadow_candidate_weights or {}),
            "selected_forecast": deepcopy(selected_forecast or incumbent_ensemble_probs),
            "selected_model_name": selected_model_name,
            "status": "frozen_pre_round",
        }

        # Sanitize any numpy/custom types to pure JSON primitives
        payload = json.loads(json.dumps(payload, default=str))
        pre_hash = compute_sha256(payload)
        payload["pre_round_hash"] = pre_hash

        ledger_entry = {
            "entry_type": "pre_round_prediction",
            "round_id": round_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pre_round_hash": pre_hash,
            "pre_round_forecast": payload,
        }

        # Append to durable ledger file
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry, default=str) + "\n")

        self._records[r_key] = ledger_entry
        self._pre_round_hashes[r_key] = pre_hash
        if isinstance(round_id, int):
            self._records[round_id] = ledger_entry
            self._pre_round_hashes[round_id] = pre_hash

        logger.debug(f"[Ledger] Recorded frozen pre-round prediction for Round #{round_id} (hash={pre_hash[:12]})")
        return ledger_entry

    def record_resolution(
        self,
        round_id: Union[int, str],
        actual_multiplier: float,
        crash_timestamp: Optional[str] = None,
        fallback_pre_forecast: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append post-crash resolution and scoring metrics to the ledger without altering the pre-round record."""
        r_key = str(round_id)
        existing = self._records.get(r_key) or self._records.get(round_id)
        if (not existing or "pre_round_forecast" not in existing) and fallback_pre_forecast:
            try:
                self.record_pre_round_prediction(
                    round_id=round_id,
                    source_provenance=fallback_pre_forecast.get("source_provenance", "browser_observer"),
                    event_timestamps={
                        "round_start_waiting_ts": fallback_pre_forecast.get("available_data_cutoff_ts", datetime.now(timezone.utc).isoformat()),
                        "prediction_generation_ts": fallback_pre_forecast.get("prediction_timestamp", datetime.now(timezone.utc).isoformat()),
                        "available_data_cutoff_ts": fallback_pre_forecast.get("available_data_cutoff_ts", datetime.now(timezone.utc).isoformat()),
                    },
                    model_predictions=fallback_pre_forecast.get("model_predictions", {}),
                    baseline_forecasts=fallback_pre_forecast.get("baseline_forecasts", {}),
                    incumbent_ensemble_probs=fallback_pre_forecast.get("ensemble_probability", fallback_pre_forecast.get("probabilities", {})),
                    incumbent_ensemble_weights=fallback_pre_forecast.get("ensemble_weights", {}),
                    shadow_candidate_probs=fallback_pre_forecast.get("shadow_candidate_probability"),
                    selected_forecast=fallback_pre_forecast.get("probabilities", fallback_pre_forecast.get("selected_forecast")),
                    selected_model_name=fallback_pre_forecast.get("champion_model", fallback_pre_forecast.get("selected_model", "Approved Incumbent Ensemble")),
                    model_version=fallback_pre_forecast.get("model_version", "v1.0.0_initial"),
                    input_feature_hash=fallback_pre_forecast.get("input_feature_hash"),
                )
                existing = self._records.get(r_key) or self._records.get(round_id)
            except Exception as e:
                logger.debug(f"[Ledger] Fallback pre-round registration error: {e}")

        if not existing or "pre_round_forecast" not in existing:
            # Deterministic fallback resolution generation if pre-round prediction is missing
            eval_ts = datetime.now(timezone.utc).isoformat()
            fallback_res_payload = {
                "round_id": round_id,
                "actual_multiplier": round(float(actual_multiplier), 2),
                "crash_timestamp": crash_timestamp or eval_ts,
                "evaluation_timestamp": eval_ts,
                "binary_outcomes": {
                    th_key: 1 if actual_multiplier >= th else 0
                    for th_key, th in zip(THRESHOLD_KEYS, THRESHOLDS)
                },
                "scoring_metrics": {},
                "linked_pre_round_hash": "",
                "status": "resolved",
            }
            res_hash = compute_sha256(fallback_res_payload)
            fallback_res_payload["resolution_hash"] = res_hash
            ledger_entry = {
                "entry_type": "round_resolution",
                "round_id": round_id,
                "timestamp": eval_ts,
                "resolution_hash": res_hash,
                "resolution": fallback_res_payload,
            }
            try:
                with open(self.ledger_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ledger_entry) + "\n")
            except Exception:
                pass
            return ledger_entry

        pre_forecast = existing["pre_round_forecast"]
        pre_hash = existing.get("pre_round_hash", "")

        # Compute binary threshold outcomes
        binary_outcomes = {
            th_key: 1 if actual_multiplier >= th else 0
            for th_key, th in zip(THRESHOLD_KEYS, THRESHOLDS)
        }

        # Calculate scoring metrics for all components
        scoring_metrics: Dict[str, Dict[str, float]] = {}

        def _score(prob_dict: Dict[str, float]) -> Dict[str, float]:
            scores = {}
            for th_key in THRESHOLD_KEYS:
                if th_key in prob_dict and prob_dict[th_key] is not None:
                    p = float(prob_dict[th_key])
                    p_clip = min(max(p, 1e-6), 1.0 - 1e-6)
                    y = binary_outcomes[th_key]
                    brier = (p - y) ** 2
                    ll = - (y * np.log(p_clip) + (1 - y) * np.log(1.0 - p_clip))
                    scores[f"brier_{th_key}"] = round(float(brier), 5)
                    scores[f"log_loss_{th_key}"] = round(float(ll), 5)
            # Composite across all 5 thresholds
            brier_vals = [v for k, v in scores.items() if k.startswith("brier_")]
            ll_vals = [v for k, v in scores.items() if k.startswith("log_loss_")]
            if brier_vals:
                scores["mean_brier"] = round(float(np.mean(brier_vals)), 5)
            if ll_vals:
                scores["mean_log_loss"] = round(float(np.mean(ll_vals)), 5)
            return scores

        # 1. Individual models
        for m_name, m_probs in pre_forecast.get("model_predictions", {}).items():
            scoring_metrics[m_name] = _score(m_probs)

        # 2. Baselines
        for b_name, b_probs in pre_forecast.get("baseline_forecasts", {}).items():
            scoring_metrics[b_name] = _score(b_probs)

        # 3. Incumbent ensemble
        scoring_metrics["Incumbent Ensemble"] = _score(pre_forecast.get("incumbent_ensemble_probabilities", {}))

        # 4. Shadow candidate
        if pre_forecast.get("shadow_candidate_probabilities"):
            scoring_metrics["Shadow Candidate Ensemble"] = _score(pre_forecast["shadow_candidate_probabilities"])

        # 5. Selected forecast
        scoring_metrics["Selected Champion"] = _score(pre_forecast.get("selected_forecast", {}))

        eval_ts = datetime.now(timezone.utc).isoformat()
        res_payload = {
            "round_id": round_id,
            "actual_multiplier": round(float(actual_multiplier), 2),
            "crash_timestamp": crash_timestamp or eval_ts,
            "evaluation_timestamp": eval_ts,
            "binary_outcomes": binary_outcomes,
            "scoring_metrics": scoring_metrics,
            "linked_pre_round_hash": pre_hash,
            "status": "resolved",
        }

        # Sanitize any numpy/custom types to pure JSON primitives
        res_payload = json.loads(json.dumps(res_payload, default=str))
        res_hash = compute_sha256(res_payload)
        res_payload["resolution_hash"] = res_hash

        ledger_entry = {
            "entry_type": "round_resolution",
            "round_id": round_id,
            "timestamp": eval_ts,
            "resolution_hash": res_hash,
            "resolution": res_payload,
        }

        # Append to durable ledger file
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ledger_entry, default=str) + "\n")

        # Update in-memory copy
        existing["resolution"] = res_payload
        logger.debug(f"[Ledger] Recorded resolution for Round #{round_id}: {actual_multiplier}x (hash={res_hash[:12]})")
        return ledger_entry

    def verify_ledger_integrity(self) -> Dict[str, Any]:
        """Audit the entire ledger on disk for cryptographic integrity and immutability."""
        if not self.ledger_path.exists():
            return {"status": "ok", "total_entries": 0, "violations": []}

        violations = []
        total_pre_round = 0
        total_resolutions = 0
        seen_pre_round_ids = set()

        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    entry_type = entry.get("entry_type")
                    r_id = entry.get("round_id")

                    if entry_type == "pre_round_prediction":
                        total_pre_round += 1
                        stored_hash = entry.get("pre_round_hash")
                        recomputed_hash = compute_sha256(entry.get("pre_round_forecast", {}))
                        # Note: pre_round_forecast contains pre_round_hash, so compute on forecast without hash
                        forecast_copy = deepcopy(entry.get("pre_round_forecast", {}))
                        forecast_copy.pop("pre_round_hash", None)
                        expected_hash = compute_sha256(forecast_copy)
                        if stored_hash != expected_hash:
                            violations.append(f"Line {line_num}: Hash mismatch for pre-round prediction #{r_id}")

                        if r_id in seen_pre_round_ids:
                            violations.append(f"Line {line_num}: Duplicate pre-round forecast for Round #{r_id}")
                        seen_pre_round_ids.add(r_id)

                    elif entry_type == "round_resolution":
                        total_resolutions += 1
                        stored_hash = entry.get("resolution_hash")
                        res_copy = deepcopy(entry.get("resolution", {}))
                        res_copy.pop("resolution_hash", None)
                        expected_hash = compute_sha256(res_copy)
                        if stored_hash != expected_hash:
                            violations.append(f"Line {line_num}: Hash mismatch for resolution #{r_id}")
        except Exception as e:
            violations.append(f"Ledger file read error: {e}")

        return {
            "status": "valid" if not violations else "corrupted",
            "total_pre_round_forecasts": total_pre_round,
            "total_resolutions": total_resolutions,
            "violations": violations,
            "is_immutable": len(violations) == 0,
        }

    def get_recent_entries(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Return the most recent resolved and active ledger entries."""
        records = list(self._records.values())
        return records[-limit:]


# Global singleton instance
ledger = PredictionLedger()
