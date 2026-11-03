"""Deterministic Dataset Aggregator, Manifest Generator, and Quality Gatekeeper.

Strict Protocol:
- Aggregates approved historical dataset and all validated overnight sessions.
- Never silently selects only the latest session or omits approved datasets.
- Excludes synthetic/demo data, quarantined records, invalid/incomplete records, and unapproved sources.
- Deduplicates using documented key (round_id).
- Detects conflicting duplicate records (same round_id, conflicting multipliers/data),
  quarantines them, and reports them in the reproducible input manifest rather than silently choosing one.
- Generates reproducible input manifest with file path, SHA-256, source, total/valid/rejected rows,
  unique rounds, duplicates removed, and date range.
- Chronologically orders the final aggregated corpus.
- Provides exact final row count entering training.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.collector.verifier import collection_supervisor, parse_iso_timestamp
from src.data.loader import load_csv
from src.utils.logging import get_logger

logger = get_logger("dataset_aggregator")


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute deterministic SHA-256 fingerprint of a file on disk."""
    p = Path(file_path)
    if not p.exists() or p.stat().st_size == 0:
        return "empty_file"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dataframe_sha256(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 fingerprint of a canonical DataFrame."""
    canonical_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


class DatasetAggregator:
    """Aggregates, validates, deduplicates, and manifests all eligible training datasets."""

    def __init__(
        self,
        historical_file: str = "data/raw/real_aviator_historical.csv",
        sessions_dir: str = "data/live/sessions",
        live_csv_path: str = "data/live/stake_collected_rounds.csv",
        live_jsonl_path: str = "data/live/stake_collected_rounds.jsonl",
        quarantine_path: str = "data/live/quarantine_rounds.jsonl",
    ):
        self.historical_file = Path(historical_file)
        self.sessions_dir = Path(sessions_dir)
        self.live_csv_path = Path(live_csv_path)
        self.live_jsonl_path = Path(live_jsonl_path)
        self.quarantine_path = Path(quarantine_path)

    def load_quarantined_round_ids(self) -> Set[Any]:
        """Load set of round IDs present in quarantine log."""
        quarantined = set()
        if self.quarantine_path.exists():
            try:
                with open(self.quarantine_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            r_id = entry.get("round_id") or entry.get("record", {}).get("round_id")
                            if r_id is not None:
                                quarantined.add(r_id)
                                quarantined.add(str(r_id))
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Error reading quarantine log: {e}")
        return quarantined

    def discover_eligible_sources(self) -> List[Dict[str, Any]]:
        """Discover all candidate training files across historical, sessions, and live collector."""
        sources = []

        # 1. Approved Historical Dataset
        if self.historical_file.exists():
            sources.append({
                "path": str(self.historical_file),
                "type": "csv",
                "source": "approved_historical",
                "provenance": "Provably Fair Stake Historical Archive",
            })

        # 2. Frozen overnight sessions in sessions_dir
        if self.sessions_dir.exists():
            # Find all frozen files
            frozen_files = sorted(list(self.sessions_dir.glob("*_frozen.jsonl")) + list(self.sessions_dir.glob("*_frozen.csv")))
            # Group by session_id to avoid loading both .jsonl and .csv of same session
            seen_session_stems = set()
            for ff in frozen_files:
                stem = ff.stem.replace("_frozen", "")
                if stem in seen_session_stems:
                    continue
                # Prefer .jsonl if exists, else .csv
                jsonl_peer = self.sessions_dir / f"{stem}_frozen.jsonl"
                target_file = jsonl_peer if jsonl_peer.exists() else ff
                seen_session_stems.add(stem)

                # Check validation report if available
                report_file = self.sessions_dir / f"{stem}_validation_report.json"
                is_report_valid = None
                if report_file.exists():
                    try:
                        with open(report_file, "r", encoding="utf-8") as rf:
                            rep = json.load(rf)
                            is_report_valid = rep.get("is_valid", False)
                    except Exception:
                        pass

                sources.append({
                    "path": str(target_file),
                    "type": "jsonl" if target_file.suffix == ".jsonl" else "csv",
                    "source": "validated_overnight_session",
                    "provenance": f"Overnight Stake Telemetry ({stem})",
                    "session_id": stem,
                    "validation_report_status": is_report_valid,
                })

        # 3. Active live collector files (if non-empty)
        if self.live_jsonl_path.exists() and self.live_jsonl_path.stat().st_size > 100:
            sources.append({
                "path": str(self.live_jsonl_path),
                "type": "jsonl",
                "source": "active_stake_collector",
                "provenance": "Active Stake Collector Live Export",
            })
        elif self.live_csv_path.exists() and self.live_csv_path.stat().st_size > 100:
            sources.append({
                "path": str(self.live_csv_path),
                "type": "csv",
                "source": "active_stake_collector",
                "provenance": "Active Stake Collector Live Export",
            })

        return sources

    def aggregate_and_manifest(
        self,
        custom_file: Optional[str] = None,
        min_valid_threshold: int = 1,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform reliable dataset aggregation, conflict checking, deduplication, and manifest generation.

        Returns:
            (df_aggregated, manifest_dict)
        """
        quarantined_ids = self.load_quarantined_round_ids()
        discovered = self.discover_eligible_sources()

        if custom_file and Path(custom_file).exists():
            # Include custom file explicitly
            p_custom = Path(custom_file)
            discovered.insert(1, {
                "path": str(p_custom),
                "type": "jsonl" if p_custom.suffix == ".jsonl" else "csv",
                "source": "custom_override",
                "provenance": f"User Provided Candidate ({p_custom.name})",
            })

        file_manifests: List[Dict[str, Any]] = []
        all_candidate_records: List[Dict[str, Any]] = []

        total_raw_rows = 0
        total_valid_rows = 0
        total_rejected_rows = 0

        # Pass 1: Parse and validate individual files
        for src_meta in discovered:
            file_path = Path(src_meta["path"])
            file_sha = compute_file_sha256(file_path)
            src_type = src_meta["type"]
            src_label = src_meta["source"]

            file_records: List[Dict[str, Any]] = []
            if src_type == "jsonl":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    file_records.append(json.loads(line))
                                except Exception:
                                    pass
                except Exception as e:
                    logger.warning(f"Error reading JSONL {file_path}: {e}")
            else:
                try:
                    df_raw = pd.read_csv(file_path)
                    file_records = df_raw.to_dict(orient="records")
                except Exception as e:
                    logger.warning(f"Error reading CSV {file_path}: {e}")

            f_total = len(file_records)
            total_raw_rows += f_total

            f_valid = 0
            f_rejected = 0
            f_unique_ids = set()
            f_timestamps = []

            for rec in file_records:
                r_id = rec.get("round_id")
                if r_id is None:
                    f_rejected += 1
                    continue

                r_str = str(r_id)
                if "test" in r_str.lower() or "demo" in r_str.lower():
                    f_rejected += 1
                    continue

                # Check quarantined
                if r_id in quarantined_ids or r_str in quarantined_ids:
                    f_rejected += 1
                    continue

                # Source authenticity
                s_source = str(rec.get("source_provenance") or rec.get("source") or "").strip().lower()
                if s_source in ("synthetic", "sample_stream", "demo", "test_fixture"):
                    f_rejected += 1
                    continue

                # Multiplier validity
                m_val = rec.get("actual_multiplier")
                if m_val is None:
                    m_val = rec.get("multiplier")
                try:
                    m_float = float(m_val)
                    if np.isnan(m_float) or np.isinf(m_float) or m_float < 1.0:
                        f_rejected += 1
                        continue
                except (ValueError, TypeError):
                    f_rejected += 1
                    continue

                # Incomplete or quarantined in record
                if rec.get("validation_status") in ("INCOMPLETE", "REJECTED"):
                    f_rejected += 1
                    continue
                if rec.get("quarantine_reasons"):
                    f_rejected += 1
                    continue

                # Timestamp
                ts = rec.get("timestamp") or rec.get("source_timestamp") or rec.get("event_timestamp")
                if ts:
                    f_timestamps.append(str(ts))

                # Normalize standard minimal fields for training
                clean_rec = {
                    "round_id": int(r_id) if str(r_id).isdigit() else r_id,
                    "multiplier": round(m_float, 2),
                    "timestamp": str(ts) if ts else datetime.now(timezone.utc).isoformat(),
                    "source": s_source or src_label,
                    "origin_file": file_path.name,
                    "raw_record": rec,
                }

                # Detailed research schema validation if from live collection
                if src_label in ("validated_overnight_session", "active_stake_collector"):
                    # Check for empty hashes or non-monotonic probabilities
                    prov = rec.get("provenance_hashes", {})
                    pre_hash = prov.get("pre_round_hash") or rec.get("pre_round_hash")
                    res_hash = prov.get("resolution_hash") or rec.get("resolution_hash")

                    # If marked VALID in file, verify hashes aren't empty strings
                    if pre_hash == "" and res_hash == "":
                        # Legacy unhashed record: reject from training per Requirement 3
                        f_rejected += 1
                        continue

                    # Check model monotonicity
                    model_preds = rec.get("model_predictions", {})
                    has_mono_violation = False
                    for m_name, preds in model_preds.items():
                        if isinstance(preds, dict):
                            th_probs = [preds.get(k) for k in ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"] if preds.get(k) is not None]
                            if len(th_probs) == 5:
                                if any(th_probs[i] < th_probs[i+1] - 1e-4 for i in range(4)):
                                    has_mono_violation = True
                                    break
                    if has_mono_violation:
                        f_rejected += 1
                        continue

                f_valid += 1
                f_unique_ids.add(clean_rec["round_id"])
                all_candidate_records.append(clean_rec)

            total_valid_rows += f_valid
            total_rejected_rows += f_rejected

            date_min = min(f_timestamps) if f_timestamps else "N/A"
            date_max = max(f_timestamps) if f_timestamps else "N/A"

            file_manifests.append({
                "file_path": str(file_path),
                "file_name": file_path.name,
                "sha256": file_sha,
                "source": src_label,
                "total_rows": f_total,
                "valid_rows": f_valid,
                "rejected_rows": f_rejected,
                "unique_rounds": len(f_unique_ids),
                "duplicates_in_file": f_valid - len(f_unique_ids) if f_valid > len(f_unique_ids) else 0,
                "date_range": {"min": date_min, "max": date_max},
            })

        # Pass 2: Global Deduplication and Conflict Detection
        rounds_by_id: Dict[Any, List[Dict[str, Any]]] = {}
        for r in all_candidate_records:
            rounds_by_id.setdefault(r["round_id"], []).append(r)

        canonical_records: List[Dict[str, Any]] = []
        conflicts_detected: List[Dict[str, Any]] = []
        duplicates_removed_count = 0

        for r_id, instances in rounds_by_id.items():
            if len(instances) == 1:
                canonical_records.append(instances[0])
            else:
                # Check for conflicting multipliers
                mults = [inst["multiplier"] for inst in instances]
                max_diff = max(mults) - min(mults)
                if max_diff > 0.05:  # Conflicting round outcome!
                    conflict_info = {
                        "round_id": r_id,
                        "multipliers_found": mults,
                        "sources": [inst["origin_file"] for inst in instances],
                        "reason": f"Discrepancy of {max_diff:.2f}x detected between duplicate records.",
                    }
                    conflicts_detected.append(conflict_info)
                    logger.warning(f"[Dataset Aggregator] Conflicting duplicate records for Round #{r_id}: {mults}")
                    # Exclude conflicting records from training!
                else:
                    # Non-conflicting duplicate: choose canonical with richest provenance
                    # Prefer live record with full research metadata if available
                    canonical = instances[0]
                    for inst in instances:
                        if "model_predictions" in inst.get("raw_record", {}):
                            canonical = inst
                            break
                    canonical_records.append(canonical)
                    duplicates_removed_count += (len(instances) - 1)

        # Build final aggregated DataFrame
        if canonical_records:
            df_final = pd.DataFrame([
                {
                    "round_id": r["round_id"],
                    "multiplier": r["multiplier"],
                    "timestamp": r["timestamp"],
                }
                for r in canonical_records
            ])
            # Strict chronological sort
            df_final["parsed_dt"] = pd.to_datetime(df_final["timestamp"], utc=True, errors="coerce")
            df_final = df_final.sort_values(by=["parsed_dt", "round_id"]).drop(columns=["parsed_dt"]).reset_index(drop=True)
        else:
            df_final = pd.DataFrame(columns=["round_id", "multiplier", "timestamp"])

        corpus_sha = compute_dataframe_sha256(df_final)

        # Build overall manifest
        min_ts = df_final["timestamp"].min() if not df_final.empty else "N/A"
        max_ts = df_final["timestamp"].max() if not df_final.empty else "N/A"

        manifest = {
            "manifest_version": "v1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus_sha256": corpus_sha,
            "total_input_files": len(file_manifests),
            "files": file_manifests,
            "summary": {
                "total_raw_rows": total_raw_rows,
                "total_valid_rows": total_valid_rows,
                "total_rejected_rows": total_rejected_rows,
                "conflicts_detected_count": len(conflicts_detected),
                "conflicts": conflicts_detected,
                "duplicates_removed_count": duplicates_removed_count,
                "final_training_row_count": len(df_final),
                "unique_round_ids": len(canonical_records),
                "date_range": {"start": min_ts, "end": max_ts},
            },
        }

        return df_final, manifest


# Global aggregator singleton
dataset_aggregator = DatasetAggregator()
