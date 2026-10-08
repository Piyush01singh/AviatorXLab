"""Data cleaner and dataset versioning manager."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from src.utils.logging import get_logger

logger = get_logger("data_cleaner")


class DataCleaner:
    """Cleans crash game datasets and computes cryptographic/statistical version metadata."""

    def __init__(self, min_multiplier: float = 1.00):
        self.min_multiplier = min_multiplier

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean dataset by removing invalid records and enforcing strict temporal ordering.
        
        Args:
            df: Raw input DataFrame.
            
        Returns:
            Tuple of (cleaned_df, cleaning_metadata).
        """
        initial_count = len(df)
        df_clean = df.copy()

        # 1. Parse timestamps
        try:
            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], format="ISO8601")
        except Exception:
            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")
        invalid_ts_count = int(df_clean["timestamp"].isnull().sum())
        df_clean = df_clean.dropna(subset=["timestamp"])

        # 2. Parse multipliers
        df_clean["multiplier"] = pd.to_numeric(df_clean["multiplier"], errors="coerce")
        invalid_mult_count = int((df_clean["multiplier"].isnull() | (df_clean["multiplier"] < self.min_multiplier)).sum())
        df_clean = df_clean[df_clean["multiplier"] >= self.min_multiplier]

        # 3. Handle duplicates
        duplicate_records = int(df_clean.duplicated().sum())
        df_clean = df_clean.drop_duplicates()

        # Remove duplicate round_ids if any remain, keeping the first
        dup_rounds = int(df_clean["round_id"].duplicated().sum())
        df_clean = df_clean.drop_duplicates(subset=["round_id"], keep="first")

        # 4. Strict chronological sorting
        df_clean = df_clean.sort_values(by=["timestamp", "round_id"]).reset_index(drop=True)

        metadata = {
            "initial_rows": initial_count,
            "final_rows": len(df_clean),
            "dropped_invalid_timestamps": invalid_ts_count,
            "dropped_invalid_multipliers": invalid_mult_count,
            "dropped_duplicate_records": duplicate_records,
            "dropped_duplicate_rounds": dup_rounds,
        }
        logger.info(f"Cleaned dataset: {initial_count} -> {len(df_clean)} rows.")
        return df_clean, metadata

    @staticmethod
    def generate_version_metadata(
        df: pd.DataFrame, dataset_version: str, extra_meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate reproducible dataset version metadata as required by Section 27.
        
        Args:
            df: Cleaned DataFrame.
            dataset_version: Identifier string (e.g., 'dataset_v001').
            extra_meta: Optional additional metrics to log.
            
        Returns:
            Metadata dictionary including row counts, date ranges, and SHA256 hash.
        """
        # Compute SHA256 of the CSV representation for exact provenance
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        sha256_hash = hashlib.sha256(csv_bytes).hexdigest()

        date_min = str(df["timestamp"].min()) if not df.empty and "timestamp" in df.columns else None
        date_max = str(df["timestamp"].max()) if not df.empty and "timestamp" in df.columns else None

        metadata = {
            "dataset_version": dataset_version,
            "row_count": len(df),
            "date_range": {
                "start": date_min,
                "end": date_max,
            },
            "columns": list(df.columns),
            "missing_values": {col: int(df[col].isnull().sum()) for col in df.columns},
            "sha256": sha256_hash,
        }

        if extra_meta:
            metadata.update(extra_meta)

        return metadata

    def save_processed(
        self, df: pd.DataFrame, output_dir: str, dataset_version: str, extra_meta: Optional[Dict[str, Any]] = None
    ) -> Tuple[Path, Path]:
        """Save cleaned data and its metadata file.
        
        Returns:
            Tuple of (data_filepath, metadata_filepath).
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        data_file = out_path / f"{dataset_version}.csv"
        meta_file = out_path / f"{dataset_version}_metadata.json"

        df.to_csv(data_file, index=False)

        metadata = self.generate_version_metadata(df, dataset_version, extra_meta)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved processed dataset to {data_file} and metadata to {meta_file}")
        return data_file, meta_file
