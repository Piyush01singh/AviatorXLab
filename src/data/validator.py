"""Data validation system for crash game historical rounds.

Verifies schema, checks for duplicates, missing rounds, non-chronological order,
and invalid multiplier values.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Detailed report on dataset integrity."""
    total_rows: int = 0
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return formatted human-readable summary of validation."""
        lines = [
            f"=== Validation Report (Total Rows: {self.total_rows}) ===",
            f"Status: {'VALID' if self.is_valid else 'INVALID'}",
        ]
        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  [ERROR] {err}")
        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  [WARN]  {warn}")
        if self.stats:
            lines.append("Statistics:")
            for k, v in self.stats.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class DataValidator:
    """Validates crash game round data against structural and domain constraints."""

    REQUIRED_COLUMNS = ["round_id", "timestamp", "multiplier"]

    def __init__(self, min_multiplier: float = 1.00):
        self.min_multiplier = min_multiplier

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Perform comprehensive data integrity validation.
        
        Args:
            df: Raw DataFrame containing game records.
            
        Returns:
            ValidationReport with errors, warnings, and distribution stats.
        """
        report = ValidationReport(total_rows=len(df))

        if df.empty:
            report.is_valid = False
            report.errors.append("Dataset is empty.")
            return report

        # 1. Required Columns Check
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            report.is_valid = False
            report.errors.append(f"Missing required columns: {missing_cols}")
            return report

        # 2. Check for missing (NaN/None) values in required columns
        for col in self.REQUIRED_COLUMNS:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                report.is_valid = False
                report.errors.append(f"Column '{col}' contains {null_count} null/missing values.")

        # 3. Check for Duplicate Records
        exact_duplicates = df.duplicated().sum()
        if exact_duplicates > 0:
            report.warnings.append(f"Found {exact_duplicates} exact duplicate records.")

        # 4. Check for Duplicate Round IDs
        dup_rounds = df["round_id"].duplicated().sum()
        if dup_rounds > 0:
            report.is_valid = False
            report.errors.append(f"Found {dup_rounds} duplicate round_id values.")

        # 5. Check Multiplier Validity
        try:
            multipliers = pd.to_numeric(df["multiplier"], errors="coerce")
            invalid_num = multipliers.isnull().sum()
            if invalid_num > 0:
                report.is_valid = False
                report.errors.append(f"Found {invalid_num} non-numeric multiplier entries.")

            below_min = (multipliers < self.min_multiplier).sum()
            if below_min > 0:
                report.is_valid = False
                report.errors.append(
                    f"Found {below_min} multipliers below theoretical minimum ({self.min_multiplier}x)."
                )

            is_inf = np.isinf(multipliers).sum()
            if is_inf > 0:
                report.is_valid = False
                report.errors.append(f"Found {is_inf} infinite multiplier values.")

            if not multipliers.dropna().empty:
                report.stats["min_multiplier"] = float(multipliers.min())
                report.stats["max_multiplier"] = float(multipliers.max())
                report.stats["median_multiplier"] = float(multipliers.median())
                report.stats["mean_multiplier"] = float(multipliers.mean())
        except Exception as e:
            report.is_valid = False
            report.errors.append(f"Failed to validate multiplier values: {str(e)}")

        # 6. Check Timestamps and Chronological Ordering
        try:
            try:
                timestamps = pd.to_datetime(df["timestamp"], format="ISO8601")
            except Exception:
                timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
            invalid_ts = timestamps.isnull().sum()
            if invalid_ts > 0:
                report.is_valid = False
                report.errors.append(f"Found {invalid_ts} unparseable timestamps.")

            # Check if chronological
            if timestamps.is_monotonic_increasing is False:
                report.warnings.append("Timestamps are not strictly in chronological order.")
        except Exception as e:
            report.is_valid = False
            report.errors.append(f"Timestamp parsing failed: {str(e)}")

        # 7. Check for Round ID Sequence Gaps
        try:
            numeric_ids = pd.to_numeric(df["round_id"], errors="coerce")
            if numeric_ids.isnull().sum() == 0 and len(numeric_ids) > 1:
                # If sorted, diff should be 1
                diffs = numeric_ids.diff().dropna()
                gaps = (diffs > 1).sum()
                if gaps > 0:
                    report.warnings.append(
                        f"Detected {gaps} sequence gaps in round_id (e.g. missing consecutive rounds)."
                    )
        except Exception:
            pass

        return report
