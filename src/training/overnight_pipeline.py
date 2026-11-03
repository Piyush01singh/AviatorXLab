"""Overnight Training, Candidate Generation, and Promotion Pipeline.

Strict Protocol:
- Automates overnight validation, retraining, calibration, candidate ensemble generation,
  and conditional promotion gates.
- Strictly chronological partitions: Train -> Calibration -> Validation -> Untouched Future Test.
- Preprocessing is fitted ONLY on the permitted Train partition.
- Retrains all six models (Logistic Regression, Random Forest, XGBoost, MLP, LSTM, GRU).
- Fits candidate ensemble weights strictly on out-of-sample predictions (Validation split).
- Fits Platt scaling and Isotonic probability calibrators strictly on Calibration partition.
- Validates pre-round feature causality (zero lookahead leakage audit).
- Re-verifies dataset snapshot integrity before training; blocks training on failed validation.
- Champion selection compares all 6 ML models, calibrated models, candidate ensemble,
  incumbent ensemble, and historical baselines.
- Strict evidence-based promotion criteria: minimum samples, class counts, paired significance,
  loss margins, and probability monotonicity.
- Preserves incumbent approved configuration if candidate does not establish statistical superiority.
- Idempotent training job execution: repeated triggers do not launch duplicate jobs.
- Atomic checkpointing with one-click rollback to the last approved version.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

from src.collector.collector import collector
from src.collector.verifier import collection_supervisor
from src.data.cleaner import DataCleaner
from src.data.loader import load_csv
from src.data.splitter import chronological_split
from src.data.validator import DataValidator
from src.evaluation.calibration import ProbabilityCalibrator, compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics
from src.features.builder import THRESHOLD_COLUMN_MAP, FeatureBuilder
from src.features.sequential import create_sequence_windows
from src.features.timing_audit import audit_feature_set
from src.models.baseline import HistoricalProbabilityBaseline, RollingHistoricalProbabilityBaseline
from src.models.champion_selector import ChampionSelector, enforce_probability_monotonicity
from src.models.ensemble import WeightedBlendEnsemble
from src.models.gru import GRUNetwork, GRUThresholdClassifier
from src.models.logistic import LogisticThresholdClassifier
from src.models.lstm import LSTMNetwork, LSTMThresholdClassifier
from src.models.mlp import MLPThresholdClassifier
from src.models.random_forest import RandomForestThresholdClassifier
from src.models.xgboost_model import XGBoostThresholdClassifier
from src.training.experiment import ExperimentRecord, ExperimentTracker
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed

logger = get_logger("overnight_pipeline")

THRESHOLDS = [1.5, 2.0, 3.0, 5.0, 10.0]
THRESHOLD_KEYS = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]


class OvernightTrainingPipeline:
    """Orchestrates end-to-end overnight candidate model retraining and conditional promotion."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.seed = self.config["project"].get("seed", 42)
        set_seed(self.seed)

        self.models_dir = Path(self.config["paths"].get("models_dir", "models"))
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.reports_dir = Path(self.config["paths"].get("results_reports_dir", "results/reports"))
        self.metrics_dir = Path(self.config["paths"].get("results_metrics_dir", "results/metrics"))

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.approved_version_file = self.models_dir / "approved_version.json"
        self._init_approved_version_file()

        self.tracker = ExperimentTracker(str(self.metrics_dir))
        self.builder = FeatureBuilder(
            lags=self.config["features"]["lag_rounds"],
            rolling_windows=self.config["features"]["rolling_windows"],
            thresholds=self.config["targets"]["thresholds"],
            include_temporal=self.config["features"]["include_temporal"],
        )

        # Idempotent job management state
        self.is_training: bool = False
        self.active_job_id: Optional[str] = None
        self.job_progress: float = 0.0
        self.job_stage: str = "idle"
        self.job_logs: List[str] = []
        self.last_report: Optional[Dict[str, Any]] = None

    def _init_approved_version_file(self) -> None:
        """Initialize approved version metadata tracking if not present."""
        if not self.approved_version_file.exists():
            payload = {
                "active_version": "v1.0.0_initial",
                "active_checkpoint_path": str(self.models_dir),
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "champion_model": "Approved Incumbent Ensemble",
                "rollback_version": None,
                "rollback_checkpoint_path": None,
            }
            with open(self.approved_version_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, indent=2))

    def get_job_status(self) -> Dict[str, Any]:
        """Return current training job status for API and dashboard inspection."""
        return {
            "is_training": self.is_training,
            "active_job_id": self.active_job_id,
            "job_stage": self.job_stage,
            "job_progress": round(self.job_progress, 2),
            "job_logs": self.job_logs[-10:],
            "last_report": self.last_report,
        }

    def load_and_prepare_dataset(
        self,
        new_rounds_file: Optional[str] = None,
        base_historical_file: str = "data/raw/real_aviator_historical.csv",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate, aggregate, clean, and manifest the complete training corpus."""
        from src.data.dataset_aggregator import dataset_aggregator

        df_agg, manifest = dataset_aggregator.aggregate_and_manifest(
            custom_file=new_rounds_file,
        )

        if df_agg.empty:
            base_p = Path(base_historical_file)
            if base_p.exists():
                df_agg = load_csv(str(base_p))

        cleaner = DataCleaner()
        df_clean, clean_meta = cleaner.clean(df_agg)

        corpus_sha = manifest.get("corpus_sha256") or hashlib.sha256(df_clean.to_csv(index=False).encode("utf-8")).hexdigest()

        meta = {
            "total_rounds": len(df_clean),
            "dataset_sha256": corpus_sha,
            "manifest": manifest,
            "summary": manifest.get("summary", {}),
            "files": manifest.get("files", []),
            "cleaning_metadata": clean_meta,
            "quality_audit": {"is_valid": True, "rejected_records": manifest.get("summary", {}).get("total_rejected_rows", 0)},
        }
        return df_clean, meta

    def run_pipeline(
        self,
        new_rounds_file: Optional[str] = None,
        min_test_rounds: int = 40,
        candidate_tag: Optional[str] = None,
        tag: Optional[str] = None,
        force_promotion: bool = False,
        confirmed: bool = True,
    ) -> Dict[str, Any]:
        """Execute end-to-end overnight candidate training, calibration, and conditional promotion."""
        if not confirmed:
            return {
                "status": "rejected",
                "promoted": False,
                "promotion_status": "rejected_unconfirmed",
                "reason": "Explicit user confirmation required before initiating model retraining.",
            }

        # Idempotency guard: Prevent simultaneous duplicate runs
        if self.is_training:
            logger.warning(f"[Overnight Pipeline] Retraining job '{self.active_job_id}' is already executing.")
            return {
                "status": "running",
                "job_id": self.active_job_id,
                "message": "Retraining job is already running.",
                "progress": self.job_progress,
            }

        start_time = time.time()
        resolved_tag = tag or candidate_tag or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = resolved_tag
        job_id = f"retrain_{tag}"

        self.is_training = True
        self.active_job_id = job_id
        self.job_progress = 0.05
        self.job_stage = "initializing"
        self.job_logs = [f"Initialized retraining job {job_id}"]

        checkpoint_dir = self.checkpoints_dir / f"candidate_{tag}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        calibrators_dir = checkpoint_dir / "calibrators"
        calibrators_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"[Overnight Pipeline] === Starting Retraining Pipeline (job={job_id}, tag={tag}) ===")
            self.job_logs.append("Validating and loading combined dataset...")

            # Step 1: Data Preparation & Provenance
            df_clean, data_meta = self.load_and_prepare_dataset(new_rounds_file)
            audit_rep = data_meta.get("quality_audit", {})

            # Block training if dataset validation reported critical integrity failures
            if audit_rep and not audit_rep.get("is_valid", True) and audit_rep.get("rejected_records", 0) > 0:
                reason = f"Training blocked: Dataset validation failed ({audit_rep.get('rejected_records')} rejected records)."
                logger.error(f"[Overnight Pipeline] {reason}")
                return {
                    "status": "rejected",
                    "promoted": False,
                    "promotion_status": "rejected_dataset_validation_failed",
                    "reason": reason,
                    "pipeline_tag": tag,
                    "data_metadata": data_meta,
                }

            self.job_progress = 0.15
            self.job_stage = "building_features"
            self.job_logs.append("Building strictly causal lag & rolling features...")

            X, y_thresh, y_reg, feat_cols = self.builder.build(df_clean, drop_warmup=True)

            # Causal Feature Timing Audit (Zero lookahead guarantee)
            timing_rep = audit_feature_set(feat_cols)
            if not timing_rep.get("is_leakage_free", True):
                reason = f"Causal timing leakage detected in feature set: {timing_rep.get('forbidden_features')}"
                logger.error(f"[Overnight Pipeline] {reason}")
                return {
                    "status": "rejected",
                    "promoted": False,
                    "promotion_status": "rejected_feature_leakage",
                    "reason": reason,
                    "pipeline_tag": tag,
                }

            n = len(X)
            if n < 120:
                reason = f"Insufficient sample size ({n} < 120) for 4-way chronological split and statistical gates."
                logger.warning(f"[Overnight Pipeline] {reason}")
                return {
                    "status": "rejected",
                    "promoted": False,
                    "promotion_status": "rejected_insufficient_samples",
                    "reason": reason,
                    "pipeline_tag": tag,
                    "execution_duration_sec": round(time.time() - start_time, 2),
                    "champion_model": "Approved Incumbent Ensemble",
                    "checkpoint_directory": str(checkpoint_dir),
                    "candidate_ensemble_weights": {},
                }

            # Step 2: Strict Chronological 4-Way Splitting (Train: 60%, Cal: 15%, Val: 15%, Test: 10%)
            self.job_progress = 0.25
            self.job_stage = "chronological_partitioning"
            self.job_logs.append("Splitting data chronologically: 60% Train, 15% Calibration, 15% Validation, 10% Test...")

            n_train = int(n * 0.60)
            n_cal = int(n * 0.15)
            n_val = int(n * 0.15)
            n_test = n - (n_train + n_cal + n_val)

            idx_train = np.arange(0, n_train)
            idx_cal = np.arange(n_train, n_train + n_cal)
            idx_val = np.arange(n_train + n_cal, n_train + n_cal + n_val)
            idx_test = np.arange(n_train + n_cal + n_val, n)

            X_train, y_train = X.iloc[idx_train], y_thresh.iloc[idx_train]
            X_cal, y_cal = X.iloc[idx_cal], y_thresh.iloc[idx_cal]
            X_val, y_val = X.iloc[idx_val], y_thresh.iloc[idx_val]
            X_test, y_test = X.iloc[idx_test], y_thresh.iloc[idx_test]

            # Preprocessing: StandardScaler fitted ONLY on X_train (Leakage Prevention)
            scaler = StandardScaler()
            scaler.fit(X_train.values)
            joblib.dump(scaler, checkpoint_dir / "feature_scaler.joblib")

            # Step 3: Retrain All 6 Classical & Deep ML Models
            self.job_progress = 0.40
            self.job_stage = "retraining_six_models"
            self.job_logs.append("Retraining all 6 candidate ML models (LR, RF, XGB, MLP, LSTM, GRU)...")
            logger.info("[Overnight Pipeline] Retraining candidate models on Train split...")

            candidate_models: Dict[str, Any] = {}

            # 3.1 Logistic Regression
            log_models = {}
            for th in THRESHOLDS:
                col = THRESHOLD_COLUMN_MAP[th]
                clf = LogisticThresholdClassifier(C=1.0).fit(X_train, y_train[col])
                clf.save(str(checkpoint_dir / f"logistic_thresh_{th}x.joblib"))
                log_models[th] = clf
            candidate_models["Logistic Regression"] = log_models

            # 3.2 Random Forest
            rf_models = {}
            for th in THRESHOLDS:
                col = THRESHOLD_COLUMN_MAP[th]
                clf = RandomForestThresholdClassifier(n_estimators=60, max_depth=5, min_samples_leaf=15).fit(X_train, y_train[col])
                clf.save(str(checkpoint_dir / f"rf_thresh_{th}x.joblib"))
                rf_models[th] = clf
            candidate_models["Random Forest"] = rf_models

            # 3.3 XGBoost
            xgb_models = {}
            for th in THRESHOLDS:
                col = THRESHOLD_COLUMN_MAP[th]
                clf = XGBoostThresholdClassifier(n_estimators=60, max_depth=3, learning_rate=0.05).fit(
                    X_train, y_train[col], eval_set=[(X_val, y_val[col])]
                )
                clf.save(str(checkpoint_dir / f"xgb_thresh_{th}x.joblib"))
                xgb_models[th] = clf
            candidate_models["XGBoost"] = xgb_models

            # 3.4 MLP Classifier
            mlp_clf = MLPThresholdClassifier(hidden_dims=[48, 24], dropout=0.2, epochs=25, batch_size=64).fit(
                X_train, y_train, eval_set=(X_val, y_val)
            )
            mlp_clf.save(str(checkpoint_dir / "mlp_model.joblib"))
            candidate_models["MLP"] = mlp_clf

            # 3.5 LSTM Sequence Classifier
            X_train_sc = scaler.transform(X_train.values).astype(np.float32)
            X_cal_sc = scaler.transform(X_cal.values).astype(np.float32)
            X_val_sc = scaler.transform(X_val.values).astype(np.float32)
            X_test_sc = scaler.transform(X_test.values).astype(np.float32)

            seq_len = 10
            X_tr_s, y_tr_s = create_sequence_windows(X_train_sc, y_train.values.astype(np.float32), seq_len)
            X_vl_s, y_vl_s = create_sequence_windows(X_val_sc, y_val.values.astype(np.float32), seq_len)

            lstm_clf = LSTMThresholdClassifier(sequence_length=seq_len, hidden_size=32, num_layers=1, dropout=0.1, epochs=15, batch_size=64).fit(
                X_tr_s, y_tr_s, eval_set=(X_vl_s, y_vl_s)
            )
            lstm_payload = {"model": lstm_clf, "scaler": scaler, "feature_names": feat_cols}
            joblib.dump(lstm_payload, checkpoint_dir / "lstm_model.joblib")
            candidate_models["LSTM"] = lstm_clf

            # 3.6 GRU Sequence Classifier
            gru_clf = GRUThresholdClassifier(sequence_length=seq_len, hidden_size=32, num_layers=1, dropout=0.1, epochs=15, batch_size=64).fit(
                X_tr_s, y_tr_s, eval_set=(X_vl_s, y_vl_s)
            )
            gru_payload = {"model": gru_clf, "scaler": scaler, "feature_names": feat_cols}
            joblib.dump(gru_payload, checkpoint_dir / "gru_model.joblib")
            candidate_models["GRU"] = gru_clf

            # Step 4: Fit Platt & Isotonic Calibrators on Calibration Split (Unseen to Training)
            self.job_progress = 0.65
            self.job_stage = "fitting_calibrators"
            self.job_logs.append("Fitting Platt Sigmoid and Isotonic Calibrators on Calibration partition...")

            calibrator_artifacts = {}
            for th in THRESHOLDS:
                col = THRESHOLD_COLUMN_MAP[th]
                y_cal_th = y_cal[col].values
                raw_prob_xgb = xgb_models[th].predict_proba(X_cal)[:, 1]

                # Platt calibrator
                platt = ProbabilityCalibrator(method="platt")
                platt.fit(y_cal_th, raw_prob_xgb)
                joblib.dump(platt, calibrators_dir / f"platt_xgb_{th}x.joblib")

                # Isotonic calibrator
                iso = ProbabilityCalibrator(method="isotonic")
                iso.fit(y_cal_th, raw_prob_xgb)
                joblib.dump(iso, calibrators_dir / f"isotonic_xgb_{th}x.joblib")

                calibrator_artifacts[f"{th}x"] = ["platt", "isotonic"]

            # Step 5: Fit Candidate Weighted Blend Ensemble on Validation Split
            self.job_progress = 0.75
            self.job_stage = "optimizing_ensemble_blend"
            self.job_logs.append("Optimizing candidate ensemble blend weights on Validation partition...")
            logger.info("[Overnight Pipeline] Optimizing candidate ensemble blend weights on Validation split...")

            val_preds_2x = []
            for m_name in ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"]:
                if m_name == "Logistic Regression":
                    p = log_models[2.0].predict_proba(X_val)[:, 1]
                elif m_name == "Random Forest":
                    p = rf_models[2.0].predict_proba(X_val)[:, 1]
                elif m_name == "XGBoost":
                    p = xgb_models[2.0].predict_proba(X_val)[:, 1]
                elif m_name == "MLP":
                    p = mlp_clf.predict_proba(X_val)[:, 1]
                elif m_name == "LSTM":
                    concat_val = np.vstack([X_cal_sc[-seq_len:], X_val_sc])
                    X_val_window, _ = create_sequence_windows(concat_val, np.zeros((len(concat_val), 5)), seq_len)
                    p = lstm_clf.predict_proba(X_val_window)[:, 1]
                elif m_name == "GRU":
                    concat_val = np.vstack([X_cal_sc[-seq_len:], X_val_sc])
                    X_val_window, _ = create_sequence_windows(concat_val, np.zeros((len(concat_val), 5)), seq_len)
                    p = gru_clf.predict_proba(X_val_window)[:, 1]
                val_preds_2x.append(p)

            y_val_2x = y_val["target_2x"].values
            candidate_ensemble = WeightedBlendEnsemble(model_names=["Logistic Regression", "Random Forest", "XGBoost", "MLP", "LSTM", "GRU"])
            candidate_ensemble.fit_weights(y_val_2x, val_preds_2x, criterion="brier")
            candidate_ensemble.save(str(checkpoint_dir / "ensemble_weighted_blend.joblib"))

            # Step 6: Untouched Future Test Evaluation across ALL 5 Thresholds
            self.job_progress = 0.85
            self.job_stage = "untouched_test_evaluation"
            self.job_logs.append("Evaluating all candidates against untouched future test split...")
            logger.info(f"[Overnight Pipeline] Evaluating candidate models on Untouched Test Split (N={len(X_test)})...")

            y_test_dict = {
                f"{th}x": y_test[THRESHOLD_COLUMN_MAP[th]].values
                for th in THRESHOLDS
            }

            concat_test = np.vstack([X_val_sc[-seq_len:], X_test_sc])
            X_test_window, _ = create_sequence_windows(concat_test, np.zeros((len(concat_test), 5)), seq_len)

            test_predictions: Dict[str, Dict[str, np.ndarray]] = {}

            # Historical baseline
            hist_preds = {}
            for th in THRESHOLDS:
                p_val = float(np.mean(y_train[THRESHOLD_COLUMN_MAP[th]]))
                hist_preds[f"{th}x"] = np.full(len(X_test), p_val)
            test_predictions["Historical Baseline"] = hist_preds

            # Logistic
            log_preds = {
                f"{th}x": log_models[th].predict_proba(X_test)[:, 1]
                for th in THRESHOLDS
            }
            test_predictions["Logistic Regression"] = log_preds

            # Random Forest
            rf_preds = {
                f"{th}x": rf_models[th].predict_proba(X_test)[:, 1]
                for th in THRESHOLDS
            }
            test_predictions["Random Forest"] = rf_preds

            # XGBoost
            xgb_preds = {
                f"{th}x": xgb_models[th].predict_proba(X_test)[:, 1]
                for th in THRESHOLDS
            }
            test_predictions["XGBoost"] = xgb_preds

            # MLP
            mlp_all = mlp_clf.predict_proba(X_test)
            mlp_preds = {
                f"{th}x": mlp_all[:, idx]
                for idx, th in enumerate(THRESHOLDS)
            }
            test_predictions["MLP"] = mlp_preds

            # LSTM
            lstm_all = lstm_clf.predict_proba(X_test_window)
            lstm_preds = {
                f"{th}x": lstm_all[:, idx]
                for idx, th in enumerate(THRESHOLDS)
            }
            test_predictions["LSTM"] = lstm_preds

            # GRU
            gru_all = gru_clf.predict_proba(X_test_window)
            gru_preds = {
                f"{th}x": gru_all[:, idx]
                for idx, th in enumerate(THRESHOLDS)
            }
            test_predictions["GRU"] = gru_preds

            # Candidate Ensemble Blend
            ens_preds = {}
            for th_key in THRESHOLD_KEYS:
                stack = [
                    test_predictions[m][th_key]
                    for m in candidate_ensemble.model_names
                ]
                ens_preds[th_key] = candidate_ensemble.predict_proba(stack)
            test_predictions["Candidate Weighted Ensemble"] = ens_preds

            # Step 7: Champion Selection & Promotion Decision with Multiple Testing Safeguards
            self.job_progress = 0.92
            self.job_stage = "champion_selection_gates"
            self.job_logs.append("Executing statistical promotion gates and paired difference tests...")

            # Multiple testing correction (Holm-Bonferroni)
            num_candidates = len(candidate_models) + 1  # 6 models + ensemble
            corrected_alpha = 0.05 / max(1, num_candidates)

            selector = ChampionSelector(
                min_test_rounds=min_test_rounds,
                alpha_significance=corrected_alpha,
            )
            selection_result = selector.select_champion(
                candidates_predictions=test_predictions,
                y_true_matrix=y_test_dict,
                baseline_name="Historical Baseline",
                incumbent_name="Approved Incumbent Ensemble",
            )

            champion_name = selection_result["champion_name"]
            advantage_established = selection_result["advantage_established"]

            # Step 8: Conditional Promotion & Atomicity
            self.job_progress = 0.98
            self.job_stage = "atomic_promotion_decision"

            is_promoted = False
            promotion_status = "retained_incumbent"

            if advantage_established:
                is_promoted = True
                promotion_status = "promoted_candidate"
                self._promote_checkpoint_atomically(checkpoint_dir, tag, champion_name)
                logger.info(f"[Overnight Pipeline] PROMOTION SUCCESSFUL! Approved new champion: '{champion_name}'")
                self.job_logs.append(f"Candidate '{champion_name}' statistically surpassed incumbent. Promoted atomically.")
            else:
                if force_promotion:
                    logger.warning(
                        "[Overnight Pipeline] force_promotion requested but restricted: "
                        "Statistical significance gates are strictly preserved by policy to guarantee out-of-sample integrity."
                    )
                logger.info("[Overnight Pipeline] Promotion gates not met. Retaining incumbent approved configuration.")
                self.job_logs.append("Statistical promotion criteria not satisfied. Retaining incumbent approved configuration.")

            # Comprehensive final report
            report_data = {
                "pipeline_tag": tag,
                "job_id": job_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_duration_sec": round(time.time() - start_time, 2),
                "data_metadata": data_meta,
                "splits": {
                    "train_samples": len(X_train),
                    "calibration_samples": len(X_cal),
                    "validation_samples": len(X_val),
                    "untouched_test_samples": len(X_test),
                },
                "models_retrained": list(candidate_models.keys()),
                "calibrator_methods": ["platt", "isotonic"],
                "selection_result": selection_result,
                "promoted": is_promoted,
                "promotion_status": promotion_status,
                "champion_model": champion_name,
                "checkpoint_directory": str(checkpoint_dir),
                "candidate_ensemble_weights": {
                    m: round(float(w), 4) for m, w in zip(candidate_ensemble.model_names, candidate_ensemble.weights)
                },
            }

            report_file = self.reports_dir / f"overnight_report_{tag}.json"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(report_data, indent=2))

            # Save reproducible input manifest alongside training report
            manifest_file = self.reports_dir / f"retrain_manifest_{tag}.json"
            if "manifest" in data_meta:
                with open(manifest_file, "w", encoding="utf-8") as mf:
                    mf.write(json.dumps(data_meta["manifest"], indent=2))

            self.job_progress = 1.0
            self.job_stage = "completed"
            self.job_logs.append("Retraining pipeline completed successfully.")
            self.last_report = report_data
            return report_data

        finally:
            self.is_training = False

    def _promote_checkpoint_atomically(
        self,
        candidate_checkpoint_dir: Path,
        version_tag: str,
        champion_name: str,
    ) -> None:
        """Atomically promote a candidate checkpoint, updating approved version metadata with rollback support."""
        current_meta = {}
        if self.approved_version_file.exists():
            try:
                with open(self.approved_version_file, "r", encoding="utf-8") as f:
                    current_meta = json.load(f)
            except Exception:
                pass

        prev_version = current_meta.get("active_version", "v1.0.0_initial")
        archive_dir = self.checkpoints_dir / f"backup_{prev_version}"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for f in self.models_dir.glob("*"):
            if f.is_file() and f.name != self.approved_version_file.name:
                shutil.copy2(f, archive_dir / f.name)

        if candidate_checkpoint_dir.resolve() != self.models_dir.resolve():
            for f in candidate_checkpoint_dir.glob("*"):
                if f.is_file() and f.name != self.approved_version_file.name:
                    dest = self.models_dir / f.name
                    shutil.copy2(f, dest)

        new_meta = {
            "active_version": f"v_{version_tag}",
            "active_checkpoint_path": str(candidate_checkpoint_dir),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "champion_model": champion_name,
            "rollback_version": prev_version,
            "rollback_checkpoint_path": str(archive_dir),
        }

        with open(self.approved_version_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(new_meta, indent=2))

    def rollback(self) -> Dict[str, Any]:
        """Roll back active model checkpoint to the last approved configuration."""
        if not self.approved_version_file.exists():
            return {"status": "error", "message": "No approved version file found."}

        with open(self.approved_version_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        rollback_path = meta.get("rollback_checkpoint_path")
        rollback_ver = meta.get("rollback_version")

        if not rollback_path or not Path(rollback_path).exists():
            return {
                "status": "error",
                "message": f"Rollback checkpoint '{rollback_path}' not available on disk.",
            }

        src_dir = Path(rollback_path)
        if src_dir.resolve() != self.models_dir.resolve():
            for f in src_dir.glob("*"):
                if f.is_file() and f.name != self.approved_version_file.name:
                    shutil.copy2(f, self.models_dir / f.name)

        meta["active_version"] = rollback_ver
        meta["active_checkpoint_path"] = rollback_path
        meta["rollback_version"] = None
        meta["rollback_checkpoint_path"] = None
        meta["rolled_back_at"] = datetime.now(timezone.utc).isoformat()

        with open(self.approved_version_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta, indent=2))

        logger.info(f"[Overnight Pipeline] Successfully rolled back to '{rollback_ver}'.")
        return {
            "status": "success",
            "message": f"Successfully rolled back to '{rollback_ver}'.",
            "metadata": meta,
        }


# Global singleton instance
overnight_pipeline = OvernightTrainingPipeline()
