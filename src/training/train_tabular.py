"""Training pipeline for tabular models (Baselines, Logistic Regression, Random Forest, XGBoost)."""

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.data.cleaner import DataCleaner
from src.data.loader import generate_synthetic_crash_data, load_csv
from src.data.splitter import chronological_split
from src.evaluation.calibration import compute_calibration_metrics
from src.evaluation.classification import compute_classification_metrics
from src.evaluation.regression import compute_regression_metrics
from src.features.builder import THRESHOLD_COLUMN_MAP, FeatureBuilder
from src.models.baseline import (
    HistoricalProbabilityBaseline,
    MajorityClassBaseline,
    MedianRegressionBaseline,
    RandomUniformBenchmark,
    RollingHistoricalProbabilityBaseline,
    RollingMedianRegressionBaseline,
)
from src.models.logistic import LogisticThresholdClassifier
from src.models.mlp import MLPThresholdClassifier
from src.models.random_forest import RandomForestThresholdClassifier
from src.models.xgboost_model import XGBoostMultiplierRegressor, XGBoostThresholdClassifier
from src.training.experiment import ExperimentRecord, ExperimentTracker
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.reproducibility import set_seed

logger = get_logger("train_tabular")


class TabularTrainer:
    """Orchestrates tabular training for baselines and classical ML models."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.seed = self.config["project"]["seed"]
        set_seed(self.seed)

        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.predictions_dir = Path(self.config["paths"]["results_predictions_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = ExperimentTracker(self.config["paths"]["results_metrics_dir"])
        self.builder = FeatureBuilder(
            lags=self.config["features"]["lag_rounds"],
            rolling_windows=self.config["features"]["rolling_windows"],
            thresholds=self.config["targets"]["thresholds"],
            include_temporal=self.config["features"]["include_temporal"],
        )

    def prepare_data(self, data_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Loads or generates raw data, cleans it, and constructs feature/target matrices."""
        file_path = data_file or self.config["data"]["default_file"]
        path = Path(file_path)

        if not path.is_file():
            logger.warning(f"Data file {file_path} not found. Generating initial synthetic verification dataset...")
            generate_synthetic_crash_data(num_rounds=10000, seed=self.seed, output_path=file_path)

        raw_df = load_csv(file_path)
        cleaner = DataCleaner()
        clean_df, _ = cleaner.clean(raw_df)

        X, y_thresh, y_reg, _ = self.builder.build(clean_df)
        return X, y_thresh, y_reg

    def run_baselines(self, X: pd.DataFrame, y_thresh: pd.DataFrame, y_reg: pd.Series) -> List[Dict[str, Any]]:
        """Train and evaluate all baseline models on the validation split."""
        split_ratios = self.config["data"]["split_ratios"]
        split_X = chronological_split(X, **split_ratios)
        split_y_th = chronological_split(y_thresh, **split_ratios)
        split_y_reg = chronological_split(y_reg.to_frame(), **split_ratios)

        val_results = []
        val_preds_df = pd.DataFrame(index=split_X.val.index)

        # 1. Historical & Majority baselines per threshold
        for th in self.config["targets"]["thresholds"]:
            target_col = THRESHOLD_COLUMN_MAP[th]
            y_tr = split_y_th.train[target_col]
            y_vl = split_y_th.val[target_col]

            # Majority class baseline
            maj_model = MajorityClassBaseline().fit(split_X.train, y_tr)
            maj_prob = maj_model.predict_proba(split_X.val)[:, 1]
            maj_metrics = compute_classification_metrics(y_vl, maj_prob)
            val_results.append({
                "model": "Majority Baseline",
                "threshold": f"{th}x",
                **maj_metrics,
            })

            # Historical probability baseline
            hist_model = HistoricalProbabilityBaseline().fit(split_X.train, y_tr)
            hist_prob = hist_model.predict_proba(split_X.val)[:, 1]
            hist_metrics = compute_classification_metrics(y_vl, hist_prob)
            val_results.append({
                "model": "Historical Baseline",
                "threshold": f"{th}x",
                **hist_metrics,
            })
            val_preds_df[f"hist_prob_{th}x"] = hist_prob

            self.tracker.log(ExperimentRecord(
                experiment_id=f"exp_baseline_hist_{th}x_{int(time.time())}",
                model_name="Historical Baseline",
                task_type="threshold_classification",
                threshold=f"{th}x",
                dataset_version=self.config["data"]["dataset_version"],
                feature_version=self.config["features"]["feature_version"],
                train_range={"samples": len(split_X.train)},
                validation_range={"samples": len(split_X.val)},
                test_range=None,
                hyperparameters={"window": "all_train"},
                random_seed=self.seed,
                training_time_sec=0.001,
                inference_time_sec=0.001,
                metrics=hist_metrics,
            ))

            # Rolling historical probability baseline
            rolling_model = RollingHistoricalProbabilityBaseline(window=50).fit(split_X.train, y_tr)
            roll_prob = rolling_model.predict_proba(split_X.val, past_target_series=split_y_th.val[target_col])[:, 1]
            roll_metrics = compute_classification_metrics(y_vl, roll_prob)
            val_results.append({
                "model": "Rolling Prob Baseline",
                "threshold": f"{th}x",
                **roll_metrics,
            })
            val_preds_df[f"rolling_prob_{th}x"] = roll_prob

            self.tracker.log(ExperimentRecord(
                experiment_id=f"exp_baseline_rolling_{th}x_{int(time.time())}",
                model_name="Rolling Prob Baseline",
                task_type="threshold_classification",
                threshold=f"{th}x",
                dataset_version=self.config["data"]["dataset_version"],
                feature_version=self.config["features"]["feature_version"],
                train_range={"samples": len(split_X.train)},
                validation_range={"samples": len(split_X.val)},
                test_range=None,
                hyperparameters={"window": 50},
                random_seed=self.seed,
                training_time_sec=0.001,
                inference_time_sec=0.001,
                metrics=roll_metrics,
            ))

        # 2. Regression baseline: Median Multiplier
        med_model = MedianRegressionBaseline().fit(split_X.train, split_y_reg.train["multiplier"])
        med_preds = med_model.predict(split_X.val)
        med_metrics = compute_regression_metrics(split_y_reg.val["multiplier"], med_preds)
        val_results.append({
            "model": "Median Baseline (Reg)",
            "threshold": "all",
            **med_metrics,
        })
        val_preds_df["median_reg_pred"] = med_preds

        self.tracker.log(ExperimentRecord(
            experiment_id=f"exp_baseline_median_reg_{int(time.time())}",
            model_name="Median Baseline (Reg)",
            task_type="multiplier_regression",
            threshold=None,
            dataset_version=self.config["data"]["dataset_version"],
            feature_version=self.config["features"]["feature_version"],
            train_range={"samples": len(split_X.train)},
            validation_range={"samples": len(split_X.val)},
            test_range=None,
            hyperparameters={},
            random_seed=self.seed,
            training_time_sec=0.001,
            inference_time_sec=0.001,
            metrics=med_metrics,
        ))

        # Save predictions
        preds_file = self.predictions_dir / "val_predictions_baseline.csv"
        val_preds_df.to_csv(preds_file, index=False)
        logger.info(f"Saved baseline validation predictions to {preds_file}")

        return val_results

    def train_logistic(self, X: pd.DataFrame, y_thresh: pd.DataFrame) -> List[Dict[str, Any]]:
        """Train Logistic Regression classifiers across all threshold targets."""
        split_ratios = self.config["data"]["split_ratios"]
        split_X = chronological_split(X, **split_ratios)
        split_y_th = chronological_split(y_thresh, **split_ratios)

        cfg = self.config["models"]["logistic"]
        results = []
        val_preds_df = pd.DataFrame(index=split_X.val.index)

        for th in self.config["targets"]["thresholds"]:
            target_col = THRESHOLD_COLUMN_MAP[th]
            y_tr = split_y_th.train[target_col]
            y_vl = split_y_th.val[target_col]

            model = LogisticThresholdClassifier(
                C=cfg["C"],
                penalty=cfg["penalty"],
                solver=cfg["solver"],
                max_iter=cfg["max_iter"],
                class_weight=cfg["class_weight"],
                random_state=self.seed,
            )

            t0 = time.time()
            model.fit(split_X.train, y_tr)
            train_time = time.time() - t0

            t0 = time.time()
            probs = model.predict_proba(split_X.val)[:, 1]
            infer_time = time.time() - t0

            metrics = compute_classification_metrics(y_vl, probs)
            calib = compute_calibration_metrics(y_vl, probs)
            metrics["ece"] = calib["ece"]

            # Save model
            model_path = self.models_dir / f"logistic_thresh_{th}x.joblib"
            model.save(str(model_path))

            exp_record = ExperimentRecord(
                experiment_id=f"exp_logistic_{th}x_{int(time.time())}",
                model_name="Logistic Regression",
                task_type="threshold_classification",
                threshold=f"{th}x",
                dataset_version=self.config["data"]["dataset_version"],
                feature_version=self.config["features"]["feature_version"],
                train_range={"samples": len(split_X.train)},
                validation_range={"samples": len(split_X.val)},
                test_range=None,
                hyperparameters=cfg,
                random_seed=self.seed,
                training_time_sec=round(train_time, 4),
                inference_time_sec=round(infer_time, 4),
                metrics=metrics,
            )
            self.tracker.log(exp_record)

            results.append({
                "model": "Logistic Regression",
                "threshold": f"{th}x",
                **metrics,
            })
            val_preds_df[f"logistic_prob_{th}x"] = probs

        val_preds_df.to_csv(self.predictions_dir / "val_predictions_logistic.csv", index=False)
        return results

    def train_random_forest(self, X: pd.DataFrame, y_thresh: pd.DataFrame) -> List[Dict[str, Any]]:
        """Train Random Forest classifiers across all threshold targets."""
        split_ratios = self.config["data"]["split_ratios"]
        split_X = chronological_split(X, **split_ratios)
        split_y_th = chronological_split(y_thresh, **split_ratios)

        cfg = self.config["models"]["random_forest"]
        results = []
        val_preds_df = pd.DataFrame(index=split_X.val.index)

        for th in self.config["targets"]["thresholds"]:
            target_col = THRESHOLD_COLUMN_MAP[th]
            y_tr = split_y_th.train[target_col]
            y_vl = split_y_th.val[target_col]

            model = RandomForestThresholdClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                min_samples_leaf=cfg["min_samples_leaf"],
                n_jobs=cfg["n_jobs"],
                random_state=self.seed,
            )

            t0 = time.time()
            model.fit(split_X.train, y_tr)
            train_time = time.time() - t0

            t0 = time.time()
            probs = model.predict_proba(split_X.val)[:, 1]
            infer_time = time.time() - t0

            metrics = compute_classification_metrics(y_vl, probs)
            calib = compute_calibration_metrics(y_vl, probs)
            metrics["ece"] = calib["ece"]

            model_path = self.models_dir / f"rf_thresh_{th}x.joblib"
            model.save(str(model_path))

            exp_record = ExperimentRecord(
                experiment_id=f"exp_rf_{th}x_{int(time.time())}",
                model_name="Random Forest",
                task_type="threshold_classification",
                threshold=f"{th}x",
                dataset_version=self.config["data"]["dataset_version"],
                feature_version=self.config["features"]["feature_version"],
                train_range={"samples": len(split_X.train)},
                validation_range={"samples": len(split_X.val)},
                test_range=None,
                hyperparameters=cfg,
                random_seed=self.seed,
                training_time_sec=round(train_time, 4),
                inference_time_sec=round(infer_time, 4),
                metrics=metrics,
            )
            self.tracker.log(exp_record)

            results.append({
                "model": "Random Forest",
                "threshold": f"{th}x",
                **metrics,
            })
            val_preds_df[f"rf_prob_{th}x"] = probs

        val_preds_df.to_csv(self.predictions_dir / "val_predictions_random_forest.csv", index=False)
        return results

    def train_xgboost(self, X: pd.DataFrame, y_thresh: pd.DataFrame, y_reg: pd.Series) -> List[Dict[str, Any]]:
        """Train XGBoost threshold classifiers and continuous multiplier regressor."""
        split_ratios = self.config["data"]["split_ratios"]
        split_X = chronological_split(X, **split_ratios)
        split_y_th = chronological_split(y_thresh, **split_ratios)
        split_y_reg = chronological_split(y_reg.to_frame(), **split_ratios)

        cfg = self.config["models"]["xgboost"]
        results = []
        val_preds_df = pd.DataFrame(index=split_X.val.index)

        # 1. Threshold classification
        for th in self.config["targets"]["thresholds"]:
            target_col = THRESHOLD_COLUMN_MAP[th]
            y_tr = split_y_th.train[target_col]
            y_vl = split_y_th.val[target_col]

            model = XGBoostThresholdClassifier(
                n_estimators=cfg["n_estimators"],
                max_depth=cfg["max_depth"],
                learning_rate=cfg["learning_rate"],
                subsample=cfg["subsample"],
                colsample_bytree=cfg["colsample_bytree"],
                early_stopping_rounds=cfg["early_stopping_rounds"],
                eval_metric=cfg["eval_metric"],
                random_state=self.seed,
            )

            t0 = time.time()
            model.fit(split_X.train, y_tr, X_val=split_X.val, y_val=y_vl)
            train_time = time.time() - t0

            t0 = time.time()
            probs = model.predict_proba(split_X.val)[:, 1]
            infer_time = time.time() - t0

            metrics = compute_classification_metrics(y_vl, probs)
            calib = compute_calibration_metrics(y_vl, probs)
            metrics["ece"] = calib["ece"]

            model_path = self.models_dir / f"xgb_thresh_{th}x.joblib"
            model.save(str(model_path))

            exp_record = ExperimentRecord(
                experiment_id=f"exp_xgb_{th}x_{int(time.time())}",
                model_name="XGBoost",
                task_type="threshold_classification",
                threshold=f"{th}x",
                dataset_version=self.config["data"]["dataset_version"],
                feature_version=self.config["features"]["feature_version"],
                train_range={"samples": len(split_X.train)},
                validation_range={"samples": len(split_X.val)},
                test_range=None,
                hyperparameters=cfg,
                random_seed=self.seed,
                training_time_sec=round(train_time, 4),
                inference_time_sec=round(infer_time, 4),
                metrics=metrics,
            )
            self.tracker.log(exp_record)

            results.append({
                "model": "XGBoost",
                "threshold": f"{th}x",
                **metrics,
            })
            val_preds_df[f"xgb_prob_{th}x"] = probs

        # 2. Secondary regression experiment
        reg_model = XGBoostMultiplierRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            early_stopping_rounds=cfg["early_stopping_rounds"],
            eval_metric="rmse",
            random_state=self.seed,
        )
        reg_model.fit(
            split_X.train,
            split_y_reg.train["multiplier"],
            X_val=split_X.val,
            y_val=split_y_reg.val["multiplier"],
        )
        reg_preds = reg_model.predict(split_X.val)
        reg_metrics = compute_regression_metrics(split_y_reg.val["multiplier"], reg_preds)
        reg_model.save(str(self.models_dir / "xgb_regressor.joblib"))

        exp_record_reg = ExperimentRecord(
            experiment_id=f"exp_xgb_reg_{int(time.time())}",
            model_name="XGBoost Regressor",
            task_type="multiplier_regression",
            threshold=None,
            dataset_version=self.config["data"]["dataset_version"],
            feature_version=self.config["features"]["feature_version"],
            train_range={"samples": len(split_X.train)},
            validation_range={"samples": len(split_X.val)},
            test_range=None,
            hyperparameters=cfg,
            random_seed=self.seed,
            training_time_sec=0.05,
            inference_time_sec=0.005,
            metrics=reg_metrics,
        )
        self.tracker.log(exp_record_reg)

        results.append({
            "model": "XGBoost Regressor",
            "threshold": "all",
            **reg_metrics,
        })
        val_preds_df["xgb_reg_pred"] = reg_preds

        val_preds_df.to_csv(self.predictions_dir / "val_predictions_xgboost.csv", index=False)
        return results

    def train_mlp(self, X: pd.DataFrame, y_thresh: pd.DataFrame) -> List[Dict[str, Any]]:
        """Train multi-threshold PyTorch MLP classifier with early stopping."""
        split_ratios = self.config["data"]["split_ratios"]
        split_X = chronological_split(X, **split_ratios)
        split_y_th = chronological_split(y_thresh, **split_ratios)

        cfg = self.config["models"]["mlp"]
        model = MLPThresholdClassifier(
            hidden_dims=cfg.get("hidden_dims", [64, 32]),
            dropout=cfg.get("dropout", 0.2),
            learning_rate=cfg.get("learning_rate", 0.001),
            batch_size=cfg.get("batch_size", 64),
            epochs=cfg.get("epochs", 50),
            early_stopping_patience=cfg.get("early_stopping_patience", 10),
            device_preference=self.config["hardware"].get("device", "auto"),
            random_state=self.seed,
        )

        t0 = time.time()
        model.fit(
            split_X.train,
            split_y_th.train.values,
            X_val=split_X.val,
            y_val=split_y_th.val.values,
        )
        train_time = time.time() - t0

        t0 = time.time()
        val_probs_matrix = model.predict_proba(split_X.val)
        infer_time = time.time() - t0

        # Save model
        model_path = self.models_dir / "mlp_model.joblib"
        model.save(str(model_path))

        threshold_list = self.config["targets"]["thresholds"]
        results = []
        val_preds_df = pd.DataFrame(index=split_X.val.index)

        for idx, th in enumerate(threshold_list):
            y_vl_th = split_y_th.val[THRESHOLD_COLUMN_MAP[th]].values
            th_probs = val_probs_matrix[:, idx]

            metrics = compute_classification_metrics(y_vl_th, th_probs)
            calib = compute_calibration_metrics(y_vl_th, th_probs)
            metrics["ece"] = calib["ece"]

            results.append({
                "model": "MLP",
                "threshold": f"{th}x",
                **metrics,
            })
            val_preds_df[f"mlp_prob_{th}x"] = th_probs

            exp_record = ExperimentRecord(
                experiment_id=f"exp_mlp_{th}x_{int(time.time())}",
                model_name="MLP",
                task_type="threshold_classification",
                threshold=f"{th}x",
                dataset_version=self.config["data"]["dataset_version"],
                feature_version=self.config["features"]["feature_version"],
                train_range={"samples": len(split_X.train)},
                validation_range={"samples": len(split_X.val)},
                test_range=None,
                hyperparameters={
                    "hidden_dims": cfg.get("hidden_dims", [64, 32]),
                    "dropout": cfg.get("dropout", 0.2),
                    "learning_rate": cfg.get("learning_rate", 0.001),
                    "batch_size": cfg.get("batch_size", 64),
                    "epochs": cfg.get("epochs", 50),
                    "device": model.device,
                    "best_val_loss": round(model.best_val_loss, 5),
                },
                random_seed=self.seed,
                training_time_sec=round(train_time, 4),
                inference_time_sec=round(infer_time, 4),
                metrics=metrics,
            )
            self.tracker.log(exp_record)

        val_preds_df.to_csv(self.predictions_dir / "val_predictions_mlp.csv", index=False)
        return results
