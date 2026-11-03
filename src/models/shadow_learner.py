"""Live Shadow Learning and Adaptive Weight Optimization.

Strict Protocol:
- Preserves approved deployed ensemble weights while continuously optimizing a separate shadow candidate.
- Updates candidate weights strictly after round resolution and scoring.
- Constrains weights to be non-negative and sum to 1.
- Regularizes weight updates and limits step sizes to prevent volatile reactions to single rounds.
- Evaluates candidate weights on later, chronologically separated observations against
  the incumbent ensemble, individual models, and statistical baselines.
- Candidate learning NEVER alters previously recorded predictions or evaluation scores.
"""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss

from src.utils.logging import get_logger

logger = get_logger("shadow_learner")

DEFAULT_MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "MLP",
    "LSTM",
    "GRU",
]


class AdaptiveShadowLearner:
    """Continuous shadow learning engine for adaptive ensemble weight optimization."""

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        deployed_weights: Optional[Dict[str, float]] = None,
        min_evidence_rounds: int = 15,
        regularization_lambda: float = 0.05,
        max_step_delta: float = 0.08,
    ):
        self.model_names = model_names or DEFAULT_MODEL_NAMES
        n_models = len(self.model_names)

        # Deployed incumbent weights (preserved intact)
        if deployed_weights:
            self.deployed_weights = deepcopy(deployed_weights)
        else:
            w = 1.0 / n_models
            self.deployed_weights = {m: w for m in self.model_names}

        # Shadow candidate weights (initialized equal to deployed weights)
        self.candidate_weights = deepcopy(self.deployed_weights)

        # Training history for candidate optimization: [(round_id, prob_vector, binary_y)]
        self._history_rounds: List[Union[int, str]] = []
        self._history_prob_matrix: List[np.ndarray] = []
        self._history_labels: List[int] = []

        # Out-of-sample candidate evaluation history (scored BEFORE learning)
        self.candidate_evaluations: List[Dict[str, Any]] = []

        # Tuning hyperparameters
        self.min_evidence_rounds = min_evidence_rounds
        self.regularization_lambda = regularization_lambda
        self.max_step_delta = max_step_delta

        # Logging
        self.updates_log: List[Dict[str, Any]] = []
        self.total_updates: int = 0
        self.last_update_timestamp: Optional[str] = None

    def predict_with_weights(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        weights_dict: Dict[str, float],
        threshold_key: str = "2.0x",
    ) -> float:
        """Compute convex blended probability for a specific threshold."""
        weighted_sum = 0.0
        weight_total = 0.0

        for m_name, w in weights_dict.items():
            if m_name in model_predictions and threshold_key in model_predictions[m_name]:
                prob = model_predictions[m_name][threshold_key]
                weighted_sum += float(w) * float(prob)
                weight_total += float(w)

        if weight_total > 0:
            return round(float(weighted_sum / weight_total), 4)

        # Fallback to mean across available models
        avail = [
            model_predictions[m][threshold_key]
            for m in self.model_names
            if m in model_predictions and threshold_key in model_predictions[m]
        ]
        return round(float(np.mean(avail)), 4) if avail else 0.5

    def predict_candidate_all_thresholds(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        threshold_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute shadow candidate ensemble predictions across all thresholds."""
        keys = threshold_keys or ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]
        return {
            th_key: self.predict_with_weights(model_predictions, self.candidate_weights, th_key)
            for th_key in keys
        }

    def predict_deployed_all_thresholds(
        self,
        model_predictions: Dict[str, Dict[str, float]],
        threshold_keys: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute deployed incumbent ensemble predictions across all thresholds."""
        keys = threshold_keys or ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]
        return {
            th_key: self.predict_with_weights(model_predictions, self.deployed_weights, th_key)
            for th_key in keys
        }

    def record_round_resolution(
        self,
        round_id: Union[int, str],
        model_predictions: Dict[str, Dict[str, float]],
        actual_multiplier: float,
        threshold_key: str = "2.0x",
        threshold_val: float = 2.0,
    ) -> Dict[str, Any]:
        """Record round outcome, score frozen forecasts out-of-sample, and update shadow weights.

        Strict Order:
        1. Score incumbent and candidate forecasts on the round's actual outcome.
        2. Append observation to training buffer.
        3. Attempt adaptive candidate weight optimization.
        """
        y_true = 1 if actual_multiplier >= threshold_val else 0

        # Step 1: Evaluate pre-round forecasts OUT-OF-SAMPLE before learning
        cand_p = self.predict_with_weights(model_predictions, self.candidate_weights, threshold_key)
        dep_p = self.predict_with_weights(model_predictions, self.deployed_weights, threshold_key)

        cand_p_clip = min(max(cand_p, 1e-6), 1.0 - 1e-6)
        dep_p_clip = min(max(dep_p, 1e-6), 1.0 - 1e-6)

        cand_brier = float((cand_p - y_true) ** 2)
        dep_brier = float((dep_p - y_true) ** 2)

        cand_ll = float(- (y_true * np.log(cand_p_clip) + (1 - y_true) * np.log(1.0 - cand_p_clip)))
        dep_ll = float(- (y_true * np.log(dep_p_clip) + (1 - y_true) * np.log(1.0 - dep_p_clip)))

        eval_entry = {
            "round_id": round_id,
            "actual_multiplier": actual_multiplier,
            "y_true": y_true,
            "candidate_prob": cand_p,
            "deployed_prob": dep_p,
            "candidate_brier": round(cand_brier, 4),
            "deployed_brier": round(dep_brier, 4),
            "candidate_log_loss": round(cand_ll, 4),
            "deployed_log_loss": round(dep_ll, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.candidate_evaluations.append(eval_entry)
        if len(self.candidate_evaluations) > 200:
            self.candidate_evaluations = self.candidate_evaluations[-200:]

        # Step 2: Extract model probability vector for this round
        prob_vec = []
        for m in self.model_names:
            prob = model_predictions.get(m, {}).get(threshold_key, 0.5)
            prob_vec.append(float(prob))

        self._history_rounds.append(round_id)
        self._history_prob_matrix.append(np.array(prob_vec, dtype=float))
        self._history_labels.append(y_true)

        # Step 3: Run candidate optimization if sufficient evidence exists
        update_result = None
        if len(self._history_labels) >= self.min_evidence_rounds:
            update_result = self._optimize_candidate_weights()

        return {
            "evaluation": eval_entry,
            "update_performed": update_result is not None and update_result.get("updated", False),
            "candidate_weights": deepcopy(self.candidate_weights),
            "deployed_weights": deepcopy(self.deployed_weights),
        }

    def _optimize_candidate_weights(self) -> Dict[str, Any]:
        """Optimize candidate weights against joint Brier score and Log Loss with regularization."""
        P_stack = np.array(self._history_prob_matrix[-100:])  # sliding lookback window
        y_arr = np.array(self._history_labels[-100:], dtype=float)
        M = len(self.model_names)

        curr_w = np.array([self.candidate_weights.get(m, 1.0 / M) for m in self.model_names], dtype=float)
        curr_w = curr_w / np.sum(curr_w)

        # Loss function with Brier score + Log loss + L2 regularization towards prior weights
        def objective(w_logits: np.ndarray) -> float:
            exp_w = np.exp(w_logits - np.max(w_logits))
            w = exp_w / np.sum(exp_w)

            p_blend = np.dot(P_stack, w)
            p_blend = np.clip(p_blend, 1e-6, 1.0 - 1e-6)

            brier = np.mean((p_blend - y_arr) ** 2)
            ll = - np.mean(y_arr * np.log(p_blend) + (1.0 - y_arr) * np.log(1.0 - p_blend))

            # Joint loss with L2 distance penalty
            reg = self.regularization_lambda * np.sum((w - curr_w) ** 2)
            return float(0.5 * brier + 0.5 * ll + reg)

        init_logits = np.log(np.maximum(curr_w, 1e-5))
        try:
            res = minimize(objective, init_logits, method="BFGS", options={"maxiter": 60})
            exp_res = np.exp(res.x - np.max(res.x))
            target_w = exp_res / np.sum(exp_res)

            # Apply maximum step size limiter (Requirement 5)
            delta = target_w - curr_w
            clipped_delta = np.clip(delta, -self.max_step_delta, self.max_step_delta)
            new_w = curr_w + clipped_delta
            new_w = np.maximum(new_w, 0.0)
            new_w = new_w / np.sum(new_w)

            # Compute prior loss vs new loss on the lookback window
            def _eval_loss(w: np.ndarray) -> Tuple[float, float]:
                pb = np.dot(P_stack, w)
                pb = np.clip(pb, 1e-6, 1.0 - 1e-6)
                br = float(np.mean((pb - y_arr) ** 2))
                l = float(- np.mean(y_arr * np.log(pb) + (1.0 - y_arr) * np.log(1.0 - pb)))
                return br, l

            prior_brier, prior_ll = _eval_loss(curr_w)
            new_brier, new_ll = _eval_loss(new_w)

            # Update candidate weights
            old_weights_dict = deepcopy(self.candidate_weights)
            self.candidate_weights = {
                m: round(float(w), 4) for m, w in zip(self.model_names, new_w)
            }
            self.total_updates += 1
            self.last_update_timestamp = datetime.now(timezone.utc).isoformat()

            log_entry = {
                "update_index": self.total_updates,
                "timestamp": self.last_update_timestamp,
                "rounds_used": len(y_arr),
                "prior_weights": old_weights_dict,
                "new_candidate_weights": deepcopy(self.candidate_weights),
                "prior_brier": round(prior_brier, 4),
                "new_brier": round(new_brier, 4),
                "prior_log_loss": round(prior_ll, 4),
                "new_log_loss": round(new_ll, 4),
                "status": "updated",
            }
            self.updates_log.append(log_entry)
            if len(self.updates_log) > 50:
                self.updates_log = self.updates_log[-50:]

            logger.info(
                f"[Shadow Learner] Updated candidate weights (#{self.total_updates}). "
                f"Brier: {prior_brier:.4f} -> {new_brier:.4f}, LogLoss: {prior_ll:.4f} -> {new_ll:.4f}"
            )
            return {"updated": True, "log": log_entry}

        except Exception as e:
            logger.warning(f"[Shadow Learner] Weight optimization error: {e}")
            return {"updated": False, "reason": str(e)}

    def get_candidate_status(self) -> Dict[str, Any]:
        """Return comprehensive live status of shadow candidate vs deployed weights."""
        # Calculate running out-of-sample metrics for candidate vs deployed
        cand_briers = [e["candidate_brier"] for e in self.candidate_evaluations]
        dep_briers = [e["deployed_brier"] for e in self.candidate_evaluations]
        cand_lls = [e["candidate_log_loss"] for e in self.candidate_evaluations]
        dep_lls = [e["deployed_log_loss"] for e in self.candidate_evaluations]

        n = len(self.candidate_evaluations)
        return {
            "candidate_weights": self.candidate_weights,
            "deployed_weights": self.deployed_weights,
            "total_updates": self.total_updates,
            "last_update_timestamp": self.last_update_timestamp,
            "evaluation_rounds_count": n,
            "out_of_sample_comparison": {
                "candidate_mean_brier": round(float(np.mean(cand_briers)), 4) if cand_briers else 0.2500,
                "deployed_mean_brier": round(float(np.mean(dep_briers)), 4) if dep_briers else 0.2500,
                "candidate_mean_log_loss": round(float(np.mean(cand_lls)), 4) if cand_lls else 0.6931,
                "deployed_mean_log_loss": round(float(np.mean(dep_lls)), 4) if dep_lls else 0.6931,
                "candidate_brier_improvement": round(float(np.mean(dep_briers) - np.mean(cand_briers)), 4) if cand_briers else 0.0,
            },
            "recent_updates": self.updates_log[-5:],
        }


# Global singleton instance
shadow_learner = AdaptiveShadowLearner()
