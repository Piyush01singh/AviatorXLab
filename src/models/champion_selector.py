"""Best-Model Champion Selection and Monotonic Multi-Threshold Forecasting.

Strict Protocol:
- Evaluates all 6 trained ML models, statistical baselines, incumbent ensemble,
  and shadow adaptive ensembles on untouched out-of-sample validation data.
- Multi-threshold objective emphasizing Brier score, Log Loss, and calibration error.
- Strict threshold monotonicity enforcement:
    P(>=1.5x) >= P(>=2x) >= P(>=3x) >= P(>=5x) >= P(>=10x)
- Historical Probability Baseline fallback:
    If no model demonstrates statistically significant improvement over baseline,
    explicitly report no advantage and retain the approved baseline/incumbent.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss

from src.utils.logging import get_logger

logger = get_logger("champion_selector")

THRESHOLDS = [1.5, 2.0, 3.0, 5.0, 10.0]
THRESHOLD_KEYS = ["1.5x", "2.0x", "3.0x", "5.0x", "10.0x"]


def enforce_probability_monotonicity(
    prob_dict: Dict[str, float],
    threshold_keys: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Enforce strict monotonicity across threshold probabilities:

        P(>=1.5x) >= P(>=2.0x) >= P(>=3.0x) >= P(>=5.0x) >= P(>=10.0x)

    Uses Pool-Adjacent-Violators / backward-forward isotonic projection to ensure
    valid probability physics while preserving calibration bounds.
    """
    keys = threshold_keys or THRESHOLD_KEYS
    raw_vals = [float(prob_dict.get(k, 0.5)) for k in keys]
    n = len(raw_vals)

    # Monotonic decreasing projection: p[0] >= p[1] >= ... >= p[n-1]
    # Invert signs to apply standard monotonic increasing isotonic regression
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip")
    indices = np.arange(n)
    adjusted_vals = ir.fit_transform(indices, raw_vals)

    # Ensure bounds [0.0001, 0.9999]
    adjusted_vals = np.clip(adjusted_vals, 0.0001, 0.9999)

    return {k: round(float(v), 4) for k, v in zip(keys, adjusted_vals)}


class ChampionSelector:
    """Evaluates candidates across out-of-sample data and selects the approved prediction source."""

    def __init__(
        self,
        min_test_rounds: int = 50,
        brier_improvement_margin: float = 0.0001,
        log_loss_improvement_margin: float = 0.0001,
        alpha_significance: float = 0.05,
    ):
        self.min_test_rounds = min_test_rounds
        self.brier_margin = brier_improvement_margin
        self.log_loss_margin = log_loss_improvement_margin
        self.alpha = alpha_significance

    def evaluate_candidate(
        self,
        y_true_matrix: Dict[str, np.ndarray],
        predicted_probs: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """Compute multi-threshold evaluation metrics for a candidate prediction source."""
        metrics = {}
        briers = []
        lls = []

        for th_key in THRESHOLD_KEYS:
            if th_key in y_true_matrix and th_key in predicted_probs:
                y = np.asarray(y_true_matrix[th_key]).astype(int)
                p = np.asarray(predicted_probs[th_key]).astype(float)
                p = np.clip(p, 1e-6, 1.0 - 1e-6)

                br = float(brier_score_loss(y, p))
                ll = float(log_loss(y, p, labels=[0, 1]))
                metrics[f"brier_{th_key}"] = round(br, 5)
                metrics[f"log_loss_{th_key}"] = round(ll, 5)
                briers.append(br)
                lls.append(ll)

        metrics["mean_brier"] = round(float(np.mean(briers)), 5) if briers else 0.2500
        metrics["mean_log_loss"] = round(float(np.mean(lls)), 5) if lls else 0.6931
        return metrics

    def compare_against_baseline(
        self,
        candidate_losses: np.ndarray,
        baseline_losses: np.ndarray,
    ) -> Tuple[float, float, bool]:
        """Run paired difference test (Wilcoxon signed-rank / paired t-test) on loss vectors."""
        diffs = candidate_losses - baseline_losses
        n = len(diffs)
        if n < 10 or np.all(diffs == 0):
            return 0.0, 1.0, False

        # Paired t-test
        t_stat, p_val = stats.ttest_1samp(diffs, 0.0)
        # One-sided test: is candidate loss significantly lower than baseline? (diff < 0)
        p_one_sided = float(p_val / 2.0 if t_stat < 0 else 1.0 - (p_val / 2.0))
        is_significant = p_one_sided < self.alpha and float(np.mean(diffs)) < 0.0
        return float(t_stat), p_one_sided, is_significant

    def select_champion(
        self,
        candidates_predictions: Dict[str, Dict[str, np.ndarray]],
        y_true_matrix: Dict[str, np.ndarray],
        baseline_name: str = "Historical Probability Baseline",
        incumbent_name: str = "Approved Incumbent Ensemble",
    ) -> Dict[str, Any]:
        """Perform champion selection across all models, baselines, and ensembles."""
        test_size = len(next(iter(y_true_matrix.values())))
        evaluation_results = {}

        for cand_name, pred_dict in candidates_predictions.items():
            metrics = self.evaluate_candidate(y_true_matrix, pred_dict)
            evaluation_results[cand_name] = metrics

        baseline_metrics = evaluation_results.get(
            baseline_name,
            {"mean_brier": 0.2498, "mean_log_loss": 0.6928}
        )
        base_brier = baseline_metrics["mean_brier"]
        base_ll = baseline_metrics["mean_log_loss"]

        # Check sample size requirement
        if test_size < self.min_test_rounds:
            logger.info(
                f"[Champion Selector] Test size {test_size} < {self.min_test_rounds}. "
                "Evidence insufficient; retaining baseline/incumbent fallback."
            )
            return {
                "champion_name": incumbent_name if incumbent_name in evaluation_results else baseline_name,
                "is_fallback": True,
                "advantage_established": False,
                "reason": f"Sample size {test_size} < minimum required ({self.min_test_rounds}). Retained approved baseline fallback.",
                "evaluations": evaluation_results,
            }

        # Identify candidate with lowest mean Brier score
        best_cand = None
        best_brier = float("inf")

        for cand_name, res in evaluation_results.items():
            if cand_name == baseline_name:
                continue
            b = res["mean_brier"]
            if b < best_brier:
                best_brier = b
                best_cand = cand_name

        # Validate whether best candidate statistically beats the baseline
        advantage_established = False
        rejection_reasons = []

        if best_cand is not None:
            cand_res = evaluation_results[best_cand]
            cand_brier = cand_res["mean_brier"]
            cand_ll = cand_res["mean_log_loss"]

            if cand_brier >= base_brier - self.brier_margin:
                rejection_reasons.append(
                    f"Candidate Brier score ({cand_brier:.5f}) did not exceed baseline ({base_brier:.5f}) by required margin ({self.brier_margin})."
                )

            if cand_ll >= base_ll - self.log_loss_margin:
                rejection_reasons.append(
                    f"Candidate Log Loss ({cand_ll:.5f}) did not beat baseline ({base_ll:.5f}) by required margin ({self.log_loss_margin})."
                )

            # Compute sample-wise loss vectors for paired statistical significance test
            cand_sample_losses = []
            base_sample_losses = []
            for th_key in THRESHOLD_KEYS:
                if th_key in y_true_matrix and th_key in candidates_predictions.get(best_cand, {}):
                    y_arr = np.asarray(y_true_matrix[th_key]).astype(int)
                    p_cand = np.asarray(candidates_predictions[best_cand][th_key]).astype(float)
                    p_base = np.asarray(candidates_predictions.get(baseline_name, {}).get(th_key, p_cand)).astype(float)
                    cand_sample_losses.append((p_cand - y_arr) ** 2)
                    base_sample_losses.append((p_base - y_arr) ** 2)

            if cand_sample_losses and base_sample_losses:
                cand_loss_vec = np.mean(cand_sample_losses, axis=0)
                base_loss_vec = np.mean(base_sample_losses, axis=0)
                t_stat, p_val, is_sig = self.compare_against_baseline(cand_loss_vec, base_loss_vec)
                cand_res["paired_test_p_value"] = round(p_val, 5)
                cand_res["paired_test_t_stat"] = round(t_stat, 4)
                cand_res["is_statistically_significant"] = is_sig
                if not is_sig:
                    rejection_reasons.append(
                        f"Candidate paired loss difference not statistically significant (p={p_val:.4f} >= alpha={self.alpha:.4f}, t={t_stat:.3f})."
                    )

            if not rejection_reasons:
                advantage_established = True

        if advantage_established and best_cand is not None:
            logger.info(f"[Champion Selector] Promoted NEW Champion: '{best_cand}' (Brier: {best_brier:.5f} vs Base: {base_brier:.5f})")
            return {
                "champion_name": best_cand,
                "is_fallback": False,
                "advantage_established": True,
                "reason": f"Established validated out-of-sample loss reduction over baseline: Brier={best_brier:.5f} vs {base_brier:.5f}",
                "evaluations": evaluation_results,
            }
        else:
            # Baseline fallback rule (Requirement 7 & Executive Summary)
            fallback_source = incumbent_name if incumbent_name in evaluation_results else baseline_name
            reason_msg = (
                f"No ML candidate established a reliable out-of-sample advantage over {baseline_name}. "
                + (" ".join(rejection_reasons) if rejection_reasons else "Loss centered on random guessing.")
                + f" Retaining {fallback_source}."
            )
            logger.info(f"[Champion Selector] {reason_msg}")
            return {
                "champion_name": fallback_source,
                "is_fallback": True,
                "advantage_established": False,
                "reason": reason_msg,
                "evaluations": evaluation_results,
            }
