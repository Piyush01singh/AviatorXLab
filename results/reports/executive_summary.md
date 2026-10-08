# Executive Summary: Aviator Crash Game Predictability & Information Audit

**Date:** 2026-10-08  
**Project:** Aviator Offline Machine Learning & Information Theory Benchmark  
**Deliverable File:** `results/reports/executive_summary.md`  

---

## 1. Key Findings at a Glance

| Research Dimension | Metric / Scope | Naive Baseline | Best ML Architecture | Scientific Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Total Real Empirical Corpus** | **435,824 rounds** (419,862 four-platform corpus + 15,962 from 22bet) | N/A | N/A | Longitudinal stability confirmed across 2 distinct collection campaigns (~8 months apart) |
| **Out-of-Sample ROC-AUC** | 15,124 held-out test rounds | **0.5000** | **0.5069** (Random Forest) | **Indistinguishable from random guessing** ($p > 0.05$) |
| **Out-of-Sample Brier Score**| 15,124 held-out test rounds | **0.2498** | **0.2499** (Random Forest) | Baseline achieves lowest probabilistic error |
| **Walk-Forward ROC-AUC** | 6 expanding folds, 35.9 days | **0.5000** | **0.5049** $\pm$ 0.004 | All 95% CIs encompass **0.5000** |
| **Cryptographic RTP** | Betika $N = 100,868$ | 97.00% (Theoretical) | **97.03%** (Empirical) | $\chi^2 = 2.145$ ($p = 0.8288$), consistent with memoryless process |
| **Post-Round Timing Leak** | Single unshifted interval | N/A | **0.7511 – 0.9268** (LEAK) | **Proves predictor scams exploit completion leakage** |
| **Pre-Round Timing (Clean)**| Shifted inter-round delay | N/A | **0.4989 – 0.5020** (CLEAN) | Clean signal collapses to chance |

---

## 2. Core Takeaways

1. **Zero Genuine Predictive Signal**:  
   Across 6 machine learning architectures (Logistic Regression, Random Forest, XGBoost, PyTorch MLP, LSTM, GRU) and 51 genuine pre-round features, no model outperforms the naive Historical Probability Baseline with statistical significance. Out-of-sample ROC-AUC values are tightly bounded between **0.498 and 0.507**, and walk-forward 95% confidence intervals encompass 0.5000.
2. **The "Predictor" Myth Demystified**:  
   Third-party predictor tools claiming high accuracy (AUC 0.70–0.90) exploit **post-round flight duration leakage**: scraped timestamps record when a round finishes, directly encoding plane flight duration. When timestamps are correctly shifted to measure pre-round intervals, the artificial predictive signal vanishes completely.
3. **Provably-Fair Cryptographic Wall**:  
   Spribe Aviator's SHA-512 architecture combines a pre-committed server seed with client seeds from the first 3 independent bettors. Pre-image resistance of SHA-256 prevents prospective prediction, and independent player seed injection prevents operator predetermination.
4. **Selective Positive-ROI Backtests Are Statistical Artifacts**:  
   Simulated threshold strategies ($P \ge 0.55$) placed only 41–42 bets out of 15,124 rounds (0.27% participation rate). Their empirical win rate confidence intervals span well below breakeven, and the apparent positive return decays to negative expectation over wider horizons. The unconditional game ROI is **-2.97%**, matching the 3% house edge.

---

## 3. Limitations & Future Work

- **Historical Seed Logging**: Available scrapers logged multipliers and timestamps; future work includes dedicated WebSocket client listeners to log raw pre-round SHA-256 server seed commitments and client seeds directly.
- **Cross-Game Validation**: Extending this verification pipeline to other provably-fair crash titles (e.g. Bustabit, Stake Crash).
- **Automated PRNG Testing**: Integrating NIST SP 800-22 statistical test suites for continuous monitoring of live casino streams.

---

## 4. Recommendations for Machine Learning Researchers

- **Enforce Causal Shifting**: Always verify that inter-round timestamps are shifted by $\ge 1$ round to avoid leaking round completion time.
- **Benchmark Against Marginal Frequency**: Evaluate every ML model against the naive historical base rate; never claim predictive signal based on uncalibrated classification accuracy.
- **Require Walk-Forward Temporal Cross-Validation**: Reject any edge claim that does not survive large-sample walk-forward cross-validation with statistical significance tests ($p < 0.05$).
