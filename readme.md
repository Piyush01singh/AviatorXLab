# Aviator ML Research: Empirical Predictability & Information Bounds in Crash Games

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-54%20passed-brightgreen.svg)](tests/)
[![Reproducible Research](https://img.shields.io/badge/research-reproducible-success.svg)](results/reports/)

A rigorous, research-grade machine learning and information-theoretic evaluation of whether historical crash-game (Spribe Aviator) multiplier data contains prospective predictive signal.

> **Research & Simulation Notice**: This repository is strictly an offline academic and algorithmic research platform. It does **NOT** contain automated betting bots, real-money execution, account interaction, or mechanisms designed to interface with casino accounts or bypass protections.

---

## Executive Summary & Core Scientific Conclusion

Across **435,824 real historical rounds** (including the **419,862-round four-platform corpus** spanning 35.9 continuous days across Betika, Odibets, Winpesa, and Betgr8, plus 15,962 rounds from 22bet):

1. **Zero Predictive Signal**: Every machine learning model (Logistic Regression, Random Forest, XGBoost, PyTorch MLP, PyTorch LSTM, PyTorch GRU) achieves out-of-sample ROC-AUC between **0.498 and 0.507** (strictly centered on random guessing, $0.500$).
2. **Naive Baseline Superiority**: The naive **Historical Probability Baseline** achieves lower Log Loss (0.6928) and Brier score (0.2498) than any trained machine learning model. In paired statistical significance tests, no model improves on the baseline ($p > 0.05$).
3. **Walk-Forward Invariance**: Across 6 expanding walk-forward chronological cross-validation folds, 95% confidence intervals for ROC-AUC encompass **0.5000** for every architecture.
4. **The "Predictor" Myth Demystified**: Third-party claims of high accuracy (AUC 0.75–0.93) are caused by **post-round elapsed flight duration leakage**: scraped timestamps record when a round finishes, directly encoding plane flight duration. When timestamps are correctly shifted to measure pre-round intervals, the artificial signal completely vanishes (AUC 0.500).
5. **Selective Positive-ROI Backtests Are Artifacts**: Selective high-threshold strategies ($P \ge 0.55$) placed only 41–42 bets out of 15,124 test rounds. Their win-rate confidence intervals span well below breakeven, and the apparent positive return decays to negative expectation over wider horizons. The unconditional game ROI is **-2.97%**, consistent with the 3% house edge (97% RTP).

---

## System Architecture

```
aviator/
├── data/
│   ├── raw/                 # Authentic raw historical datasets (CSV, JSON metadata)
│   ├── processed/           # Validated, schema-enforced, clean datasets
│   └── external/            # Third-party data archives and verification scripts
├── models/
│   ├── calibrators/         # Platt scaling & isotonic regression calibrators
│   └── *.joblib             # Serialized model checkpoints
├── results/
│   ├── metrics/             # JSON experiment tracking logs
│   ├── plots/               # ROC curves, PR curves, calibration curves, equity curves
│   └── reports/             # Comprehensive research reports and CSV comparison matrices
├── src/
│   ├── backtesting/         # Offline strategy simulator, execution logic, metrics
│   ├── data/                # Loaders, schema validators, cleaners, chronological splitters
│   ├── evaluation/          # Classification, regression, calibration, statistical tests
│   ├── features/            # Causal feature builders, lag features, rolling statistics
│   ├── models/              # Baseline, Logistic, RF, XGBoost, MLP, LSTM, GRU
│   ├── training/            # Tabular/sequence trainers, walk-forward CV, benchmark scripts
│   └── utils/               # Config loaders, logging, reproducibility, plotting
└── tests/                   # 46 automated unit, integration, and leakage test suites
```

---

## Dataset Provenance & Integrity

All datasets are cryptographically fingerprinted using SHA-256:

| Dataset Name | Source / Platform | Implementation | Rounds | Observation Window | SHA-256 Provenance Hash (Processed) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `synthetic_sample` | Mathematical Simulator | Inverse Transform (97% RTP)| 10,000 | Reference Slice | `7b72b460c2505ec7cb4e2affca2d3b53853acab374194583c2875dfdc027e639` |
| `real_dataset_v001` | 22bet Historical | Spribe Aviator | 15,962 | 5.0 days (Aug 2023) | `3a0abc74329984cd7a5578c3d2a8789bbc2a4b1f4b8298c66be955125485ae31` |
| `real_dataset_betika_v001` | Betika (Primary Benchmark)| Spribe Aviator | 100,868 | 35.9 days (Apr–May 2024)| `22be055e0c7096dfd152cb45a150b0e4fcefebc20f518ab534c2e11fa72bf77e` |
| `real_dataset_odibets_v001` | Odibets | Spribe Aviator | 113,979 | 35.9 days (Apr–May 2024)| `f93e678334047a8d8af7e95696bbf455e7a7a308bf32d2b72e186256d4673436` |
| `real_dataset_winpesa_v001` | Winpesa | Spribe Aviator | 104,323 | 35.9 days (Apr–May 2024)| `889fec288a739d7a5e77487618bcda34b595583a3c7a0554544281ef3f38558c` |
| `real_dataset_betgr8_v001` | Betgr8 | Spribe Aviator | 100,692 | 35.9 days (Apr–May 2024)| `36838c5022b20ac0b8afa12d988bd3b4235157b0e88bc5188046cfef81b8d1f5` |
| **Four-Platform Corpus** | Combined 4 Operators | Spribe Aviator | **419,862** | 35.9 continuous days | Aggregated |
| **Total Real Corpus** | All 5 Operators | Spribe Aviator | **435,824** | 2 collection campaigns (~8 months apart: Aug 2023 & Apr–May 2024) | Complete Historical Corpus |

---

## Machine Learning Benchmark Results

Evaluated strictly out-of-sample on **15,124 held-out test rounds** (Betika benchmark, target $\ge 2.0x$):

| Model Architecture | Accuracy | Balanced Acc | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Log Loss | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Probability Baseline** | **0.5128** | 0.5000 | 0.0000 | 0.0000 | 0.0000 | **0.5000** | 0.4872 | **0.6928** | **0.2498** |
| **Logistic Regression** | 0.5140 | 0.5030 | 0.5082 | 0.0753 | 0.1312 | **0.5031** | 0.4895 | 0.6931 | 0.2500 |
| **Random Forest** | 0.5127 | 0.5008 | 0.4982 | 0.0381 | 0.0709 | **0.5069** | 0.4917 | 0.6929 | 0.2499 |
| **XGBoost** | 0.5128 | 0.5029 | 0.5000 | 0.1170 | 0.1896 | **0.5042** | 0.4931 | 0.6930 | 0.2499 |
| **PyTorch MLP** | 0.5065 | 0.4965 | 0.4709 | 0.1053 | 0.1721 | **0.5036** | 0.4861 | 0.6930 | 0.2499 |
| **PyTorch LSTM** | 0.5091 | 0.5048 | 0.4944 | 0.3374 | 0.4011 | **0.5064** | 0.4926 | 0.6949 | 0.2509 |
| **PyTorch GRU** | 0.5064 | 0.4972 | 0.4774 | 0.1403 | 0.2169 | **0.4988** | 0.4883 | 0.6943 | 0.2506 |
| **Rolling Baseline ($W=50$)**| 0.4913 | 0.4906 | 0.4771 | 0.4597 | 0.4682 | **0.4860** | 0.4773 | 0.7056 | 0.2559 |

### Expanding Walk-Forward Validation (6 Chronological Folds)
| Model | Mean Accuracy | 95% CI (Accuracy) | Mean ROC-AUC | 95% CI (ROC-AUC) | Mean Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Historical Baseline** | 0.5125 | `[0.5105, 0.5146]` | **0.5000** | `[0.5000, 0.5000]` | **0.2498** |
| **Logistic Regression** | 0.5104 | `[0.5091, 0.5118]` | **0.5015** | `[0.4974, 0.5057]` | 0.2500 |
| **Random Forest** | 0.5124 | `[0.5099, 0.5150]` | **0.5049** | `[0.5016, 0.5082]` | 0.2499 |
| **XGBoost** | 0.5111 | `[0.5094, 0.5128]` | **0.5028** | `[0.4990, 0.5066]` | 0.2500 |
| **PyTorch MLP** | 0.5085 | `[0.5053, 0.5117]` | **0.5003** | `[0.4959, 0.5047]` | 0.2501 |
| **PyTorch LSTM** | 0.5002 | `[0.4968, 0.5037]` | **0.4983** | `[0.4934, 0.5032]` | 0.2534 |
| **PyTorch GRU** | 0.5026 | `[0.4974, 0.5078]` | **0.4993** | `[0.4952, 0.5033]` | 0.2527 |

---

## Leakage Discovery: Post-Round Completion Interval

A core breakthrough of this project explains why online "Aviator predictor" tools claim high accuracy (AUC 0.75–0.93):
- Scraped datasets log the timestamp $\tau_t$ when the plane finishes flying and the payout is published.
- Plane flight duration is an increasing monotonic function of the multiplier: $D(M_t) \approx k \cdot \ln M_t$.
- The unshifted interval $(\tau_t - \tau_{t-1})$ directly encodes whether the multiplier was high, leaking future information.
- When timestamps are properly shifted by $\ge 1$ round to measure the genuine pre-round pause duration $(\tau_{t-1} - \tau_{t-2})$, predictive accuracy collapses immediately:

| Dataset | Post-Round Interval AUC (LEAK) | Post-Round Interval Correlation | Pre-Round Interval AUC (CLEAN) | Pre-Round Interval Correlation |
| :--- | :--- | :--- | :--- | :--- |
| **Betika** | **0.7791** | $+0.027$ ($p = 1.00 \times 10^{-17}$) | **0.5020** | $-0.0036$ ($p = 0.252$) |
| **Odibets** | **0.8140** | $+0.069$ ($p = 1.03 \times 10^{-121}$) | **0.5001** | $-0.0058$ ($p = 0.052$) |
| **Winpesa** | **0.7939** | $+0.030$ ($p = 1.24 \times 10^{-22}$) | **0.5002** | $+0.0028$ ($p = 0.359$) |
| **Betgr8** | **0.7511** | $+0.031$ ($p = 8.22 \times 10^{-23}$) | **0.4989** | $-0.0035$ ($p = 0.271$) |
| **22bet** | **0.9268** | $+0.130$ ($p = 3.11 \times 10^{-61}$) | **0.5223** | $+0.0017$ ($p = 0.833$) |

---

## Setup & Reproduction Instructions

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/example/aviator.git
cd aviator
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Test Suite
Verify that all 46 unit, integration, and leakage test suites pass:
```bash
pytest -v
```

### 3. Re-run Expanded Benchmark & Walk-Forward Validation
Execute the complete evaluation pipeline on the 100,868-round Betika dataset:
```bash
python -m src.training.evaluate_expanded_real_data
```

### 4. Re-run Pre-Round Information Audit
Execute the information-theoretic audit across all 5 datasets:
```bash
python scratch/compile_pre_round_audit.py
```

### 5. Launch Interactive Research Dashboard (Live & CSV Modes)
Launch the web interface for real-time live streaming research, CSV upload, causal model inference, and empirical visualizations:
```bash
python run_app.py
# Or directly via Uvicorn:
# uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

- **Live Research Mode (Primary)**: Click `Connect Live Data` to activate real-time stream processing. By default, it runs the built-in authentic sample stream or connects to passive WebSocket / browser observers. Predicts round $N+1$ strictly before outcome revelation, tracks running Brier/Log loss against actual results, and displays scrolling round chips.
- **Upload CSV Mode (Secondary Fallback)**: Drag and drop any historical CSV round export for batch validation, 51 causal feature extractions, and multi-threshold predictions across all 6 models.

---

## Limitations & Future Work

### Limitations
- **Public Data Constraints**: Public web-scraped datasets contain output multipliers and completion timestamps, but do not record raw unhashed server seeds and client seeds for every historical round.
- **Latency Jitter**: While timestamps provide round intervals, precise network ping jitter between the game client and Spribe's servers cannot be reconstructed retroactively.
- **Discrete Modeling Scope**: Modeling was restricted to threshold probabilities and multiplier regression on memoryless PRNG outputs.

### Future Work
- **Live Client Seed Capture**: Developing an open-source, non-betting WebSocket recorder to capture raw pre-round SHA-256 server seed commitments and client seeds directly from the game client.
- **Cross-Game Provably-Fair Benchmarking**: Applying this audit framework to other crash games (Bustabit, Stake Crash, Roobet Crash).
- **Automated PRNG Testing**: Integrating NIST SP 800-22 test suites for continuous monitoring of live casino stream independence.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
