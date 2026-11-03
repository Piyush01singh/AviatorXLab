# Aviator ML Research

Research project for testing whether crash-game round outcomes carry exploitable predictive signal using historical data, feature engineering, forecast models, and backtesting.

## Project layout

- `config.yaml` — main training and feature configuration
- `main.py` — CLI for dataset prep, feature building, training, evaluation, and reporting
- `run_app.py` — launches the local dashboard at `http://127.0.0.1:8000`
- `src/` — data pipeline, feature generation, models, evaluation, training, and backtesting
- `data/` — raw, processed, live, and session artifacts
- `models/` — trained model checkpoints and calibration artifacts
- `results/` — metrics, plots, and reports
- `tests/` — verification and leakage checks

## Quick start

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt

python main.py prepare-data --file data/raw/synthetic_sample.csv
python main.py train --model all
python main.py walk-forward --target-col target_2x
python run_app.py --host 127.0.0.1 --port 8000
```

## Current model stack

The project is configured around threshold classifiers and sequence models, including:

- Logistic regression
- Random forest
- XGBoost
- MLP
- LSTM
- GRU
- Weighted ensemble

## Notes

- The project is research-oriented and offline by default.
- It includes validation, leakage audits, and report generation for empirical analysis.
- It does not include live betting automation or casino account integration.
