# Probability Calibration Benchmark (Out-of-Sample Test Set)

| model               | calibration         | brier_score | log_loss | ece     | mce     |
| ------------------- | ------------------- | ----------- | -------- | ------- | ------- |
| Logistic Regression | None (Raw)          | 0.16769     | 0.49723  | 0.06806 | 0.16793 |
| Logistic Regression | Platt (Logistic)    | 0.16782     | 0.49665  | 0.06987 | 0.15438 |
| Logistic Regression | Isotonic Regression | 0.1625      | 0.48687  | 0.0294  | 0.08385 |
| Random Forest       | None (Raw)          | 0.1834      | 0.55254  | 0.11737 | 0.22539 |
| Random Forest       | Platt (Logistic)    | 0.16758     | 0.50101  | 0.05984 | 0.19737 |
| Random Forest       | Isotonic Regression | 0.16652     | 0.4956   | 0.035   | 0.13751 |
| XGBoost             | None (Raw)          | 0.16154     | 0.46847  | 0.01752 | 0.14084 |
| XGBoost             | Platt (Logistic)    | 0.16164     | 0.46789  | 0.01759 | 0.04951 |
| XGBoost             | Isotonic Regression | 0.1625      | 0.48434  | 0.02138 | 0.25671 |
