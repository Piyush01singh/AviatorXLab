# Offline Historical Strategy Backtest Results (Target: 2.0x)

> **Strict Safety Rule**: Offline historical research simulation only. Live betting is never implemented.

| strategy                 | target | total_rounds | total_bets | win_rate | net_profit | final_balance | roi_pct | max_drawdown | max_drawdown_pct | profit_factor |
| ------------------------ | ------ | ------------ | ---------- | -------- | ---------- | ------------- | ------- | ------------ | ---------------- | ------------- |
| Always Enter (Baseline)  | 2.0x   | 1493         | 1493       | 0.4682   | -950.0     | 50.0          | -6.36   | 1100.0       | 99.1             | 0.88          |
| Never Enter (Baseline)   | 2.0x   | 1493         | 0          | 0.0      | 0.0        | 1000.0        | 0.0     | 0.0          | 0.0              | 0.0           |
| Threshold P >= 0.55      | 2.0x   | 1493         | 376        | 0.9255   | 3200.0     | 4200.0        | 85.11   | 30.0         | 0.85             | 12.429        |
| Threshold P >= 0.60      | 2.0x   | 1493         | 376        | 0.9255   | 3200.0     | 4200.0        | 85.11   | 30.0         | 0.85             | 12.429        |
| Positive EV (EV >= 0.02) | 2.0x   | 1493         | 376        | 0.9255   | 3200.0     | 4200.0        | 85.11   | 30.0         | 0.85             | 12.429        |
