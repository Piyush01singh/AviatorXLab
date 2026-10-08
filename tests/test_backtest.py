"""Unit tests for offline backtesting simulator, strategies, and financial metrics."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import compute_backtest_metrics
from src.backtesting.simulator import OfflineHistoricalSimulator
from src.backtesting.strategies import (
    AlwaysEnterStrategy,
    NeverEnterStrategy,
    PositiveExpectedValueStrategy,
    ThresholdProbabilityStrategy,
)


def test_strategies_decision_rules():
    always = AlwaysEnterStrategy()
    never = NeverEnterStrategy()
    thresh = ThresholdProbabilityStrategy(min_prob=0.60)
    ev_strat = PositiveExpectedValueStrategy(min_ev=0.05)

    assert always.should_enter(0.1, 2.0) is True
    assert never.should_enter(0.9, 2.0) is False

    assert thresh.should_enter(0.65, 2.0) is True
    assert thresh.should_enter(0.55, 2.0) is False

    # EV = P * target - 1.0. For P=0.55, target=2.0 -> EV = 1.1 - 1 = 0.10 >= 0.05 -> True
    assert ev_strat.should_enter(0.55, 2.0) is True
    # For P=0.51, target=2.0 -> EV = 1.02 - 1 = 0.02 < 0.05 -> False
    assert ev_strat.should_enter(0.51, 2.0) is False


def test_compute_backtest_metrics():
    # 4 rounds: 2 wins, 2 losses at 2.0x
    trades = pd.DataFrame({
        "entered": [True, True, True, True],
        "stake": [10.0, 10.0, 10.0, 10.0],
        "pnl": [10.0, -10.0, 10.0, -10.0],
        "balance": [1010.0, 1000.0, 1010.0, 1000.0],
    })
    metrics = compute_backtest_metrics(trades, initial_bankroll=1000.0)
    assert metrics["total_rounds"] == 4
    assert metrics["total_bets"] == 4
    assert metrics["win_rate"] == 0.5
    assert metrics["net_profit"] == 0.0
    assert metrics["final_balance"] == 1000.0
    assert metrics["roi_pct"] == 0.0
    assert metrics["max_drawdown"] == 10.0


def test_offline_historical_simulator_simulation():
    actuals = np.array([2.5, 1.2, 3.1, 1.8, 5.0])
    probs = np.array([0.7, 0.4, 0.65, 0.45, 0.8])
    target = 2.0
    initial_bankroll = 1000.0
    stake = 10.0

    simulator = OfflineHistoricalSimulator(initial_bankroll=initial_bankroll, fixed_stake=stake)
    thresh_strat = ThresholdProbabilityStrategy(min_prob=0.50)

    res = simulator.run_simulation(
        actual_multipliers=actuals,
        predicted_probs=probs,
        target_multiplier=target,
        strategy=thresh_strat,
        strategy_name="ThresholdTest",
    )

    log = res.trade_log
    assert len(log) == 5
    # Rounds with prob >= 0.50 are idx 0 (prob 0.7), idx 2 (prob 0.65), idx 4 (prob 0.8)
    assert list(log["entered"]) == [True, False, True, False, True]
    # For entered rounds:
    # idx 0: actual 2.5 >= 2.0 -> win +10
    # idx 2: actual 3.1 >= 2.0 -> win +10
    # idx 4: actual 5.0 >= 2.0 -> win +10
    assert res.metrics["total_bets"] == 3
    assert res.metrics["win_rate"] == 1.0
    assert res.metrics["net_profit"] == 30.0
    assert res.metrics["final_balance"] == 1030.0
