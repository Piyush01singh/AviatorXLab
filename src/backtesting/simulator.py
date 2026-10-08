"""Offline historical backtesting simulator.

Evaluates hypothetical fixed stakes and cash-out policies against historical outcomes.
Strict Safety Rule: Offline simulation only. Never places live bets.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.backtesting.metrics import compute_backtest_metrics
from src.backtesting.strategies import (
    AlwaysEnterStrategy,
    BaseStrategy,
    NeverEnterStrategy,
    PositiveExpectedValueStrategy,
    ThresholdProbabilityStrategy,
)
from src.utils.logging import get_logger

logger = get_logger("backtest_simulator")


@dataclass
class SimulationResult:
    """Encapsulates backtesting logs and aggregate summary metrics."""
    strategy_name: str
    target_multiplier: float
    metrics: Dict[str, Any]
    trade_log: pd.DataFrame


class OfflineHistoricalSimulator:
    """Historical strategy simulator with zero lookahead bias."""

    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        fixed_stake: float = 10.0,
    ):
        self.initial_bankroll = initial_bankroll
        self.fixed_stake = fixed_stake

    def run_simulation(
        self,
        actual_multipliers: Union[pd.Series, np.ndarray],
        predicted_probs: Union[pd.Series, np.ndarray],
        target_multiplier: float,
        strategy: BaseStrategy,
        strategy_name: str = "Strategy",
    ) -> SimulationResult:
        """Simulate strategy across sequential historical rounds.
        
        Args:
            actual_multipliers: Ground truth multipliers for round t.
            predicted_probs: Predicted probability P(multiplier >= target) available BEFORE round t.
            target_multiplier: Predefined cash-out multiplier.
            strategy: Strategy decision instance.
            strategy_name: Descriptive name.
        """
        mult_arr = np.asarray(actual_multipliers, dtype=float)
        prob_arr = np.asarray(predicted_probs, dtype=float)
        n = len(mult_arr)

        if len(prob_arr) != n:
            raise ValueError(f"Length mismatch: {n} outcomes vs {len(prob_arr)} predictions.")

        entered_flags = []
        stakes = []
        pnls = []
        balance_curve = []
        current_balance = self.initial_bankroll

        for t in range(n):
            prob_t = float(prob_arr[t])
            actual_t = float(mult_arr[t])

            should_bet = strategy.should_enter(prob_t, target_multiplier)

            if should_bet and current_balance >= self.fixed_stake:
                entered_flags.append(True)
                stakes.append(self.fixed_stake)

                if actual_t >= target_multiplier:
                    # Won cashout
                    profit = self.fixed_stake * (target_multiplier - 1.0)
                else:
                    # Crashed before cashout
                    profit = -self.fixed_stake

                current_balance += profit
                pnls.append(profit)
            else:
                entered_flags.append(False)
                stakes.append(0.0)
                pnls.append(0.0)

            balance_curve.append(current_balance)

        trades_df = pd.DataFrame({
            "round_idx": np.arange(n),
            "actual_multiplier": mult_arr,
            "predicted_prob": prob_arr,
            "target_multiplier": target_multiplier,
            "entered": entered_flags,
            "stake": stakes,
            "pnl": pnls,
            "balance": balance_curve,
        })

        metrics = compute_backtest_metrics(trades_df, self.initial_bankroll)
        logger.info(
            f"Backtest complete [{strategy_name} @ {target_multiplier}x]: "
            f"Bets={metrics['total_bets']}, WinRate={metrics['win_rate']:.1%}, "
            f"ROI={metrics['roi_pct']}%, NetProfit={metrics['net_profit']}, MaxDD={metrics['max_drawdown_pct']}%"
        )

        return SimulationResult(
            strategy_name=strategy_name,
            target_multiplier=target_multiplier,
            metrics=metrics,
            trade_log=trades_df,
        )
