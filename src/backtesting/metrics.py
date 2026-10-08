"""Financial and risk evaluation metrics for offline strategy backtesting."""

from typing import Any, Dict
import numpy as np
import pandas as pd


def compute_backtest_metrics(
    trades_df: pd.DataFrame,
    initial_bankroll: float = 1000.0,
) -> Dict[str, Any]:
    """Calculate key financial performance, risk, and drawdown metrics.
    
    Args:
        trades_df: DataFrame containing per-round simulation details:
                   ['entered', 'stake', 'pnl', 'balance']
        initial_bankroll: Starting simulated balance.
        
    Returns:
        Dictionary of performance metrics.
    """
    if trades_df.empty:
        return {
            "total_rounds": 0,
            "total_bets": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "final_balance": initial_bankroll,
            "roi_pct": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
        }

    total_rounds = len(trades_df)
    bets = trades_df[trades_df["entered"] == True]
    total_bets = len(bets)

    if total_bets == 0:
        return {
            "total_rounds": total_rounds,
            "total_bets": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "final_balance": round(initial_bankroll, 2),
            "roi_pct": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
        }

    wins = (bets["pnl"] > 0).sum()
    win_rate = float(wins / total_bets)

    total_staked = float(bets["stake"].sum())
    net_profit = float(bets["pnl"].sum())
    final_balance = initial_bankroll + net_profit
    roi_pct = float((net_profit / total_staked) * 100.0) if total_staked > 0 else 0.0

    # Drawdown calculation
    balance_curve = trades_df["balance"].to_numpy()
    cum_max = np.maximum.accumulate(balance_curve)
    drawdowns = cum_max - balance_curve
    max_drawdown = float(np.max(drawdowns))

    dd_pct = np.where(cum_max > 0, drawdowns / cum_max, 0.0)
    max_drawdown_pct = float(np.max(dd_pct) * 100.0)

    gross_wins = float(bets[bets["pnl"] > 0]["pnl"].sum())
    gross_losses = float(abs(bets[bets["pnl"] < 0]["pnl"].sum()))
    profit_factor = round(gross_wins / gross_losses, 3) if gross_losses > 0 else float("inf")

    return {
        "total_rounds": total_rounds,
        "total_bets": total_bets,
        "win_rate": round(win_rate, 4),
        "net_profit": round(net_profit, 2),
        "final_balance": round(final_balance, 2),
        "roi_pct": round(roi_pct, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "profit_factor": profit_factor,
    }
