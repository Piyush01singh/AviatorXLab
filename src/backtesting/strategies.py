"""Hypothetical simulation strategy decision rules.

Strict Safety Constraint:
Offline simulation only. Never interacts with accounts or live wagering systems.
"""

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Abstract base class for simulation cash-out and entry decision rules."""

    @abstractmethod
    def should_enter(self, predicted_prob: float, target_multiplier: float) -> bool:
        """Decide whether to place a hypothetical stake on this round."""
        pass


class AlwaysEnterStrategy(BaseStrategy):
    """Baseline strategy: Enters every single round unconditionally."""

    def should_enter(self, predicted_prob: float, target_multiplier: float) -> bool:
        return True


class NeverEnterStrategy(BaseStrategy):
    """Baseline strategy: Never enters any round."""

    def should_enter(self, predicted_prob: float, target_multiplier: float) -> bool:
        return False


class ThresholdProbabilityStrategy(BaseStrategy):
    """Model-driven rule: Enters only when predicted probability exceeds a decision threshold."""

    def __init__(self, min_prob: float = 0.55):
        self.min_prob = min_prob

    def should_enter(self, predicted_prob: float, target_multiplier: float) -> bool:
        return float(predicted_prob) >= self.min_prob


class PositiveExpectedValueStrategy(BaseStrategy):
    """Expected value rule: Enters only if EV = (P * target_multiplier - 1.0) >= min_ev."""

    def __init__(self, min_ev: float = 0.02):
        self.min_ev = min_ev

    def should_enter(self, predicted_prob: float, target_multiplier: float) -> bool:
        ev = (predicted_prob * target_multiplier) - 1.0
        return ev >= self.min_ev
