"""
Execution model — transaction cost accounting and trade simulation.

Applies a flat per-trade cost in basis points (from config) to the
absolute dollar value of each trade.  No partial fills or market impact
(appropriate for highly-liquid sector ETFs).

References
----------
Configuration: ``config.TRANSACTION_COST_BPS``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class ExecutionLog:
    """Accumulated execution statistics."""

    total_cost: float = 0.0              # cumulative cost drag (in return units)
    total_turnover: float = 0.0          # cumulative turnover (|Δw|)
    n_trades: int = 0                    # number of individual trades
    daily_costs: list[float] = field(default_factory=list)
    daily_turnover: list[float] = field(default_factory=list)


class ExecutionModel:
    """Flat-cost execution simulator."""

    def __init__(self, cost_bps: int = config.TRANSACTION_COST_BPS) -> None:
        """
        Parameters
        ----------
        cost_bps : int
            Transaction cost in basis points per trade.
        """
        self.cost_bps = cost_bps
        self.cost_frac = cost_bps / 10_000.0
        self.log = ExecutionLog()

    def execute(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        """Simulate execution of a rebalance and compute costs.

        Parameters
        ----------
        old_weights : dict[str, float]
            Pre-trade weights.
        new_weights : dict[str, float]
            Target weights.

        Returns
        -------
        tuple[dict[str, float], float]
            Executed weights and total cost for this trade (in return units).
        """
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        turnover = 0.0
        n_trades = 0

        for tkr in all_tickers:
            old_w = old_weights.get(tkr, 0.0)
            new_w = new_weights.get(tkr, 0.0)
            delta = abs(new_w - old_w)
            if delta > 1e-8:
                turnover += delta
                n_trades += 1

        cost = turnover * self.cost_frac

        self.log.total_cost += cost
        self.log.total_turnover += turnover
        self.log.n_trades += n_trades
        self.log.daily_costs.append(cost)
        self.log.daily_turnover.append(turnover)

        return dict(new_weights), cost

    def reset(self) -> None:
        """Reset execution log."""
        self.log = ExecutionLog()
