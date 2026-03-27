"""
Basket classifier — assigns sector ETFs to tactical baskets.

Uses data-driven boundaries (median half-life, 75th percentile VaR) to
classify each ETF into one of three baskets at each rebalance date.

Basket A (Tactical) : fast recovery + high vol → liquidate on stress, re-enter
Basket B (Avoid)    : slow recovery + high vol → permanent underweight
Basket C (Core)     : low vol → hold through mild stress
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.characterization.volatility import GARCHResult
from src.characterization.recovery import RecoveryResult


@dataclass
class BasketAssignment:
    """Basket assignment for a single ticker at a rebalance date."""

    ticker: str
    basket: str          # "A", "B", or "C"
    half_life: float
    cond_var99: float    # last conditional VaR(99%)
    cond_vol: float      # last annualised conditional vol


class BasketClassifier:
    """Data-driven basket classifier (no hardcoded boundaries)."""

    def assign(
        self,
        garch_results: dict[str, GARCHResult],
        recovery_results: dict[str, RecoveryResult],
    ) -> dict[str, BasketAssignment]:
        """Classify each ETF into a basket based on training-window statistics.

        All thresholds are computed from the cross-section of current
        GARCH and recovery estimates (fully data-driven).

        Parameters
        ----------
        garch_results : dict[str, GARCHResult]
            GARCH outputs keyed by ticker.
        recovery_results : dict[str, RecoveryResult]
            Recovery outputs keyed by ticker.

        Returns
        -------
        dict[str, BasketAssignment]
            Basket assignments keyed by ticker.
        """
        tickers = sorted(
            set(garch_results.keys()) & set(recovery_results.keys())
        )
        if not tickers:
            return {}

        # Collect metrics
        half_lives = []
        var99s = []
        vols = []
        for tkr in tickers:
            gr = garch_results[tkr]
            rr = recovery_results[tkr]
            hl = rr.half_life if rr.mean_reverting else np.inf
            half_lives.append(hl)
            var99s.append(float(gr.conditional_var99.iloc[-1]))
            vols.append(gr.last_vol)

        half_lives_arr = np.array(half_lives)
        var99s_arr = np.array(var99s)

        # Data-driven boundaries
        # Replace inf with a large number for median computation
        finite_hl = half_lives_arr[np.isfinite(half_lives_arr)]
        half_life_median = float(np.median(finite_hl)) if len(finite_hl) > 0 else 30.0
        # VaR is negative; 75th percentile of |VaR| = most negative quartile
        cvar_75 = float(np.percentile(var99s_arr, 25))  # 25th pctile (most negative)

        assignments: dict[str, BasketAssignment] = {}
        for i, tkr in enumerate(tickers):
            hl = half_lives_arr[i]
            v99 = var99s_arr[i]

            if v99 < cvar_75:  # High vol (more negative VaR)
                if hl < half_life_median:
                    basket = "A"  # Tactical: fast recovery + high vol
                else:
                    basket = "B"  # Avoid: slow recovery + high vol
            else:
                basket = "C"      # Core: low vol

            assignments[tkr] = BasketAssignment(
                ticker=tkr,
                basket=basket,
                half_life=float(hl),
                cond_var99=float(v99),
                cond_vol=vols[i],
            )

        return assignments

    @staticmethod
    def summary_df(
        assignments: dict[str, BasketAssignment],
    ) -> pd.DataFrame:
        """Convert assignments to a summary DataFrame."""
        rows = []
        for tkr, ba in assignments.items():
            rows.append({
                "Ticker": tkr,
                "Basket": ba.basket,
                "Half-Life": ba.half_life,
                "VaR(99%)": ba.cond_var99,
                "Ann. Vol": ba.cond_vol,
            })
        return pd.DataFrame(rows).set_index("Ticker").sort_values("Basket")
