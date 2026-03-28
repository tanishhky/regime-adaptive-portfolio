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
        vols = []
        for tkr in tickers:
            gr = garch_results[tkr]
            rr = recovery_results[tkr]
            hl = rr.half_life if rr.mean_reverting else np.inf
            half_lives.append(hl)
            vols.append(gr.last_vol)

        vols_arr = np.array(vols)
        hl_arr = np.array(half_lives)

        # Fix: use tercile boundaries on conditional volatility instead of
        # the 25th-percentile VaR boundary.  The old VaR threshold placed
        # ~75% of assets in Basket C, leaving almost nothing to actively
        # manage.  Terciles guarantee a roughly 1/3 split.
        vol_33 = float(np.percentile(vols_arr, 33.3))
        vol_67 = float(np.percentile(vols_arr, 66.7))

        # Half-life boundary: median of finite half-lives
        finite_hl = hl_arr[np.isfinite(hl_arr)]
        hl_median = float(np.median(finite_hl)) if len(finite_hl) > 0 else 30.0

        assignments: dict[str, BasketAssignment] = {}
        for i, tkr in enumerate(tickers):
            vol = vols_arr[i]
            hl = hl_arr[i]

            if vol > vol_67:            # Top tercile volatility
                if hl < hl_median:
                    basket = "A"        # Tactical: high vol, fast recovery
                else:
                    basket = "B"        # Avoid: high vol, slow recovery
            elif vol > vol_33:          # Middle tercile volatility
                if hl < hl_median:
                    basket = "A"        # Tactical: medium vol, fast recovery
                else:
                    basket = "C"        # Core: medium vol, slow recovery
            else:                       # Bottom tercile volatility
                basket = "C"            # Core: low vol (hold)

            # Store VaR for backward compatibility
            var99 = float(garch_results[tkr].conditional_var99.iloc[-1])
            assignments[tkr] = BasketAssignment(
                ticker=tkr,
                basket=basket,
                half_life=float(hl),
                cond_var99=var99,
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
