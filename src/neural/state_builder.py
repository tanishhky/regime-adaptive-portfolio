"""
State vector construction for the LSTM-PPO policy.

Builds a flat 99-dimensional state vector from point-in-time market data:
  - Per-asset features (11 × 8 = 88): weight, GARCH vol, basket one-hot,
    trailing 5/21/63-day returns
  - Market features (7): 4 detector signals, p_stress, VIX, sector dispersion
  - Portfolio features (4): rolling Sharpe, rolling vol, drawdown, days since rebalance

CRITICAL: At time t, only data from times ≤ t is accessed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class StateBuilder:
    """Constructs the flat state vector from raw market data."""

    # Per-asset: weight(1) + vol(1) + basket_onehot(3) + trailing_ret(3) = 8
    F_ASSET: int = 8
    # Market: 4 detectors + p_stress + VIX_norm + sector_dispersion = 7
    F_MARKET: int = 7
    # Portfolio: rolling_sharpe + rolling_vol + drawdown + days_since_rebal = 4
    F_PORTFOLIO: int = 4

    def __init__(self, tickers: list[str]) -> None:
        self.tickers = sorted(tickers)
        self.n_assets = len(tickers)
        self.state_dim = self.n_assets * self.F_ASSET + self.F_MARKET + self.F_PORTFOLIO

    def build(
        self,
        day_idx: int,
        log_ret: pd.DataFrame,
        current_weights: dict[str, float],
        assignments: dict,
        vol_dict: dict[str, float],
        p_stress: float,
        detector_signals: dict[str, float],
        spy_ret: pd.Series,
        vix_prices: pd.Series | None = None,
        portfolio_returns: list[float] | None = None,
        days_since_rebalance: int = 0,
    ) -> np.ndarray:
        """Build flat state vector at time t using ONLY data ≤ t.

        Parameters
        ----------
        day_idx : int
            Current position in log_ret (0-indexed).
        log_ret : pd.DataFrame
            Full log-return DataFrame (only rows ≤ day_idx are accessed).
        current_weights : dict[str, float]
            Current portfolio weights.
        assignments : dict
            Basket assignments (ticker → BasketAssignment or str).
        vol_dict : dict[str, float]
            GARCH annualised conditional vol per ticker.
        p_stress : float
            Composite stress probability from aggregator.
        detector_signals : dict[str, float]
            Keys: cusum, ewma, markov, structural.
        spy_ret : pd.Series
            SPY daily log returns.
        vix_prices : pd.Series | None
            VIX close prices (raw, not returns).
        portfolio_returns : list[float] | None
            Accumulated portfolio daily returns up to now.
        days_since_rebalance : int
            Trading days since last walk-forward recalibration.

        Returns
        -------
        np.ndarray
            Shape (state_dim,), all finite values.
        """
        state = np.zeros(self.state_dim, dtype=np.float64)
        offset = 0

        # ── Per-asset features (11 × 8) ──────────────────────────────
        for tkr in self.tickers:
            # 1. Current weight
            state[offset] = current_weights.get(tkr, 0.0)
            # 2. GARCH conditional vol (annualised)
            state[offset + 1] = vol_dict.get(tkr, 0.2)
            # 3-5. Basket one-hot [is_A, is_B, is_C]
            basket = self._get_basket(assignments, tkr)
            state[offset + 2] = 1.0 if basket == "A" else 0.0
            state[offset + 3] = 1.0 if basket == "B" else 0.0
            state[offset + 4] = 1.0 if basket == "C" else 0.0
            # 6-8. Trailing returns (5d, 21d, 63d)
            if tkr in log_ret.columns:
                col = log_ret[tkr].iloc[: day_idx + 1]
                state[offset + 5] = float(col.iloc[-5:].sum()) if len(col) >= 5 else 0.0
                state[offset + 6] = float(col.iloc[-21:].sum()) if len(col) >= 21 else 0.0
                state[offset + 7] = float(col.iloc[-63:].sum()) if len(col) >= 63 else 0.0
            offset += self.F_ASSET

        # ── Market features (7) ───────────────────────────────────────
        state[offset] = detector_signals.get("cusum", 0.0)
        state[offset + 1] = detector_signals.get("ewma", 0.0)
        state[offset + 2] = detector_signals.get("markov", 0.0)
        state[offset + 3] = detector_signals.get("structural", 0.0)
        state[offset + 4] = p_stress
        # VIX normalised by 30
        if vix_prices is not None and day_idx < len(vix_prices):
            vix_val = vix_prices.iloc[day_idx]
            state[offset + 5] = (vix_val / 30.0) if not np.isnan(vix_val) else 0.0
        # Cross-sector return dispersion
        day_rets = []
        for tkr in self.tickers:
            if tkr in log_ret.columns and day_idx < len(log_ret):
                r = log_ret[tkr].iloc[day_idx]
                if not np.isnan(r):
                    day_rets.append(r)
        state[offset + 6] = float(np.std(day_rets)) if len(day_rets) > 1 else 0.0
        offset += self.F_MARKET

        # ── Portfolio features (4) ────────────────────────────────────
        if portfolio_returns and len(portfolio_returns) >= 2:
            pr = np.array(portfolio_returns)
            recent = pr[-21:] if len(pr) >= 21 else pr
            # Rolling 21-day realised Sharpe
            if len(recent) >= 5 and np.std(recent) > 1e-8:
                state[offset] = float(np.mean(recent) / np.std(recent) * np.sqrt(252))
            # Rolling 21-day realised vol
            state[offset + 1] = float(np.std(recent) * np.sqrt(252))
            # Current drawdown from peak
            cum = np.cumsum(pr)
            peak = np.maximum.accumulate(cum)
            dd = cum[-1] - peak[-1]
            state[offset + 2] = float(dd)
        # Days since last rebalance (normalised by 63)
        state[offset + 3] = days_since_rebalance / 63.0
        offset += self.F_PORTFOLIO

        # Replace any NaN/inf with 0
        state = np.nan_to_num(state, nan=0.0, posinf=5.0, neginf=-5.0)
        return state

    @staticmethod
    def _get_basket(assignments: dict, ticker: str) -> str:
        """Extract basket label from assignments dict."""
        if ticker not in assignments:
            return "C"  # Default to Core
        ba = assignments[ticker]
        if hasattr(ba, "basket"):
            return ba.basket
        return str(ba)
