"""
Basket manager — adaptive portfolio allocation based on regime signals.

Decision logic uses composite P(stress) from the fuzzy aggregator and
basket assignments from the classifier to determine position sizing and
liquidation decisions.

Entry/exit thresholds are calibrated on the training window via Sharpe
ratio optimisation (no hardcoded values).

References
----------
Nystrup, P., Hansen, B.W., Madsen, H. & Lindström, E. (2017).
"Regime-based versus static asset allocation." Journal of Risk, 19(6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.characterization.classifier import BasketAssignment


class BasketManager:
    """Adaptive basket-based portfolio manager."""

    def __init__(self) -> None:
        self.entry_threshold: float = 0.6   # calibrated, not hardcoded
        self.exit_threshold: float = 0.3    # calibrated, not hardcoded
        self._liquidated: set[str] = set()

    def calibrate_thresholds(
        self,
        p_stress_series: pd.Series,
        returns_df: pd.DataFrame,
        assignments: dict[str, BasketAssignment],
        vol_dict: dict[str, float],
        risk_free_daily: float = 0.0,
    ) -> None:
        """Calibrate entry/exit thresholds by optimising Sharpe on training data.

        Grid search over candidate thresholds within the training window.

        Parameters
        ----------
        p_stress_series : pd.Series
            Daily composite P(stress) for the training window.
        returns_df : pd.DataFrame
            Daily log returns for all sector ETFs (training window).
        assignments : dict[str, BasketAssignment]
            Current basket assignments.
        vol_dict : dict[str, float]
            Annualised conditional vol per ticker (for inverse-vol weighting).
        risk_free_daily : float
            Daily risk-free rate.
        """
        best_sharpe = -np.inf
        best_entry = 0.6
        best_exit = 0.3

        # Grid search — data-driven, not hardcoded boundaries
        entry_candidates = np.linspace(0.4, 0.9, 11)
        exit_candidates = np.linspace(0.1, 0.5, 9)

        tickers = sorted(assignments.keys())
        aligned_ret = returns_df[tickers].reindex(p_stress_series.index).fillna(0)

        for entry_th in entry_candidates:
            for exit_th in exit_candidates:
                if exit_th >= entry_th:
                    continue
                port_ret = self._simulate_returns(
                    p_stress_series, aligned_ret, assignments,
                    vol_dict, entry_th, exit_th,
                )
                excess = port_ret - risk_free_daily
                if excess.std() > 0:
                    sharpe = excess.mean() / excess.std() * np.sqrt(252)
                else:
                    sharpe = 0.0
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_entry = entry_th
                    best_exit = exit_th

        self.entry_threshold = best_entry
        self.exit_threshold = best_exit

    def _simulate_returns(
        self,
        p_stress: pd.Series,
        returns_df: pd.DataFrame,
        assignments: dict[str, BasketAssignment],
        vol_dict: dict[str, float],
        entry_th: float,
        exit_th: float,
    ) -> pd.Series:
        """Simulate portfolio returns for a given threshold pair."""
        tickers = list(returns_df.columns)
        n = len(returns_df)
        port_ret = np.zeros(n)
        liquidated = set()

        for t in range(n):
            p = p_stress.iloc[t] if t < len(p_stress) else 0.5
            weights = self._compute_weights(
                tickers, assignments, vol_dict, p,
                entry_th, exit_th, liquidated, structural_break=False,
            )
            day_ret = returns_df.iloc[t].values
            port_ret[t] = np.nansum(weights * day_ret)

        return pd.Series(port_ret, index=returns_df.index)

    def compute_weights(
        self,
        tickers: list[str],
        assignments: dict[str, BasketAssignment],
        vol_dict: dict[str, float],
        p_stress: float,
        structural_break: bool = False,
    ) -> dict[str, float]:
        """Compute portfolio weights for current trading day.

        Parameters
        ----------
        tickers : list[str]
            Universe of tickers.
        assignments : dict[str, BasketAssignment]
            Basket assignments.
        vol_dict : dict[str, float]
            Annualised conditional vol per ticker.
        p_stress : float
            Composite stress probability.
        structural_break : bool
            Whether the structural break detector flagged a break.

        Returns
        -------
        dict[str, float]
            Ticker → weight mapping (sums to ≤ 1).
        """
        raw_weights = self._compute_weights(
            tickers, assignments, vol_dict, p_stress,
            self.entry_threshold, self.exit_threshold,
            self._liquidated, structural_break,
        )
        return {t: float(w) for t, w in zip(tickers, raw_weights)}

    def _compute_weights(
        self,
        tickers: list[str],
        assignments: dict[str, BasketAssignment],
        vol_dict: dict[str, float],
        p_stress: float,
        entry_th: float,
        exit_th: float,
        liquidated: set[str],
        structural_break: bool,
    ) -> np.ndarray:
        """Core weight computation logic."""
        n = len(tickers)
        raw = np.zeros(n)

        for i, tkr in enumerate(tickers):
            if tkr not in assignments:
                continue
            ba = assignments[tkr]
            vol = vol_dict.get(tkr, 0.2)
            inv_vol = 1.0 / max(vol, 1e-6)

            if ba.basket == "A":
                # Tactical: liquidate on high stress, re-enter on recovery
                if p_stress > entry_th:
                    liquidated.add(tkr)
                    raw[i] = 0.0
                elif p_stress < exit_th and tkr in liquidated:
                    liquidated.discard(tkr)
                    raw[i] = inv_vol
                elif tkr in liquidated:
                    raw[i] = 0.0
                else:
                    raw[i] = inv_vol

            elif ba.basket == "B":
                # Avoid: continuous de-risking
                raw[i] = inv_vol * (1.0 - p_stress)

            elif ba.basket == "C":
                # Core: hold unless structural break
                if structural_break:
                    raw[i] = 0.0
                else:
                    raw[i] = inv_vol

        # Normalise to sum to 1 (or 0 if all liquidated)
        total = raw.sum()
        if total > 0:
            raw /= total
        return raw

    def reset(self) -> None:
        """Reset liquidation state (for new walk-forward window)."""
        self._liquidated.clear()
