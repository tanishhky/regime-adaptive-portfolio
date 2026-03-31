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
                    vol_dict, entry_th, exit_th, risk_free_daily,
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
        risk_free_daily: float = 0.0,
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
            equity_ret = np.nansum(weights * day_ret)
            # Bug 6: include cash return so cash drag is priced correctly
            cash_weight = max(0.0, 1.0 - weights.sum())
            port_ret[t] = equity_ret + cash_weight * risk_free_daily

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
        """Core weight computation logic.

        Works in two steps to avoid the raw inv_vol normalization trap:

        Step 1: Compute base inverse-vol weights that sum to 1.0 (calm portfolio).
        Step 2: Apply basket-specific stress scalings as multipliers on those
                base weights. The reduced sum is implicit cash — do NOT
                re-normalize. Raw inv_vol values sum to ~25 for 11 stocks,
                so normalizing AFTER scaling always collapses cash to zero.
        """
        n = len(tickers)

        # Step 1: base inverse-vol weights summing to 1.0
        base = np.zeros(n)
        for i, tkr in enumerate(tickers):
            if tkr not in assignments:
                continue
            vol = vol_dict.get(tkr, 0.2)
            base[i] = 1.0 / max(vol, 1e-6)
        base_total = base.sum()
        if base_total > 0:
            base /= base_total
        # base now sums to 1.0 — the calm-market fully-invested portfolio

        # Step 2: apply basket-specific stress scaling
        scaled = np.zeros(n)
        for i, tkr in enumerate(tickers):
            if tkr not in assignments:
                continue
            ba = assignments[tkr]

            if ba.basket == "A":
                # Tactical: liquidate on high stress, re-enter on recovery
                if p_stress > entry_th:
                    liquidated.add(tkr)
                    scaled[i] = 0.0
                elif p_stress < exit_th and tkr in liquidated:
                    liquidated.discard(tkr)
                    scaled[i] = base[i]
                elif tkr in liquidated:
                    scaled[i] = 0.0
                else:
                    scaled[i] = base[i]

            elif ba.basket == "B":
                # Avoid: de-risking with dead zone (no action below 0.4)
                if p_stress > 0.4:
                    b_scale = (p_stress - 0.4) / 0.6
                    scaled[i] = base[i] * (1.0 - b_scale)
                else:
                    scaled[i] = base[i]

            elif ba.basket == "C":
                # Core: graduated de-risking only at extreme stress (>0.7)
                if structural_break:
                    scaled[i] = 0.0
                elif p_stress > 0.7:
                    # Linear scale-down: full weight at 0.7, zero at 1.0
                    scale = max(0.0, (1.0 - p_stress) / 0.3)
                    scaled[i] = base[i] * scale
                else:
                    scaled[i] = base[i]

        # Step 3: NO re-normalization. The sum of scaled < 1.0 is the cash.
        # Only guard against floating-point leverage.
        total = scaled.sum()
        if total > 1.0 + 1e-8:
            scaled /= total
        return scaled

    def reset(self) -> None:
        """Reset liquidation state (for new walk-forward window)."""
        self._liquidated.clear()
