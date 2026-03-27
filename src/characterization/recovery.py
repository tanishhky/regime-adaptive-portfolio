"""
Recovery half-life estimation via Ornstein-Uhlenbeck mean-reversion.

After each significant drawdown (threshold estimated from data), fits a
discretised OU process to the drawdown-recovery path to estimate the
mean-reversion speed κ and the half-life ln(2)/κ.

References
----------
Uhlenbeck, G.E. & Ornstein, L.S. (1930). "On the theory of the Brownian
motion." Physical Review, 36(5), 823.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RecoveryResult:
    """Container for recovery analysis outputs."""

    kappa: float               # mean-reversion speed
    half_life: float           # ln(2) / κ in trading days (NaN if κ ≤ 0)
    n_episodes: int            # number of drawdown episodes analysed
    mean_reverting: bool       # True if κ > 0


class RecoveryEstimator:
    """Ornstein-Uhlenbeck half-life estimator for drawdown recovery."""

    def __init__(self) -> None:
        self._results: dict[str, RecoveryResult] = {}

    @staticmethod
    def _drawdown_series(prices: pd.Series) -> pd.Series:
        """Compute drawdown series: (P_t - peak_t) / peak_t."""
        peak = prices.cummax()
        return (prices - peak) / peak

    @staticmethod
    def _find_drawdown_episodes(
        dd: pd.Series, threshold: float
    ) -> list[pd.Series]:
        """Extract drawdown-recovery episodes exceeding threshold.

        Parameters
        ----------
        dd : pd.Series
            Drawdown series (negative values).
        threshold : float
            Drawdown level to qualify (e.g., -0.05 for 5% drawdown).

        Returns
        -------
        list[pd.Series]
            Each element is the drawdown path from trough to recovery.
        """
        episodes: list[pd.Series] = []
        in_dd = False
        start_idx = 0

        for i in range(len(dd)):
            if not in_dd and dd.iloc[i] < threshold:
                in_dd = True
                start_idx = i
            elif in_dd and dd.iloc[i] >= 0:
                # Recovery complete
                episodes.append(dd.iloc[start_idx : i + 1])
                in_dd = False

        # Include ongoing drawdown if still active
        if in_dd and len(dd) - start_idx > 5:
            episodes.append(dd.iloc[start_idx:])

        return episodes

    def estimate(self, returns: pd.Series, ticker: str = "") -> RecoveryResult:
        """Estimate OU half-life from drawdown episodes in training data.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns (training window).
        ticker : str
            Identifier for caching.

        Returns
        -------
        RecoveryResult
        """
        clean = returns.dropna()
        # Reconstruct price from log returns
        prices = np.exp(clean.cumsum())
        prices = pd.Series(prices.values, index=clean.index)

        dd = self._drawdown_series(prices)

        # Data-driven threshold: 10th percentile of all drawdown values
        dd_threshold = float(dd.quantile(0.10))
        if dd_threshold >= 0:
            dd_threshold = -0.01  # Fallback for very benign periods

        episodes = self._find_drawdown_episodes(dd, dd_threshold)

        if len(episodes) == 0:
            result = RecoveryResult(
                kappa=0.0, half_life=np.nan, n_episodes=0, mean_reverting=False
            )
            if ticker:
                self._results[ticker] = result
            return result

        # Pool all episodes and estimate κ via OLS on discretised OU:
        #   X_t - X_{t-1} = κ(μ - X_{t-1})Δt + ε_t
        # With Δt = 1 day and μ = 0 (full recovery target):
        #   ΔX = -κ X_{t-1} + ε
        # OLS: regress ΔX on X_{t-1}, slope = -κ
        all_x = []
        all_dx = []
        for ep in episodes:
            vals = ep.values.astype(float)
            if len(vals) < 3:
                continue
            dx = np.diff(vals)
            x_lag = vals[:-1]
            all_x.extend(x_lag.tolist())
            all_dx.extend(dx.tolist())

        all_x = np.array(all_x)
        all_dx = np.array(all_dx)

        if len(all_x) < 5:
            result = RecoveryResult(
                kappa=0.0, half_life=np.nan, n_episodes=len(episodes),
                mean_reverting=False,
            )
            if ticker:
                self._results[ticker] = result
            return result

        # OLS regression: ΔX = β₀ + β₁ * X_{t-1}
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_x, all_dx)
        kappa = -slope  # κ = -β₁

        if kappa > 0:
            half_life = np.log(2) / kappa
            mean_reverting = True
        else:
            half_life = np.nan
            mean_reverting = False

        result = RecoveryResult(
            kappa=float(kappa),
            half_life=float(half_life),
            n_episodes=len(episodes),
            mean_reverting=mean_reverting,
        )
        if ticker:
            self._results[ticker] = result
        return result

    def get_result(self, ticker: str) -> RecoveryResult | None:
        """Retrieve cached result."""
        return self._results.get(ticker)

    def estimate_all(
        self, returns_df: pd.DataFrame, tickers: list[str] | None = None
    ) -> dict[str, RecoveryResult]:
        """Estimate recovery for multiple tickers.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Daily log returns, columns = tickers.
        tickers : list[str] | None
            Subset of columns. Defaults to all.

        Returns
        -------
        dict[str, RecoveryResult]
        """
        if tickers is None:
            tickers = list(returns_df.columns)
        for tkr in tickers:
            if tkr in returns_df.columns:
                self.estimate(returns_df[tkr], ticker=tkr)
        return dict(self._results)
