"""
EWMA volatility crossover detector — short time-scale (21-42 day).

Two EWMA volatility estimators with different decay factors.  The regime
signal is the normalised ratio (σ_fast − σ_slow) / σ_slow, mapped to [0, 1]
via the empirical CDF of historical crossover values from the training window.

The decay factors λ_fast and λ_slow are estimated by maximising the
log-likelihood of observed squared returns under a Gaussian assumption.

References
----------
Longerstaey, J. & Spencer, M. (1996). "RiskMetrics Technical Document."
J.P. Morgan/Reuters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata


class EWMADetector:
    """Dual-EWMA volatility crossover detector with MLE calibration."""

    def __init__(self) -> None:
        self.lambda_fast: float = 0.94
        self.lambda_slow: float = 0.97
        self._var_fast: float = 0.0
        self._var_slow: float = 0.0
        self._ecdf_values: np.ndarray = np.array([])

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _ewma_variance_series(
        returns: np.ndarray, lam: float
    ) -> np.ndarray:
        """Compute recursive EWMA variance series.

        σ²_t = λ * σ²_{t-1} + (1 − λ) * r_t²
        """
        n = len(returns)
        var = np.empty(n)
        var[0] = returns[0] ** 2
        for t in range(1, n):
            var[t] = lam * var[t - 1] + (1.0 - lam) * returns[t] ** 2
        return var

    @staticmethod
    def _neg_log_likelihood(lam: float, returns: np.ndarray) -> float:
        """Negative Gaussian log-likelihood of squared returns given λ.

        L = -0.5 * Σ [ln(2π σ²_t) + r_t² / σ²_t]
        """
        var = EWMADetector._ewma_variance_series(returns, lam)
        # Avoid log(0)
        var = np.maximum(var, 1e-12)
        ll = -0.5 * np.sum(np.log(2.0 * np.pi * var) + returns ** 2 / var)
        return -ll

    # ── public API ────────────────────────────────────────────────────────

    def calibrate(self, returns: pd.Series) -> None:
        """Calibrate λ_fast and λ_slow via MLE on the training window.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns in the training window.
        """
        r = returns.dropna().values.astype(float)
        if len(r) < 30:
            return

        # MLE for λ_fast (shorter memory → search [0.85, 0.96])
        res_fast = minimize_scalar(
            self._neg_log_likelihood,
            bounds=(0.85, 0.96),
            args=(r,),
            method="bounded",
        )
        self.lambda_fast = float(res_fast.x)

        # MLE for λ_slow (longer memory → search [0.95, 0.995])
        res_slow = minimize_scalar(
            self._neg_log_likelihood,
            bounds=(0.95, 0.995),
            args=(r,),
            method="bounded",
        )
        self.lambda_slow = float(res_slow.x)

        # Ensure λ_fast < λ_slow (fast decays faster → shorter memory)
        if self.lambda_fast >= self.lambda_slow:
            self.lambda_fast, self.lambda_slow = self.lambda_slow - 0.01, self.lambda_slow

        # Build empirical CDF of crossover values for normalisation
        var_fast = self._ewma_variance_series(r, self.lambda_fast)
        var_slow = self._ewma_variance_series(r, self.lambda_slow)
        sigma_fast = np.sqrt(var_fast)
        sigma_slow = np.sqrt(np.maximum(var_slow, 1e-12))
        crossover = (sigma_fast - sigma_slow) / sigma_slow
        self._ecdf_values = np.sort(crossover)

        # Initialise running variances
        self._var_fast = float(var_fast[-1])
        self._var_slow = float(var_slow[-1])

    def signal(self, r_t: float) -> float:
        """Update EWMA variances and return regime signal.

        Parameters
        ----------
        r_t : float
            Today's log return.

        Returns
        -------
        float
            Signal in [0, 1] (empirical-CDF-normalised crossover).
        """
        self._var_fast = self.lambda_fast * self._var_fast + (1.0 - self.lambda_fast) * r_t ** 2
        self._var_slow = self.lambda_slow * self._var_slow + (1.0 - self.lambda_slow) * r_t ** 2
        sigma_fast = np.sqrt(self._var_fast)
        sigma_slow = np.sqrt(max(self._var_slow, 1e-12))
        crossover = (sigma_fast - sigma_slow) / sigma_slow

        # Normalise via empirical CDF from training window
        if len(self._ecdf_values) == 0:
            return 0.0
        rank = np.searchsorted(self._ecdf_values, crossover, side="right")
        return float(rank) / len(self._ecdf_values)

    def signal_series(self, returns: pd.Series) -> pd.Series:
        """Compute signal for a full series (re-initialises running state).

        Parameters
        ----------
        returns : pd.Series
            Daily log returns.

        Returns
        -------
        pd.Series
            Signal values in [0, 1].
        """
        r = returns.dropna()
        signals = []
        # Re-initialise from first observation
        self._var_fast = float(r.iloc[0] ** 2)
        self._var_slow = float(r.iloc[0] ** 2)
        for val in r:
            signals.append(self.signal(float(val)))
        return pd.Series(signals, index=r.index, name="ewma_signal")
