"""
CUSUM sequential detector — ultra-short time-scale (5-10 day).

Implements the Page (1954) CUSUM sequential analysis test on daily returns.
The allowance parameter k and decision threshold h are calibrated from data
(no hardcoded values).

Mathematics
-----------
S_t^+ = max(0, S_{t-1}^+ + (r_t - μ₀ - k))    # upward CUSUM
S_t^- = max(0, S_{t-1}^- - (r_t - μ₀ + k))    # downward CUSUM

k = 0.5 * δ  where δ = σ_train  (one-std shift to detect)
h ≈ (ln(ARL₀) + 1.166) / δ     (Siegmund 1985 approximation, ARL₀ = 252)

References
----------
Page, E.S. (1954). "Continuous inspection schemes." Biometrika, 41(1/2),
100-115.

Siegmund, D. (1985). "Sequential Analysis: Tests and Confidence Intervals."
Springer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CUSUMDetector:
    """Page (1954) CUSUM detector with data-driven calibration."""

    def __init__(self, target_arl0: int = 252) -> None:
        """
        Parameters
        ----------
        target_arl0 : int
            Target average run length under H₀ (one false alarm per year).
        """
        self.target_arl0 = target_arl0
        self.mu0: float = 0.0
        self.delta: float = 0.0
        self.k: float = 0.0
        self.h: float = 0.0
        self._s_plus: float = 0.0
        self._s_minus: float = 0.0

    def calibrate(self, returns: pd.Series) -> None:
        """Calibrate parameters from training-window returns.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns in the training window.
        """
        clean = returns.dropna()
        self.mu0 = float(clean.mean())
        # δ = 1 standard deviation of training returns (data-driven shift size)
        self.delta = float(clean.std(ddof=1))
        # Allowance parameter  (Page, 1954)
        self.k = 0.5 * self.delta
        # Decision threshold via Siegmund (1985) approximation
        self.h = (np.log(self.target_arl0) + 1.166) / self.delta
        # Reset accumulators
        self._s_plus = 0.0
        self._s_minus = 0.0

    def signal(self, r_t: float) -> float:
        """Update accumulators and return regime signal for one observation.

        Parameters
        ----------
        r_t : float
            Today's log return.

        Returns
        -------
        float
            Continuous signal in [0, 1].  Values near 1 indicate a detected
            mean shift (stress).
        """
        self._s_plus = max(0.0, self._s_plus + (r_t - self.mu0 - self.k))
        self._s_minus = max(0.0, self._s_minus - (r_t - self.mu0 + self.k))
        raw = max(self._s_plus, self._s_minus) / self.h if self.h > 0 else 0.0
        return min(raw, 1.0)

    def reset_accumulators(self) -> None:
        """Reset CUSUM accumulators to zero (after a detected change)."""
        self._s_plus = 0.0
        self._s_minus = 0.0

    def signal_series(self, returns: pd.Series) -> pd.Series:
        """Compute signal for a full series of returns.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns.

        Returns
        -------
        pd.Series
            Signal values in [0, 1], same index as *returns*.
        """
        self.reset_accumulators()
        signals = []
        for r_t in returns:
            signals.append(self.signal(float(r_t)) if not np.isnan(r_t) else 0.0)
        return pd.Series(signals, index=returns.index, name="cusum_signal")
