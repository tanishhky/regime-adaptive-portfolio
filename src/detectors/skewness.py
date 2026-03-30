"""
Tail Asymmetry (Skewness) Detector.

Measures rolling skewness of benchmark returns over a medium-term window.
Orthogonal to CUSUM: captures distributional asymmetry (tail risk building)
rather than volatility level. Acts as a leading indicator.
"""

import numpy as np
import pandas as pd
from collections import deque


class SkewnessDetector:
    """
    Computes rolling skewness of benchmark (SPY) daily returns.

    Parameters
    ----------
    window : int
        Rolling window for skewness calculation (default: 63 trading days / 1 quarter).
    """

    def __init__(self, window: int = 63):
        self.window = window
        self._buffer = deque(maxlen=window)

    def fit(self, spy_returns: np.ndarray):
        """
        Fit on training data. Pre-fills buffer with the last `window` days.

        Parameters
        ----------
        spy_returns : np.ndarray
            1-D array of daily SPY returns from training period.
        """
        tail = spy_returns[-self.window:]
        self._buffer = deque(tail, maxlen=self.window)

    def signal(self, r_t: float) -> float:
        """
        Update with today's SPY return and compute skewness signal.

        Parameters
        ----------
        r_t : float
            Today's SPY daily return.

        Returns
        -------
        float
            Signal in [0, 1]. Higher = more negative skew (stress building).
        """
        self._buffer.append(r_t)

        if len(self._buffer) < self.window:
            return 0.0

        arr = np.array(self._buffer)
        mu = arr.mean()
        std = arr.std(ddof=1)

        if std < 1e-10:
            return 0.0

        # Fisher skewness (unbiased)
        n = len(arr)
        skew = (n / ((n - 1) * (n - 2))) * np.sum(((arr - mu) / std) ** 3)

        # skew is typically in [-2, +1] for daily equity returns.
        # Normal market: ~ -0.3. Pre-crisis: ~ -1.5 to -2.0. Post-crash recovery: ~ +0.5.
        # We want: more negative skew → higher signal.
        # Map: skew in [-2.0, 0.0] → signal in [1.0, 0.0] (inverted, linear).
        # Positive skew → 0 (no stress signal).
        if skew >= 0.0:
            return 0.0

        raw = skew / -0.8  # skew=-0.8 → raw=1, skew=0 → raw=0
        return float(np.clip(raw, 0.0, 1.0))

    def signal_series(self, spy_returns: pd.Series) -> pd.Series:
        """Generate signal series over a Series of SPY returns.

        Parameters
        ----------
        spy_returns : pd.Series
            Daily SPY returns.

        Returns
        -------
        pd.Series
            Signal for each date.
        """
        signals = []
        for r in spy_returns.values:
            signals.append(self.signal(float(r) if not np.isnan(r) else 0.0))
        return pd.Series(signals, index=spy_returns.index, name="skewness")

    def reset(self):
        """Clear buffer for a new OOS window."""
        self._buffer = deque(maxlen=self.window)
