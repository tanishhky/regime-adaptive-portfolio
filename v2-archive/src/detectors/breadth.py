"""
Breadth Momentum Detector.

Measures the fraction of sectors with negative rolling returns.
Orthogonal to CUSUM: captures breadth (how many sectors are declining)
rather than magnitude (how much SPY dropped).
"""

import numpy as np
import pandas as pd


class BreadthDetector:
    """
    Computes rolling breadth: fraction of sectors with negative cumulative returns.

    Parameters
    ----------
    window : int
        Lookback for cumulative sector returns (default: 21 trading days).
    """

    def __init__(self, window: int = 21):
        self.window = window
        self._sector_buffer = []  # list of 1-D arrays
        self._n_sectors = None

    def fit(self, sector_returns: pd.DataFrame):
        """
        Fit on training data. Stores sector count and pre-fills buffer.

        Parameters
        ----------
        sector_returns : pd.DataFrame
            Columns = sector tickers, rows = dates, values = daily returns.
        """
        self._n_sectors = sector_returns.shape[1]
        tail = sector_returns.iloc[-self.window:]
        self._sector_buffer = [row.values for _, row in tail.iterrows()]

    def signal(self, sector_returns_today: np.ndarray) -> float:
        """
        Update with one day of sector returns and return breadth stress signal.

        Parameters
        ----------
        sector_returns_today : np.ndarray
            1-D array of length n_sectors — today's return for each sector.

        Returns
        -------
        float
            Signal in [0, 1]. Higher = fewer sectors positive (stress).
        """
        self._sector_buffer.append(np.asarray(sector_returns_today, dtype=np.float64))
        if len(self._sector_buffer) > self.window:
            self._sector_buffer.pop(0)

        if len(self._sector_buffer) < self.window:
            return 0.0

        mat = np.array(self._sector_buffer)  # (window, n_sectors)

        # Cumulative return over the window for each sector
        cum_returns = mat.sum(axis=0)  # simple sum of log-ish returns is fine for 21d

        # Fraction of sectors with NEGATIVE cumulative return
        frac_negative = np.sum(cum_returns < 0) / self._n_sectors

        # Map to [0, 1].
        # Calm: ~45% negative (5/11 is normal). Stress: ~80%+ negative (9/11).
        # Linear map from [0.25, 0.80] → [0, 1] so calm markets produce ~0.36.
        lo, hi = 0.10, 0.60
        raw = (frac_negative - lo) / (hi - lo)
        return float(np.clip(raw, 0.0, 1.0))

    def signal_series(self, sector_returns: pd.DataFrame) -> pd.Series:
        """Generate signal series over a DataFrame of sector returns.

        Parameters
        ----------
        sector_returns : pd.DataFrame
            Columns = sector tickers, rows = dates.

        Returns
        -------
        pd.Series
            Signal for each date.
        """
        signals = []
        for _, row in sector_returns.iterrows():
            signals.append(self.signal(row.values))
        return pd.Series(signals, index=sector_returns.index, name="breadth")

    def reset(self):
        """Clear buffer for a new OOS window."""
        self._sector_buffer = []
