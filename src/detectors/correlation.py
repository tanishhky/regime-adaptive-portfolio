"""
Correlation Contagion Detector.

Measures average pairwise correlation among sector ETFs using a rolling window.
Orthogonal to CUSUM: captures market structure (diversification collapse)
rather than volatility magnitude.
"""

import numpy as np
import pandas as pd


class CorrelationDetector:
    """
    Computes rolling average pairwise correlation across sector returns.

    Parameters
    ----------
    window : int
        Rolling window for pairwise correlation calculation (default: 21 trading days).
    """

    def __init__(self, window: int = 21):
        self.window = window
        self._sector_buffer = []  # list of 1-D arrays, one per day
        self._n_sectors = None

    def fit(self, sector_returns: pd.DataFrame):
        """
        Fit on training data. Stores the number of sectors.

        Parameters
        ----------
        sector_returns : pd.DataFrame
            DataFrame with columns = sector tickers, rows = dates, values = daily returns.
        """
        self._n_sectors = sector_returns.shape[1]
        # Pre-fill buffer with last `window` days of training data
        # so OOS signal is available from day 1.
        tail = sector_returns.iloc[-self.window:]
        self._sector_buffer = [row.values for _, row in tail.iterrows()]

    def signal(self, sector_returns_today: np.ndarray) -> float:
        """
        Update with one day of sector returns and return correlation signal.

        Parameters
        ----------
        sector_returns_today : np.ndarray
            1-D array of length n_sectors — today's return for each sector.

        Returns
        -------
        float
            Signal in [0, 1]. Higher = more correlated (stress).
        """
        self._sector_buffer.append(np.asarray(sector_returns_today, dtype=np.float64))
        if len(self._sector_buffer) > self.window:
            self._sector_buffer.pop(0)

        if len(self._sector_buffer) < self.window:
            return 0.0

        # Build matrix: (window, n_sectors)
        mat = np.column_stack(self._sector_buffer) if self._n_sectors == 1 \
            else np.array(self._sector_buffer)

        # Compute pairwise Pearson correlation matrix
        corr_matrix = np.corrcoef(mat.T)  # shape (n_sectors, n_sectors)

        # Average of upper-triangle (excluding diagonal)
        n = corr_matrix.shape[0]
        mask = np.triu_indices(n, k=1)
        avg_corr = np.nanmean(corr_matrix[mask])

        # Map from correlation space to [0, 1] signal.
        # Typical range: 0.1 (calm) to 0.60 (elevated).
        # Linear map: clamp to [0.1, 0.60], then rescale to [0, 1].
        lo, hi = 0.10, 0.60
        raw = (avg_corr - lo) / (hi - lo)
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
        return pd.Series(signals, index=sector_returns.index, name="correlation")

    def reset(self):
        """Clear buffer for a new OOS window (buffer gets re-primed on next fit)."""
        self._sector_buffer = []
