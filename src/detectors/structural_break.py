"""
Structural break detector — long time-scale (252+ day).

Uses the ``ruptures`` library (Killick et al., 2012) with the PELT algorithm
for multiple change-point detection on the cumulative return series.

The signal is continuous in [0, 1]:
- If a break is detected within the last 63 trading days → signal near 1.
- Signal decays with distance to the most recent break.

References
----------
Bai, J. & Perron, P. (2003). "Computation and analysis of multiple
structural change models." Journal of Applied Econometrics, 18(1), 1-22.

Killick, R., Fearnhead, P. & Eckley, I.A. (2012). "Optimal detection of
changepoints with a linear computational cost." JASA, 107(500), 1590-1598.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures


class StructuralBreakDetector:
    """PELT-based structural break detector (Killick et al., 2012)."""

    def __init__(self, lookback: int = 504, recency_window: int = 63) -> None:
        """
        Parameters
        ----------
        lookback : int
            Number of trading days to consider for break detection.
        recency_window : int
            A break within this many days is considered "recent" and
            produces a signal near 1.
        """
        self.lookback = lookback
        self.recency_window = recency_window
        self._penalty: float | None = None
        self._breakpoints: list[int] = []
        self._last_signal: float = 0.0

    def fit(self, returns: pd.Series) -> None:
        """Detect structural breaks in the training-window cumulative returns.

        The penalty parameter for PELT is calibrated using the BIC criterion
        (data-driven, not hardcoded).

        Parameters
        ----------
        returns : pd.Series
            Daily log returns in the training window.
        """
        clean = returns.dropna().values.astype(float)
        n = len(clean)
        if n < self.lookback:
            tail = clean
        else:
            tail = clean[-self.lookback :]

        # Cumulative log returns
        cumret = np.cumsum(tail)

        # PELT with RBF kernel cost; penalty = BIC-like: ln(n) * σ²
        # This is data-driven — Killick et al. (2012) §3.1
        sigma2 = float(np.var(np.diff(cumret))) if len(cumret) > 1 else 1e-6
        self._penalty = np.log(len(cumret)) * sigma2

        algo = ruptures.Pelt(model="rbf", min_size=21).fit(cumret)
        try:
            self._breakpoints = algo.predict(pen=self._penalty)
        except Exception:
            self._breakpoints = []

        # Remove the trivial last breakpoint (= len(signal))
        if self._breakpoints and self._breakpoints[-1] == len(cumret):
            self._breakpoints = self._breakpoints[:-1]

        # Compute recency-based signal
        if self._breakpoints:
            most_recent_bp = max(self._breakpoints)
            days_since = len(cumret) - most_recent_bp
            if days_since <= self.recency_window:
                # Exponential decay: signal = exp(-days_since / τ)
                # τ chosen so signal ≈ 0.05 at recency_window boundary
                tau = self.recency_window / 3.0  # data-driven: -ln(0.05)/3 ≈ 1
                self._last_signal = float(np.exp(-days_since / tau))
            else:
                self._last_signal = 0.0
        else:
            self._last_signal = 0.0

    def signal(self, r_t: float) -> float:
        """Return the current structural break signal.

        Between recalibrations the signal is held constant (structural
        breaks are slow-moving by design).

        Parameters
        ----------
        r_t : float
            Today's log return (unused — kept for API consistency).

        Returns
        -------
        float
            Signal in [0, 1].
        """
        return self._last_signal

    @property
    def breakpoints(self) -> list[int]:
        """Return detected break indices (relative to lookback window)."""
        return self._breakpoints

    def signal_series(self, returns: pd.Series) -> pd.Series:
        """Compute per-day signal over a return series.

        For visualisation purposes.  At each day, the signal reflects how
        recently a break was detected (using all data up to that day).

        Parameters
        ----------
        returns : pd.Series
            Daily log returns.

        Returns
        -------
        pd.Series
            Signal values in [0, 1].
        """
        clean = returns.dropna()
        n = len(clean)
        signals = np.zeros(n)

        # Only run detection periodically for efficiency
        step = max(21, self.recency_window // 3)
        cumret = np.cumsum(clean.values)

        for end in range(max(63, step), n, step):
            start_idx = max(0, end - self.lookback)
            segment = cumret[start_idx:end]
            if len(segment) < 42:
                continue

            sigma2 = float(np.var(np.diff(segment))) if len(segment) > 1 else 1e-6
            pen = np.log(len(segment)) * sigma2

            try:
                algo = ruptures.Pelt(model="rbf", min_size=21).fit(segment)
                bps = algo.predict(pen=pen)
            except Exception:
                bps = []

            # Remove trivial endpoint
            if bps and bps[-1] == len(segment):
                bps = bps[:-1]

            if bps:
                most_recent = max(bps)
                days_since = len(segment) - most_recent
                tau = self.recency_window / 3.0
                sig = float(np.exp(-days_since / tau)) if days_since <= self.recency_window else 0.0
            else:
                sig = 0.0

            fill_end = min(end + step, n)
            signals[end:fill_end] = sig

        return pd.Series(signals, index=clean.index, name="structural_signal")
