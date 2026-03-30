"""
Takagi-Sugeno fuzzy aggregator for combining multi-scale detector outputs.

Each detector's [0, 1] signal is mapped through a calibrated sigmoid
membership function.  The composite signal is the second-largest of the
four transformed signals (max-of-top-2 rule), requiring at least two
detectors to agree on elevated stress before acting.

Sigmoid parameters are optimised by minimising the Brier score against
realised drawdowns in the training window.

References
----------
Takagi, T. & Sugeno, M. (1985). "Fuzzy identification of systems and its
applications to modeling and control." IEEE Trans. SMC, (1), 116-132.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class FuzzyAggregator:
    """Takagi-Sugeno fuzzy inference system for regime signal aggregation."""

    N_DETECTORS = 4  # CUSUM, Correlation, Breadth, Skewness

    def __init__(self) -> None:
        # Sigmoid parameters per detector: a (steepness), c (crossover)
        self.sigmoid_params: np.ndarray = np.tile([10.0, 0.5], (self.N_DETECTORS, 1))

    # ── sigmoid membership ────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray, a: float, c: float) -> np.ndarray:
        """Sigmoid membership function μ_high(x) = 1 / (1 + exp(-a*(x - c)))."""
        z = -a * (x - c)
        return 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))

    # ── calibration ───────────────────────────────────────────────────────

    def calibrate(
        self,
        signal_matrix: np.ndarray,
        returns: pd.Series,
        drawdown_pct: float = 10.0,
        forward_window: int = 21,
    ) -> None:
        """Calibrate sigmoid parameters via Brier score minimisation.

        Parameters
        ----------
        signal_matrix : np.ndarray
            Shape (T, 4) — daily signals from the four detectors.
        returns : pd.Series
            Daily log returns (aligned with signal_matrix rows).
        drawdown_pct : float
            Percentile of training-window drawdowns used to define
            "significant" drawdowns (data-driven threshold).
        forward_window : int
            Number of days to look ahead for drawdown realisation.
        """
        # Compute drawdown series
        prices = np.exp(np.cumsum(returns.dropna().values))
        peak = np.maximum.accumulate(prices)
        dd = (prices - peak) / peak  # negative values

        # Data-driven threshold: 10th percentile of drawdowns (most negative)
        dd_threshold = np.percentile(dd, drawdown_pct)

        # Binary target: did a drawdown exceeding threshold occur
        # within the next `forward_window` days?
        T = len(dd)
        target = np.zeros(T)
        for t in range(T - forward_window):
            if np.min(dd[t : t + forward_window]) < dd_threshold:
                target[t] = 1.0

        # Trim signal_matrix to match
        n = min(len(signal_matrix), T)
        S = signal_matrix[:n]
        D = target[:n]

        # Pack parameters: [a0, c0, a1, c1, ..., a3, c3] — sigmoid only, no weights
        x0 = np.zeros(self.N_DETECTORS * 2)
        for i in range(self.N_DETECTORS):
            x0[2 * i] = 10.0       # steepness
            x0[2 * i + 1] = 0.5    # crossover

        def objective(params: np.ndarray) -> float:
            """Brier score objective with max-of-top-2 aggregation."""
            a_c = params.reshape(self.N_DETECTORS, 2)

            # Compute membership values for all detectors
            memberships = np.zeros((n, self.N_DETECTORS))
            for i in range(self.N_DETECTORS):
                memberships[:, i] = self._sigmoid(S[:, i], a_c[i, 0], a_c[i, 1])

            # Max-of-top-2: sort descending per row, take the second value
            sorted_memberships = np.sort(memberships, axis=1)[:, ::-1]
            composite = sorted_memberships[:, 1]  # second-largest

            # Brier score
            return float(np.mean((composite - D) ** 2))

        # Bounds: steepness a in [1, 50], crossover c in [0.05, 0.95]
        bounds = []
        for i in range(self.N_DETECTORS):
            bounds.append((1.0, 50.0))    # a_i: steepness
            bounds.append((0.05, 0.95))   # c_i: crossover

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-10},
        )

        best = result.x
        self.sigmoid_params = best.reshape(self.N_DETECTORS, 2)

    # ── inference ─────────────────────────────────────────────────────────

    def aggregate(self, signals: list[float] | np.ndarray) -> float:
        """Compute composite P(stress) from four detector signals.

        Uses the max-of-top-2 rule: the second-largest sigmoid-transformed
        signal. This requires at least two detectors to agree on stress.

        Parameters
        ----------
        signals : list[float] or np.ndarray
            Length-4 array of detector signals, each in [0, 1].

        Returns
        -------
        float
            Composite stress probability in [0, 1].
        """
        s = np.asarray(signals, dtype=float)
        memberships = np.zeros(self.N_DETECTORS)
        for i in range(self.N_DETECTORS):
            a, c = self.sigmoid_params[i]
            memberships[i] = self._sigmoid(s[i : i + 1], a, c)[0]

        # Sort descending, take second-largest
        sorted_m = np.sort(memberships)[::-1]
        return float(np.clip(sorted_m[1], 0.0, 1.0))

    def aggregate_series(self, signal_matrix: np.ndarray) -> np.ndarray:
        """Aggregate a matrix of detector signals (T × 4) → (T,).

        Parameters
        ----------
        signal_matrix : np.ndarray
            Shape (T, 4).

        Returns
        -------
        np.ndarray
            Composite stress probability for each day.
        """
        T = signal_matrix.shape[0]
        result = np.zeros(T)
        for t in range(T):
            result[t] = self.aggregate(signal_matrix[t])
        return result
