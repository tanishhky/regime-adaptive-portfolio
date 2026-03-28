"""
Takagi-Sugeno fuzzy aggregator for combining multi-scale detector outputs.

Each detector's [0, 1] signal is mapped through a calibrated sigmoid
membership function.  The composite signal is a weighted average of detector
signals, with weights and membership parameters jointly optimised by
minimising the Brier score against realised drawdowns in the training window.

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

    N_DETECTORS = 4  # CUSUM, EWMA, Markov, Structural

    def __init__(self) -> None:
        # Sigmoid parameters per detector: a (steepness), c (crossover)
        self.sigmoid_params: np.ndarray = np.tile([10.0, 0.5], (self.N_DETECTORS, 1))
        # Detector weights (sum to 1)
        self.weights: np.ndarray = np.ones(self.N_DETECTORS) / self.N_DETECTORS

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
        """Calibrate membership functions and weights via Brier score minimisation.

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

        # Pack parameters: [a0, c0, a1, c1, ..., a3, c3, w0, w1, w2, w3]
        x0 = np.zeros(self.N_DETECTORS * 3)
        for i in range(self.N_DETECTORS):
            x0[2 * i] = 10.0       # steepness
            x0[2 * i + 1] = 0.5    # crossover
        # Raw weights (will be softmaxed for sum-to-1 constraint)
        x0[2 * self.N_DETECTORS :] = np.zeros(self.N_DETECTORS)

        def objective(params: np.ndarray) -> float:
            """Brier score objective."""
            a_c = params[: 2 * self.N_DETECTORS].reshape(self.N_DETECTORS, 2)
            raw_w = params[2 * self.N_DETECTORS :]
            # Softmax for weights
            exp_w = np.exp(raw_w - np.max(raw_w))
            w = exp_w / exp_w.sum()

            composite = np.zeros(n)
            for i in range(self.N_DETECTORS):
                membership = self._sigmoid(S[:, i], a_c[i, 0], a_c[i, 1])
                composite += w[i] * membership

            # Brier score
            return float(np.mean((composite - D) ** 2))

        # Bounds: steepness a in [1, 50], crossover c in [0.05, 0.95],
        # raw weights in [-5, 5].  Prevents degenerate sigmoid parameters
        # that zero out detectors (e.g. c=1082 effectively disables a detector).
        bounds = []
        for i in range(self.N_DETECTORS):
            bounds.append((1.0, 50.0))    # a_i: steepness
            bounds.append((0.05, 0.95))   # c_i: crossover
        for i in range(self.N_DETECTORS):
            bounds.append((-5.0, 5.0))    # raw_w_i

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-10},
        )

        best = result.x
        self.sigmoid_params = best[: 2 * self.N_DETECTORS].reshape(
            self.N_DETECTORS, 2
        )
        raw_w = best[2 * self.N_DETECTORS :]
        exp_w = np.exp(raw_w - np.max(raw_w))
        self.weights = exp_w / exp_w.sum()

    # ── inference ─────────────────────────────────────────────────────────

    def aggregate(self, signals: list[float] | np.ndarray) -> float:
        """Compute composite P(stress) from four detector signals.

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
        composite = 0.0
        for i in range(self.N_DETECTORS):
            a, c = self.sigmoid_params[i]
            membership = self._sigmoid(s[i : i + 1], a, c)[0]
            composite += self.weights[i] * membership
        return float(np.clip(composite, 0.0, 1.0))

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
