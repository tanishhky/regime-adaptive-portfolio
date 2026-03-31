"""
Takagi-Sugeno fuzzy aggregator for combining multi-scale detector outputs.

Each detector's [0, 1] signal is mapped through a calibrated sigmoid
membership function.  The composite signal is the second-largest of the
four transformed signals (max-of-top-2 rule), requiring at least two
detectors to agree on elevated stress before acting.

Sigmoid parameters are optimised by minimising the Brier score against
realised drawdowns in the training window.

An agreement-scaled ramp converts the number of agreeing detectors into
a defensive allocation intensity, modulated by the mean membership of
the agreeing detectors.

References
----------
Takagi, T. & Sugeno, M. (1985). "Fuzzy identification of systems and its
applications to modeling and control." IEEE Trans. SMC, (1), 116-132.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution


DETECTOR_NAMES = ["cusum", "correlation", "breadth", "skewness"]
AGREE_THRESHOLD = 0.5


class FuzzyAggregator:
    """Takagi-Sugeno fuzzy inference system for regime signal aggregation."""

    N_DETECTORS = 4  # CUSUM, Correlation, Breadth, Skewness

    def __init__(self) -> None:
        # Sigmoid parameters per detector: {"a": steepness, "c": crossover}
        self.sigmoid_params: dict[str, dict[str, float]] = {
            name: {"a": 10.0, "c": 0.5} for name in DETECTOR_NAMES
        }
        # Ramp parameters (agreement-scaled)
        self.ramp_params: dict[str, float] = {"r1": 0.15, "r2": 0.50}

    # ── sigmoid membership ────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray, a: float, c: float) -> np.ndarray:
        """Sigmoid membership function μ_high(x) = 1 / (1 + exp(-a*(x - c)))."""
        z = -a * (x - c)
        return 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))

    # ── calibration ───────────────────────────────────────────────────────

    def calibrate_sigmoids(
        self,
        signal_matrix: np.ndarray,
        returns: pd.Series,
        drawdown_pct: float = 10.0,
        forward_window: int = 21,
    ) -> None:
        """Calibrate sigmoid parameters via Brier score minimisation.

        Uses the max-of-top-2 rule for the composite: the second-largest
        sigmoid-transformed signal per row.

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

        # Pack parameters: [a0, a1, a2, a3, c0, c1, c2, c3]
        x0 = np.concatenate([np.full(self.N_DETECTORS, 10.0),
                             np.full(self.N_DETECTORS, 0.5)])

        def objective(params: np.ndarray) -> float:
            """Brier score objective with max-of-top-2 aggregation."""
            a_vals = params[: self.N_DETECTORS]
            c_vals = params[self.N_DETECTORS :]

            # Compute membership values for all detectors
            memberships = np.zeros((n, self.N_DETECTORS))
            for i in range(self.N_DETECTORS):
                memberships[:, i] = self._sigmoid(S[:, i], a_vals[i], c_vals[i])

            # Max-of-top-2: sort descending per row, take the second value
            sorted_memberships = np.sort(memberships, axis=1)[:, ::-1]
            composite = sorted_memberships[:, 1]  # second-largest

            # Brier score
            return float(np.mean((composite - D) ** 2))

        # Bounds: steepness a in [1, 50], crossover c in [0.05, 0.80]
        bounds = [(1.0, 50.0)] * self.N_DETECTORS + [(0.05, 0.80)] * self.N_DETECTORS

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-10},
        )

        best = result.x
        a_vals = best[: self.N_DETECTORS]
        c_vals = best[self.N_DETECTORS :]
        self.sigmoid_params = {
            name: {"a": float(a_vals[i]), "c": float(c_vals[i])}
            for i, name in enumerate(DETECTOR_NAMES)
        }

    # ── inference ─────────────────────────────────────────────────────────

    def compute_response(
        self, signals: dict[str, float]
    ) -> tuple[float, int]:
        """Compute composite P(stress) and agreement count from detector signals.

        Uses the max-of-top-2 rule: the second-largest sigmoid-transformed
        signal. Agreement count is the number of memberships exceeding
        AGREE_THRESHOLD.

        Parameters
        ----------
        signals : dict[str, float]
            Mapping of detector name to signal value, each in [0, 1].

        Returns
        -------
        tuple[float, int]
            (composite stress probability, number of agreeing detectors).
        """
        memberships = np.zeros(self.N_DETECTORS)
        for i, name in enumerate(DETECTOR_NAMES):
            p = self.sigmoid_params[name]
            val = signals.get(name, 0.0)
            memberships[i] = self._sigmoid(
                np.array([val]), p["a"], p["c"]
            )[0]

        # Sort descending, take second-largest
        sorted_m = np.sort(memberships)[::-1]
        composite = float(np.clip(sorted_m[1], 0.0, 1.0))

        # Agreement count
        n_agree = int(np.sum(memberships > AGREE_THRESHOLD))

        return composite, n_agree

    def compute_response_series(
        self, signal_matrix: np.ndarray, index: pd.DatetimeIndex
    ) -> pd.Series:
        """Apply compute_response over a matrix of detector signals.

        Parameters
        ----------
        signal_matrix : np.ndarray
            Shape (T, 4).
        index : pd.DatetimeIndex
            Index for the resulting Series.

        Returns
        -------
        pd.Series
            Composite stress probability for each day.
        """
        T = signal_matrix.shape[0]
        result = np.zeros(T)
        for t in range(T):
            signals = {
                name: signal_matrix[t, i]
                for i, name in enumerate(DETECTOR_NAMES)
            }
            result[t], _ = self.compute_response(signals)
        return pd.Series(result, index=index[:T])

    # ── agreement-scaled ramp (retained, unused) ─────────────────────────

    def _ramp_lookup(self, n_agree: int) -> float:
        """Map agreement count to base defensive intensity via ramp.

        Mapping:
            0 → 0.0
            1 → r1
            2 → r2
            3 → r2 + (1 - r2) * 0.25
            4 → 1.0
        """
        r1 = self.ramp_params["r1"]
        r2 = self.ramp_params["r2"]
        mapping = {
            0: 0.0,
            1: r1,
            2: r2,
            3: r2 + (1.0 - r2) * 0.25,
            4: 1.0,
        }
        return mapping.get(n_agree, 0.0)

    def _ramp_objective(
        self,
        params: np.ndarray,
        signal_matrix: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """Negative Sharpe objective for ramp parameter optimisation.

        Parameters
        ----------
        params : np.ndarray
            [r1, r2] — ramp knot values.
        signal_matrix : np.ndarray
            Shape (T, 4).
        returns : np.ndarray
            Daily returns aligned with signal_matrix.

        Returns
        -------
        float
            Negative Sharpe ratio.
        """
        self.ramp_params["r1"] = params[0]
        self.ramp_params["r2"] = params[1]

        T = signal_matrix.shape[0]
        intensities = np.zeros(T)
        for t in range(T):
            signals = {
                name: signal_matrix[t, i]
                for i, name in enumerate(DETECTOR_NAMES)
            }
            _, n_agree = self.compute_response(signals)

            # Base ramp
            base = self._ramp_lookup(n_agree)

            # Modulate by mean membership of agreeing detectors
            memberships = np.zeros(self.N_DETECTORS)
            for i, name in enumerate(DETECTOR_NAMES):
                p = self.sigmoid_params[name]
                memberships[i] = self._sigmoid(
                    np.array([signals[name]]), p["a"], p["c"]
                )[0]
            agreeing = memberships[memberships > AGREE_THRESHOLD]
            modulation = float(np.mean(agreeing)) if len(agreeing) > 0 else 0.0
            intensities[t] = base * modulation

        # Simple defensive portfolio: reduce exposure proportionally
        adj_returns = returns[:T] * (1.0 - intensities)
        mu = np.mean(adj_returns)
        sigma = np.std(adj_returns) + 1e-12
        sharpe = mu / sigma * np.sqrt(252)
        return -sharpe

    def optimize_ramp(
        self, signal_matrix: np.ndarray, returns: np.ndarray
    ) -> None:
        """Optimise ramp parameters via differential evolution.

        Parameters
        ----------
        signal_matrix : np.ndarray
            Shape (T, 4).
        returns : np.ndarray
            Daily returns aligned with signal_matrix.
        """
        bounds = [(0.05, 0.30), (0.30, 0.70)]
        result = differential_evolution(
            self._ramp_objective,
            bounds=bounds,
            args=(signal_matrix, returns),
            seed=42,
            maxiter=200,
            tol=1e-8,
        )
        self.ramp_params["r1"] = result.x[0]
        self.ramp_params["r2"] = result.x[1]

    # ── top-level optimisation entry point ────────────────────────────────

    def optimize_sharpe(
        self,
        spy_returns_train: pd.Series,
        sector_returns_train: pd.DataFrame,
        signal_matrix: np.ndarray,
        train_returns_aligned: pd.Series,
        assignments: dict,
        vol_dict: dict,
        entry_threshold: float,
        exit_threshold: float,
        rf_daily: float,
        cost_bps: float,
    ) -> None:
        """Optimise fuzzy aggregator for maximum risk-adjusted performance.

        Currently only calibrates sigmoid parameters; the DE ramp
        optimisation is retained in code but skipped.

        Parameters
        ----------
        spy_returns_train : pd.Series
            SPY daily returns for the training window.
        sector_returns_train : pd.DataFrame
            Sector daily returns for the training window.
        signal_matrix : np.ndarray
            Shape (T, 4) — daily signals from the four detectors.
        train_returns_aligned : pd.Series
            Daily returns aligned with signal_matrix rows.
        assignments : dict
            Sector cluster assignments.
        vol_dict : dict
            Volatility estimates per sector.
        entry_threshold : float
            Threshold for entering defensive mode.
        exit_threshold : float
            Threshold for exiting defensive mode.
        rf_daily : float
            Daily risk-free rate.
        cost_bps : float
            Transaction cost in basis points.
        """
        self.calibrate_sigmoids(signal_matrix, train_returns_aligned)
