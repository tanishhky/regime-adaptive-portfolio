"""
Unit tests for regime detectors and agreement-scaled aggregator.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.detectors.cusum import CUSUMDetector
from src.detectors.correlation import CorrelationDetector
from src.detectors.breadth import BreadthDetector
from src.detectors.skewness import SkewnessDetector
from src.detectors.fuzzy_aggregator import (
    FuzzyAggregator,
    _compute_response_single,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_returns():
    np.random.seed(42)
    calm = np.random.normal(0.0005, 0.01, 400)
    crisis = np.random.normal(-0.002, 0.035, 100)
    recovery = np.random.normal(0.001, 0.015, 200)
    returns = np.concatenate([calm, crisis, recovery])
    dates = pd.bdate_range("2020-01-01", periods=len(returns))
    return pd.Series(returns, index=dates, name="synthetic")


@pytest.fixture
def synthetic_sector_returns():
    np.random.seed(42)
    n_days = 700
    n_sectors = 11
    data = np.random.normal(0.0005, 0.01, (n_days, n_sectors))
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"SEC{i}" for i in range(n_sectors)]
    return pd.DataFrame(data, index=dates, columns=tickers)


# ── CUSUM tests ───────────────────────────────────────────────────────────────

class TestCUSUM:
    def test_signal_range(self, synthetic_returns):
        det = CUSUMDetector()
        det.calibrate(synthetic_returns[:300])
        sig = det.signal_series(synthetic_returns)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0

    def test_calibration_data_driven(self, synthetic_returns):
        det = CUSUMDetector()
        det.calibrate(synthetic_returns[:300])
        assert det.delta > 0
        assert det.h > 0

    def test_crisis_detection(self, synthetic_returns):
        det = CUSUMDetector()
        det.calibrate(synthetic_returns[:300])
        sig = det.signal_series(synthetic_returns)
        calm_mean = sig.iloc[:300].mean()
        crisis_mean = sig.iloc[400:500].mean()
        assert crisis_mean > calm_mean


# ── Correlation Detector tests ───────────────────────────────────────────────

class TestCorrelation:
    def test_signal_range(self, synthetic_sector_returns):
        det = CorrelationDetector(window=21)
        det.fit(synthetic_sector_returns.iloc[:100])
        sig = det.signal_series(synthetic_sector_returns)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0

    def test_correlated_input(self):
        det = CorrelationDetector(window=5)
        fake = pd.DataFrame(np.random.randn(10, 11) * 0.01)
        det.fit(fake)
        same_move = np.ones(11) * -0.03
        sig = det.signal(same_move)
        assert 0.0 <= sig <= 1.0


# ── Breadth Detector tests ───────────────────────────────────────────────────

class TestBreadth:
    def test_signal_range(self, synthetic_sector_returns):
        det = BreadthDetector(window=21)
        det.fit(synthetic_sector_returns.iloc[:100])
        sig = det.signal_series(synthetic_sector_returns)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0

    def test_all_negative(self):
        det = BreadthDetector(window=5)
        fake = pd.DataFrame(np.random.randn(10, 11) * 0.01)
        det.fit(fake)
        bad_day = np.ones(11) * -0.02
        sig = det.signal(bad_day)
        assert 0.0 <= sig <= 1.0


# ── Skewness Detector tests ─────────────────────────────────────────────────

class TestSkewness:
    def test_signal_range(self, synthetic_returns):
        det = SkewnessDetector(window=63)
        det.fit(synthetic_returns.iloc[:100].values)
        sig = det.signal_series(synthetic_returns)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0

    def test_negative_skew_produces_signal(self):
        det = SkewnessDetector(window=10)
        returns = np.concatenate([np.ones(8) * 0.005, np.array([-0.05, -0.06])])
        det.fit(returns)
        sig = det.signal(-0.04)
        assert 0.0 <= sig <= 1.0


# ── Agreement-Scaled Aggregator tests ────────────────────────────────────────

class TestAgreementScaling:
    def _make_params(self):
        sigmoid_a = np.array([10.0, 10.0, 10.0, 10.0])
        sigmoid_c = np.array([0.5, 0.5, 0.5, 0.5])
        ramp = np.array([0.0, 0.15, 0.50, 0.80, 1.00])
        return sigmoid_a, sigmoid_c, ramp

    def test_zero_agreement(self):
        """No detectors agree -> response = 0 (fully invested)."""
        sa, sc, ramp = self._make_params()
        signals = np.array([0.1, 0.1, 0.1, 0.1])
        resp, n = _compute_response_single(signals, sa, sc, ramp)
        assert resp == 0.0
        assert n == 0

    def test_full_agreement(self):
        """All 4 detectors agree -> response = 1.0 (full de-risk)."""
        sa, sc, ramp = self._make_params()
        signals = np.array([0.9, 0.9, 0.9, 0.9])
        resp, n = _compute_response_single(signals, sa, sc, ramp)
        assert resp == 1.0
        assert n == 4

    def test_partial_agreement(self):
        """2 detectors agree -> response in [0.50, 0.80)."""
        sa, sc, ramp = self._make_params()
        signals = np.array([0.9, 0.9, 0.1, 0.1])
        resp, n = _compute_response_single(signals, sa, sc, ramp)
        assert 0.45 <= resp <= 0.80
        assert n == 2

    def test_monotonic(self):
        """More agreement -> higher response."""
        sa, sc, ramp = self._make_params()
        responses = []
        for n_high in range(5):
            sigs = np.array(
                [0.9 if i < n_high else 0.1 for i in range(4)]
            )
            resp, _ = _compute_response_single(sigs, sa, sc, ramp)
            responses.append(resp)
        for i in range(len(responses) - 1):
            assert responses[i] <= responses[i + 1], (
                f"Monotonicity violated: {responses}"
            )

    def test_class_compute_response(self):
        """FuzzyAggregator.compute_response returns valid output."""
        fuzzy = FuzzyAggregator()
        signals = {
            "cusum": 0.5, "correlation": 0.5,
            "breadth": 0.5, "skewness": 0.5,
        }
        resp, n_agree = fuzzy.compute_response(signals)
        assert 0.0 <= resp <= 1.0
        assert 0 <= n_agree <= 4

    def test_response_series(self, synthetic_returns):
        """compute_response_series returns correct length."""
        fuzzy = FuzzyAggregator()
        sig_matrix = np.random.rand(len(synthetic_returns), 4)
        series = fuzzy.compute_response_series(
            sig_matrix, synthetic_returns.index
        )
        assert len(series) == len(synthetic_returns)
        assert series.min() >= 0.0
        assert series.max() <= 1.0


class TestDifferentialEvolution:
    def test_de_beats_lbfgsb_on_rastrigin(self):
        """DE finds a better optimum on a function with many local minima."""
        from scipy.optimize import differential_evolution, minimize

        def rastrigin(x):
            return 10 * len(x) + sum(
                xi ** 2 - 10 * np.cos(2 * np.pi * xi) for xi in x
            )

        bounds = [(-5.12, 5.12)] * 4
        de_result = differential_evolution(
            rastrigin, bounds, seed=42, maxiter=100
        )
        lb_result = minimize(
            rastrigin, x0=[2.0] * 4, method="L-BFGS-B", bounds=bounds
        )
        assert de_result.fun < lb_result.fun + 1.0


class TestSigmoidParamsArray:
    def test_backward_compat(self):
        """sigmoid_params_array returns (4, 2) array."""
        fuzzy = FuzzyAggregator()
        arr = fuzzy.sigmoid_params_array
        assert arr.shape == (4, 2)
        assert arr[0, 0] == 10.0
        assert arr[0, 1] == 0.5
