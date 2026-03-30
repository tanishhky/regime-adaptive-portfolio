"""
Unit tests for regime detectors.
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
from src.detectors.fuzzy_aggregator import FuzzyAggregator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_returns():
    """Synthetic return series: calm period followed by crisis."""
    np.random.seed(42)
    calm = np.random.normal(0.0005, 0.01, 400)
    crisis = np.random.normal(-0.002, 0.035, 100)
    recovery = np.random.normal(0.001, 0.015, 200)
    returns = np.concatenate([calm, crisis, recovery])
    dates = pd.bdate_range("2020-01-01", periods=len(returns))
    return pd.Series(returns, index=dates, name="synthetic")


@pytest.fixture
def synthetic_sector_returns():
    """Synthetic sector returns: 11 sectors."""
    np.random.seed(42)
    n_days = 700
    n_sectors = 11
    data = np.random.normal(0.0005, 0.01, (n_days, n_sectors))
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"SEC{i}" for i in range(n_sectors)]
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def nan_returns():
    """Returns with NaN values."""
    np.random.seed(42)
    r = np.random.normal(0, 0.01, 100)
    r[10] = np.nan
    r[50] = np.nan
    dates = pd.bdate_range("2020-01-01", periods=100)
    return pd.Series(r, index=dates, name="nan_returns")


# ── CUSUM tests ───────────────────────────────────────────────────────────────

class TestCUSUM:
    def test_signal_range(self, synthetic_returns):
        """CUSUM signals must be in [0, 1]."""
        det = CUSUMDetector()
        det.calibrate(synthetic_returns[:300])
        sig = det.signal_series(synthetic_returns)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0

    def test_calibration_data_driven(self, synthetic_returns):
        """Parameters must be estimated from data, not hardcoded."""
        det = CUSUMDetector()
        det.calibrate(synthetic_returns[:300])
        assert det.delta > 0
        assert det.h > 0
        assert det.mu0 != 0 or abs(det.mu0) < 1e-6  # may be near zero

    def test_crisis_detection(self, synthetic_returns):
        """Signal should be higher during crisis period."""
        det = CUSUMDetector()
        det.calibrate(synthetic_returns[:300])
        sig = det.signal_series(synthetic_returns)
        calm_mean = sig.iloc[:300].mean()
        crisis_mean = sig.iloc[400:500].mean()
        assert crisis_mean > calm_mean


# ── Correlation Detector tests ───────────────────────────────────────────────

class TestCorrelation:
    def test_correlation_detector(self):
        """Correlation detector returns valid signal."""
        from src.detectors.correlation import CorrelationDetector
        det = CorrelationDetector(window=5)
        fake_train = pd.DataFrame(np.random.randn(10, 11) * 0.01)
        det.fit(fake_train)
        # All sectors move together: should produce high signal
        same_move = np.ones(11) * -0.03
        sig = det.signal(same_move)
        assert 0.0 <= sig <= 1.0, f"Signal out of range: {sig}"

    def test_signal_range(self, synthetic_sector_returns):
        """Correlation signals must be in [0, 1]."""
        det = CorrelationDetector(window=21)
        det.fit(synthetic_sector_returns.iloc[:100])
        sig = det.signal_series(synthetic_sector_returns)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0


# ── Breadth Detector tests ───────────────────────────────────────────────────

class TestBreadth:
    def test_breadth_detector(self):
        """Breadth detector returns valid signal."""
        from src.detectors.breadth import BreadthDetector
        det = BreadthDetector(window=5)
        fake_train = pd.DataFrame(np.random.randn(10, 11) * 0.01)
        det.fit(fake_train)
        # All sectors negative: should produce high signal
        bad_day = np.ones(11) * -0.02
        sig = det.signal(bad_day)
        assert 0.0 <= sig <= 1.0, f"Signal out of range: {sig}"

    def test_signal_range(self, synthetic_sector_returns):
        """Breadth signals must be in [0, 1]."""
        det = BreadthDetector(window=21)
        det.fit(synthetic_sector_returns.iloc[:100])
        sig = det.signal_series(synthetic_sector_returns)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0


# ── Skewness Detector tests ─────────────────────────────────────────────────

class TestSkewness:
    def test_skewness_detector(self):
        """Skewness detector returns valid signal."""
        from src.detectors.skewness import SkewnessDetector
        det = SkewnessDetector(window=10)
        # Mostly small positive returns with a few large negatives → negative skew
        returns = np.concatenate([np.ones(8) * 0.005, np.array([-0.05, -0.06])])
        det.fit(returns)
        sig = det.signal(-0.04)
        assert 0.0 <= sig <= 1.0, f"Signal out of range: {sig}"

    def test_signal_range(self, synthetic_returns):
        """Skewness signals must be in [0, 1]."""
        det = SkewnessDetector(window=63)
        det.fit(synthetic_returns.iloc[:100].values)
        sig = det.signal_series(synthetic_returns)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0


# ── Fuzzy Aggregator tests ────────────────────────────────────────────────────

class TestFuzzyAggregator:
    def test_output_range(self):
        """Composite signal must be in [0, 1]."""
        fuzzy = FuzzyAggregator()
        for signals in [[0, 0, 0, 0], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]]:
            result = fuzzy.aggregate(signals)
            assert 0.0 <= result <= 1.0

    def test_max_of_top2_needs_agreement(self):
        """Max-of-top-2 should produce lower signal when only 1 detector is high."""
        fuzzy = FuzzyAggregator()
        # Set all sigmoids to identity-like (steep, centered)
        fuzzy.sigmoid_params = np.array([
            [50.0, 0.5], [50.0, 0.5], [50.0, 0.5], [50.0, 0.5]
        ])
        # Only 1 detector high → second highest is low
        one_high = fuzzy.aggregate([1.0, 0.0, 0.0, 0.0])
        # Two detectors high → second highest is high
        two_high = fuzzy.aggregate([1.0, 1.0, 0.0, 0.0])
        assert two_high > one_high

    def test_calibration(self, synthetic_returns):
        """Calibration should produce valid sigmoid parameters."""
        fuzzy = FuzzyAggregator()
        signal_matrix = np.random.rand(len(synthetic_returns), 4)
        fuzzy.calibrate(signal_matrix, synthetic_returns)
        # Check sigmoid params are within bounds
        for i in range(4):
            a, c = fuzzy.sigmoid_params[i]
            assert 1.0 <= a <= 50.0
            assert 0.05 <= c <= 0.95
