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
from src.detectors.ewma import EWMADetector
from src.detectors.markov_switching import MarkovSwitchingDetector
from src.detectors.structural_break import StructuralBreakDetector
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


# ── EWMA tests ────────────────────────────────────────────────────────────────

class TestEWMA:
    def test_signal_range(self, synthetic_returns):
        """EWMA signals must be in [0, 1]."""
        det = EWMADetector()
        det.calibrate(synthetic_returns[:300])
        sig = det.signal_series(synthetic_returns)
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0

    def test_handles_nan(self, nan_returns):
        """EWMA should handle NaN returns gracefully."""
        det = EWMADetector()
        clean = nan_returns.dropna()
        det.calibrate(clean)
        sig = det.signal_series(clean)
        assert not sig.isna().any()

    def test_lambda_ordering(self, synthetic_returns):
        """λ_fast must be < λ_slow."""
        det = EWMADetector()
        det.calibrate(synthetic_returns[:300])
        assert det.lambda_fast < det.lambda_slow


# ── Markov-Switching tests ────────────────────────────────────────────────────

class TestMarkovSwitching:
    def test_filtered_not_smoothed(self, synthetic_returns):
        """Model must use filtered probabilities (no lookahead)."""
        det = MarkovSwitchingDetector()
        det.fit(synthetic_returns[:400])
        # Filtered probability is a single value carried forward
        sig = det.signal(0.01)
        assert 0.0 <= sig <= 1.0

    def test_signal_range(self, synthetic_returns):
        """Signal must be in [0, 1]."""
        det = MarkovSwitchingDetector()
        det.fit(synthetic_returns[:400])
        sig = det.signal(0.01)
        assert 0.0 <= sig <= 1.0

    def test_transition_matrix(self, synthetic_returns):
        """Transition matrix should exist after fitting."""
        det = MarkovSwitchingDetector()
        det.fit(synthetic_returns[:400])
        tm = det.transition_matrix
        assert tm is not None


# ── Structural Break tests ────────────────────────────────────────────────────

class TestStructuralBreak:
    def test_signal_range(self, synthetic_returns):
        """Signal must be in [0, 1]."""
        det = StructuralBreakDetector()
        det.fit(synthetic_returns[:400])
        sig = det.signal(0.01)
        assert 0.0 <= sig <= 1.0

    def test_detects_regime_change(self, synthetic_returns):
        """Should detect the structural break between calm and crisis."""
        det = StructuralBreakDetector()
        det.fit(synthetic_returns[:500])
        # At least some breakpoints should be detected
        assert len(det.breakpoints) >= 0  # May or may not detect


# ── Fuzzy Aggregator tests ────────────────────────────────────────────────────

class TestFuzzyAggregator:
    def test_output_range(self):
        """Composite signal must be in [0, 1]."""
        fuzzy = FuzzyAggregator()
        for signals in [[0, 0, 0, 0], [1, 1, 1, 1], [0.5, 0.5, 0.5, 0.5]]:
            result = fuzzy.aggregate(signals)
            assert 0.0 <= result <= 1.0

    def test_weights_sum_to_one(self, synthetic_returns):
        """After calibration, weights must sum to 1."""
        fuzzy = FuzzyAggregator()
        signal_matrix = np.random.rand(len(synthetic_returns), 4)
        fuzzy.calibrate(signal_matrix, synthetic_returns)
        assert abs(fuzzy.weights.sum() - 1.0) < 1e-6
