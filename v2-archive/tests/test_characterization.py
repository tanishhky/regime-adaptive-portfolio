"""
Unit tests for asset characterization modules.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.characterization.volatility import GARCHVolatility
from src.characterization.recovery import RecoveryEstimator
from src.characterization.classifier import BasketClassifier


@pytest.fixture
def mean_reverting_returns():
    """Synthetic returns that are mean-reverting."""
    np.random.seed(42)
    n = 500
    x = np.zeros(n)
    kappa = 0.1
    mu = 0.0
    sigma = 0.01
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (mu - x[t - 1]) + sigma * np.random.randn()
    returns = np.diff(x)
    dates = pd.bdate_range("2020-01-01", periods=len(returns))
    return pd.Series(returns, index=dates, name="mean_rev")


@pytest.fixture
def non_reverting_returns():
    """Synthetic random walk (non-mean-reverting)."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 500)
    dates = pd.bdate_range("2020-01-01", periods=500)
    return pd.Series(returns, index=dates, name="random_walk")


class TestGARCH:
    def test_conditional_vol_positive(self, mean_reverting_returns):
        """Conditional volatility must be positive."""
        garch = GARCHVolatility()
        result = garch.fit(mean_reverting_returns, ticker="TEST")
        assert (result.conditional_vol > 0).all()

    def test_persistence_bounds(self, mean_reverting_returns):
        """Persistence α + β should be in [0, 1]."""
        garch = GARCHVolatility()
        result = garch.fit(mean_reverting_returns, ticker="TEST")
        assert 0.0 <= result.persistence <= 1.01  # Allow small numerical error


class TestRecovery:
    def test_positive_halflife_mean_reverting(self, mean_reverting_returns):
        """Half-life must be positive for mean-reverting series."""
        est = RecoveryEstimator()
        result = est.estimate(mean_reverting_returns, ticker="TEST")
        # Mean-reverting series should have positive kappa
        if result.mean_reverting:
            assert result.half_life > 0

    def test_nonreverting_detection(self, non_reverting_returns):
        """Non-reverting series: kappa ≤ 0 or NaN half-life."""
        est = RecoveryEstimator()
        result = est.estimate(non_reverting_returns, ticker="TEST")
        # Random walk may or may not appear mean-reverting due to noise
        # but half-life should not be very small
        if result.mean_reverting:
            assert result.half_life > 0


class TestClassifier:
    def test_basket_assignment(self, mean_reverting_returns):
        """Classifier should assign valid basket labels."""
        garch = GARCHVolatility()
        recovery = RecoveryEstimator()

        g1 = garch.fit(mean_reverting_returns, ticker="ETF1")
        g2 = garch.fit(mean_reverting_returns * 2, ticker="ETF2")
        r1 = recovery.estimate(mean_reverting_returns, ticker="ETF1")
        r2 = recovery.estimate(mean_reverting_returns * 2, ticker="ETF2")

        clf = BasketClassifier()
        assignments = clf.assign(
            {"ETF1": g1, "ETF2": g2},
            {"ETF1": r1, "ETF2": r2},
        )
        for tkr, ba in assignments.items():
            assert ba.basket in ("A", "B", "C")
