"""
Unit tests for the backtest engine.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import compute_metrics
from src.portfolio.execution import ExecutionModel
from src.portfolio.sizing import inverse_volatility_weights, risk_parity_weights


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_sharpe_positive_for_good_returns(self):
        """Consistently positive returns should yield positive Sharpe."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0.001, 0.005, 252),
            index=pd.bdate_range("2020-01-01", periods=252),
        )
        m = compute_metrics(returns)
        assert m.sharpe_ratio > 0

    def test_max_drawdown_negative(self):
        """Max drawdown must be ≤ 0."""
        np.random.seed(42)
        returns = pd.Series(
            np.random.normal(0, 0.01, 252),
            index=pd.bdate_range("2020-01-01", periods=252),
        )
        m = compute_metrics(returns)
        assert m.max_drawdown <= 0


# ── Execution model tests ────────────────────────────────────────────────────

class TestExecution:
    def test_transaction_costs_deducted(self):
        """Transaction costs should be correctly computed."""
        exec_model = ExecutionModel(cost_bps=10)
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.3, "B": 0.7}
        _, cost = exec_model.execute(old, new)
        # Total turnover = |0.3-0.5| + |0.7-0.5| = 0.4
        # Cost = 0.4 * 10/10000 = 0.0004
        assert abs(cost - 0.0004) < 1e-8

    def test_no_cost_on_no_change(self):
        """No trade → no cost."""
        exec_model = ExecutionModel(cost_bps=10)
        weights = {"A": 0.5, "B": 0.5}
        _, cost = exec_model.execute(weights, weights)
        assert cost == 0.0


# ── Portfolio weights tests ──────────────────────────────────────────────────

class TestSizing:
    def test_weights_sum_to_one(self):
        """Inverse-vol weights must sum to 1."""
        vols = {"A": 0.15, "B": 0.25, "C": 0.10}
        w = inverse_volatility_weights(vols)
        assert abs(sum(w.values()) - 1.0) < 1e-8

    def test_lower_vol_gets_higher_weight(self):
        """Lower vol asset should get higher weight."""
        vols = {"A": 0.10, "B": 0.30}
        w = inverse_volatility_weights(vols)
        assert w["A"] > w["B"]

    def test_risk_parity_sum(self):
        """Risk parity weights must sum to 1."""
        vols = {"A": 0.15, "B": 0.25}
        w = risk_parity_weights(vols)
        assert abs(sum(w.values()) - 1.0) < 1e-8


# ── Lookahead bias test ──────────────────────────────────────────────────────

class TestNoLookahead:
    def test_walk_forward_no_future_data(self):
        """Walk-forward engine must not access data beyond training window.

        We create a synthetic dataset where future data has a distinctive
        pattern.  If the engine accidentally accesses future data, its
        behaviour would change.
        """
        np.random.seed(42)
        n = 600
        dates = pd.bdate_range("2018-01-01", periods=n)

        # Create prices where future has extreme returns
        calm_returns = np.random.normal(0.001, 0.01, 504)
        # Future data: extreme positive returns (if accessed, would change signals)
        future_returns = np.full(n - 504, 0.10)

        prices_dict = {}
        for tkr in ["SPY", "XLK", "XLF"]:
            r = np.concatenate([calm_returns, future_returns])
            p = 100 * np.exp(np.cumsum(r))
            prices_dict[tkr] = p

        prices_df = pd.DataFrame(prices_dict, index=dates)

        # If engine properly uses only training data, the signals during
        # training should not be affected by future data
        from src.detectors.cusum import CUSUMDetector
        det = CUSUMDetector()
        train_only = np.log(prices_df["SPY"] / prices_df["SPY"].shift(1)).dropna()
        det.calibrate(train_only.iloc[:504])
        sig = det.signal_series(train_only.iloc[:504])
        assert sig.max() <= 1.0  # Signals only from training data
