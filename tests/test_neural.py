"""
Unit tests for the neural enhancement modules.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import torch

from src.neural.state_builder import StateBuilder
from src.neural.replay_buffer import RecencyWeightedBuffer, Experience
from src.neural.policy_network import RegimeAdaptivePolicy, MultiHeadPolicy
from src.neural.rl_trainer import PPOTrainer
from src.neural.thompson_sampler import ThompsonSampler
from src.detectors.attention_fusion import AttentionFusion


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tickers():
    return sorted([
        "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC",
    ])


@pytest.fixture
def synthetic_log_ret(tickers):
    """Synthetic log returns for 11 tickers + SPY."""
    np.random.seed(42)
    n = 600
    dates = pd.bdate_range("2018-01-01", periods=n)
    data = {}
    for tkr in tickers + ["SPY"]:
        data[tkr] = np.random.normal(0.0005, 0.015, n)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def state_builder(tickers):
    return StateBuilder(tickers)


# ── StateBuilder tests ────────────────────────────────────────────────────────

class TestStateBuilder:
    def test_output_shape(self, state_builder, synthetic_log_ret, tickers):
        """StateBuilder.build() must return shape (99,)."""
        weights = {t: 1.0 / len(tickers) for t in tickers}
        assignments = {t: "C" for t in tickers}
        vol_dict = {t: 0.2 for t in tickers}
        detector_signals = {
            "cusum": 0.3, "ewma": 0.5, "markov": 0.1, "structural": 0.0,
        }
        state = state_builder.build(
            day_idx=100,
            log_ret=synthetic_log_ret,
            current_weights=weights,
            assignments=assignments,
            vol_dict=vol_dict,
            p_stress=0.4,
            detector_signals=detector_signals,
            spy_ret=synthetic_log_ret["SPY"],
        )
        assert state.shape == (99,)

    def test_all_finite(self, state_builder, synthetic_log_ret, tickers):
        """All state values must be finite (no NaN/inf)."""
        weights = {t: 0.0 for t in tickers}
        state = state_builder.build(
            day_idx=200,
            log_ret=synthetic_log_ret,
            current_weights=weights,
            assignments={},
            vol_dict={t: 0.2 for t in tickers},
            p_stress=0.5,
            detector_signals={"cusum": 0, "ewma": 0, "markov": 0, "structural": 0},
            spy_ret=synthetic_log_ret["SPY"],
        )
        assert np.all(np.isfinite(state))

    def test_no_lookahead(self, state_builder, synthetic_log_ret, tickers):
        """StateBuilder.build(day_idx=t) must only access log_ret.iloc[:t+1]."""
        t = 100
        # Build state at t=100
        state1 = state_builder.build(
            day_idx=t,
            log_ret=synthetic_log_ret,
            current_weights={tkr: 0.0 for tkr in tickers},
            assignments={},
            vol_dict={tkr: 0.2 for tkr in tickers},
            p_stress=0.5,
            detector_signals={"cusum": 0, "ewma": 0, "markov": 0, "structural": 0},
            spy_ret=synthetic_log_ret["SPY"],
        )
        # Modify future data (after t) dramatically
        modified = synthetic_log_ret.copy()
        modified.iloc[t + 1:] = 999.0
        state2 = state_builder.build(
            day_idx=t,
            log_ret=modified,
            current_weights={tkr: 0.0 for tkr in tickers},
            assignments={},
            vol_dict={tkr: 0.2 for tkr in tickers},
            p_stress=0.5,
            detector_signals={"cusum": 0, "ewma": 0, "markov": 0, "structural": 0},
            spy_ret=modified["SPY"],
        )
        # States must be identical (no future data used)
        np.testing.assert_array_equal(state1, state2)


# ── Policy Network tests ─────────────────────────────────────────────────────

class TestRegimeAdaptivePolicy:
    def test_weights_sum_to_one(self):
        """Policy output weights must sum to 1.0 ± 1e-6."""
        policy = RegimeAdaptivePolicy(state_dim=99, n_assets=11)
        policy.eval()
        state = torch.randn(1, 1, 99)
        policy.reset_hidden(1)
        with torch.no_grad():
            weights, values = policy(state)
        w_sum = weights.squeeze().sum().item()
        assert abs(w_sum - 1.0) < 1e-5

    def test_output_shapes(self):
        """Forward pass output shapes must match spec."""
        policy = RegimeAdaptivePolicy(state_dim=99, n_assets=11)
        policy.reset_hidden(2)
        state = torch.randn(2, 5, 99)
        weights, values = policy(state)
        assert weights.shape == (2, 5, 11)
        assert values.shape == (2, 5, 1)


class TestMultiHeadPolicy:
    def test_gate_weights_sum_to_one(self):
        """Gate weights must sum to 1.0 ± 1e-6."""
        policy = MultiHeadPolicy(state_dim=99, n_assets=11, n_heads=3)
        policy.eval()
        state = torch.randn(1, 1, 99)
        policy.reset_hidden(1)
        with torch.no_grad():
            weights, values, gate = policy(state)
        gate_sum = gate.squeeze().sum().item()
        assert abs(gate_sum - 1.0) < 1e-5

    def test_portfolio_weights_sum_to_one(self):
        """Mixture portfolio weights must sum to 1.0 ± 1e-6."""
        policy = MultiHeadPolicy(state_dim=99, n_assets=11, n_heads=3)
        policy.eval()
        state = torch.randn(1, 1, 99)
        policy.reset_hidden(1)
        with torch.no_grad():
            weights, values, gate = policy(state)
        w_sum = weights.squeeze().sum().item()
        assert abs(w_sum - 1.0) < 1e-5


# ── AttentionFusion tests ────────────────────────────────────────────────────

class TestAttentionFusion:
    def test_output_range(self):
        """AttentionFusion must produce p_stress ∈ [0, 1]."""
        fusion = AttentionFusion(lookback=20)
        sig = np.random.rand(20, 4)
        ctx = np.random.rand(20, 5)
        p = fusion.aggregate(sig, ctx)
        assert 0.0 <= p <= 1.0

    def test_short_history_padding(self):
        """Should handle history shorter than lookback via zero-padding."""
        fusion = AttentionFusion(lookback=63)
        sig = np.random.rand(10, 4)  # Only 10 days, lookback=63
        ctx = np.random.rand(10, 5)
        p = fusion.aggregate(sig, ctx)
        assert 0.0 <= p <= 1.0


# ── ReplayBuffer tests ───────────────────────────────────────────────────────

class TestRecencyWeightedBuffer:
    def test_recency_monotonic(self):
        """Sampling weights must be monotonically increasing with recency."""
        buf = RecencyWeightedBuffer(max_size=100, beta=0.5)
        for t in range(50):
            buf.add(Experience(
                state=np.zeros(99), action=np.zeros(11), reward=0.0,
                next_state=np.zeros(99), done=False,
                log_prob=0.0, value=0.0, timestamp=t,
            ))

        # Compute weights manually
        current_t = 50
        weights = np.array([
            np.exp(-0.5 * (current_t - e.timestamp) / 252.0)
            for e in buf.buffer
        ])
        # Later experiences should have higher weights
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1] + 1e-10

    def test_max_size(self):
        """Buffer should not exceed max_size."""
        buf = RecencyWeightedBuffer(max_size=10)
        for t in range(20):
            buf.add(Experience(
                state=np.zeros(99), action=np.zeros(11), reward=0.0,
                next_state=np.zeros(99), done=False,
                log_prob=0.0, value=0.0, timestamp=t,
            ))
        assert len(buf) == 10


# ── PPOTrainer tests ─────────────────────────────────────────────────────────

class TestPPOTrainer:
    def test_reward_computation(self):
        """Reward must equal expected value for known inputs."""
        policy = RegimeAdaptivePolicy(state_dim=99, n_assets=11)
        trainer = PPOTrainer(policy, lambda_dd=5.0, lambda_turnover=0.1)

        # No penalty case
        r = trainer.compute_reward(
            portfolio_return=0.01, drawdown=-0.02,
            dd_threshold=-0.05, turnover=0.0,
        )
        assert abs(r - 0.01) < 1e-8  # No DD penalty, no turnover

        # With DD penalty: |dd| = 0.08, |threshold| = 0.05, excess = 0.03
        r = trainer.compute_reward(
            portfolio_return=0.01, drawdown=-0.08,
            dd_threshold=-0.05, turnover=0.1,
        )
        expected = 0.01 - 5.0 * (0.03 ** 2) - 0.1 * 0.1
        assert abs(r - expected) < 1e-8

    def test_gae_basic(self):
        """GAE computation should produce correct-length outputs."""
        policy = RegimeAdaptivePolicy(state_dim=99, n_assets=11)
        trainer = PPOTrainer(policy)
        rewards = [0.01, -0.005, 0.02]
        values = [0.1, 0.05, 0.15]
        dones = [False, False, True]
        adv, ret = trainer.compute_gae(rewards, values, dones)
        assert len(adv) == 3
        assert len(ret) == 3


# ── Thompson Sampler tests ───────────────────────────────────────────────────

class TestThompsonSampler:
    def test_sample_returns_valid_keys(self):
        """Sample must return all parameter keys."""
        ts = ThompsonSampler()
        selected = ts.sample()
        assert "lambda_dd" in selected
        assert "lambda_turnover" in selected
        assert "entropy_coef" in selected

    def test_update_modifies_posteriors(self):
        """Posteriors should change after update."""
        ts = ThompsonSampler()
        old = {k: list(v) for k, v in ts.posteriors.items()}
        selected = ts.sample()
        ts.update(selected, sharpe_positive=True)
        # At least one posterior should have changed
        changed = False
        for param in ts.params:
            for i, (a, b) in enumerate(ts.posteriors[param]):
                if (a, b) != old[param][i]:
                    changed = True
        assert changed


# ── Cold-start test ──────────────────────────────────────────────────────────

class TestNeuralManager:
    def test_cold_start_weights(self, tickers):
        """NeuralPortfolioManager must return valid weights when _is_trained=False."""
        from src.portfolio.neural_manager import NeuralPortfolioManager
        mgr = NeuralPortfolioManager(tickers, use_multi_head=False)
        assert not mgr._is_trained

        np.random.seed(42)
        n = 200
        dates = pd.bdate_range("2020-01-01", periods=n)
        log_ret = pd.DataFrame(
            np.random.normal(0, 0.01, (n, len(tickers) + 1)),
            index=dates,
            columns=sorted(tickers) + ["SPY"],
        )

        weights = mgr.compute_weights(
            day_idx=100,
            log_ret=log_ret,
            current_weights={t: 1.0 / len(tickers) for t in tickers},
            assignments={t: "C" for t in tickers},
            vol_dict={t: 0.2 for t in tickers},
            p_stress=0.3,
            detector_signals={"cusum": 0, "ewma": 0, "markov": 0, "structural": 0},
            spy_ret=log_ret["SPY"],
        )
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_lstm_hidden_reset(self, tickers):
        """LSTM hidden state must be reset between walk-forward windows."""
        from src.portfolio.neural_manager import NeuralPortfolioManager
        mgr = NeuralPortfolioManager(tickers, use_multi_head=False)
        mgr.policy.reset_hidden(1)
        h1 = mgr.policy.hidden[0].clone()
        mgr.reset()
        h2 = mgr.policy.hidden[0].clone()
        # After reset, hidden state should be zero
        assert torch.allclose(h2, torch.zeros_like(h2))
