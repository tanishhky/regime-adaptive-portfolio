"""
Neural portfolio manager — drop-in replacement for BasketManager.

Uses the LSTM-PPO policy (single-head or multi-head) to generate portfolio
weights.  Falls back to inverse-vol weights with basket-aware stress scaling
during cold-start (first COLD_START_WINDOWS walk-forward windows).

The manager maintains:
  - A replay buffer of experiences across walk-forward windows
  - A PPO trainer for periodic policy updates
  - A StandardScaler fitted on training data for state normalisation
  - An equity curve for drawdown computation in the reward function

References
----------
Schulman, J. et al. (2017). "Proximal policy optimization algorithms."
arXiv:1707.06347.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import config
from src.neural.state_builder import StateBuilder
from src.neural.replay_buffer import RecencyWeightedBuffer, Experience
from src.neural.policy_network import RegimeAdaptivePolicy, MultiHeadPolicy
from src.neural.rl_trainer import PPOTrainer
from src.portfolio.sizing import inverse_volatility_weights


class NeuralPortfolioManager:
    """Neural portfolio manager using LSTM-PPO policy."""

    def __init__(
        self,
        tickers: list[str],
        state_dim: int = 99,
        use_multi_head: bool = config.USE_MULTI_HEAD,
    ) -> None:
        self.tickers = sorted(tickers)
        self.n_assets = len(self.tickers)
        self.state_dim = state_dim
        self.use_multi_head = use_multi_head

        # Policy network
        if use_multi_head:
            self.policy: RegimeAdaptivePolicy | MultiHeadPolicy = MultiHeadPolicy(
                state_dim=state_dim, n_assets=self.n_assets,
            )
        else:
            self.policy = RegimeAdaptivePolicy(
                state_dim=state_dim, n_assets=self.n_assets,
            )
        self.policy.eval()

        # Trainer and buffer
        self.trainer = PPOTrainer(self.policy)
        self.buffer = RecencyWeightedBuffer(
            max_size=config.REPLAY_BUFFER_SIZE,
            beta=config.REPLAY_BETA,
        )

        # State builder and scaler
        self.state_builder = StateBuilder(self.tickers)
        self.scaler = StandardScaler()
        self._scaler_fitted = False

        # Training state
        self._is_trained = False
        self._windows_completed = 0
        self._dd_threshold: float = -0.05  # Will be calibrated from data

        # Running equity curve (for drawdown in reward)
        self._equity_curve: list[float] = [1.0]
        self._peak_equity: float = 1.0
        self._portfolio_returns: list[float] = []

        # Pending experience (filled after observing outcome)
        self._pending_state: np.ndarray | None = None
        self._pending_action: np.ndarray | None = None
        self._pending_log_prob: float = 0.0
        self._pending_value: float = 0.0
        self._pending_timestamp: int = 0

        # Gate weights storage (for multi-head visualisation)
        self.gate_weights_history: list[np.ndarray] = []

        # Days counter within current test window
        self._days_in_window: int = 0

    def calibrate(
        self,
        train_ret: pd.DataFrame,
        assignments: dict,
        vol_dict: dict[str, float],
        signal_matrix: np.ndarray,
        context_matrix: np.ndarray,
        spy_ret: pd.Series,
        vix_prices: pd.Series | None,
        rf_daily: float,
    ) -> None:
        """Called at each walk-forward rebalance.

        1. Fit the state scaler on training data
        2. Compute dd_threshold from training returns
        3. If buffer has enough data, run PPO update

        Parameters
        ----------
        train_ret : pd.DataFrame
            Training-window sector returns.
        assignments : dict
            Basket assignments.
        vol_dict : dict[str, float]
            GARCH volatilities.
        signal_matrix : np.ndarray
            (T_train × 4) detector signals.
        context_matrix : np.ndarray
            (T_train × 5) context features.
        spy_ret : pd.Series
            SPY returns (training window).
        vix_prices : pd.Series | None
            VIX close prices.
        rf_daily : float
            Daily risk-free rate.
        """
        # 1. Build training states for scaler fitting
        n_train = len(train_ret)
        states_for_fit = []
        n_signal = min(len(signal_matrix), n_train)

        for t in range(max(63, 1), n_signal):
            detector_sigs = {
                "cusum": float(signal_matrix[t, 0]),
                "ewma": float(signal_matrix[t, 1]),
                "markov": float(signal_matrix[t, 2]),
                "structural": float(signal_matrix[t, 3]),
            }
            # Use equal weights as proxy during calibration
            equal_w = {tkr: 1.0 / self.n_assets for tkr in self.tickers}
            state = self.state_builder.build(
                day_idx=t, log_ret=train_ret,
                current_weights=equal_w, assignments=assignments,
                vol_dict=vol_dict, p_stress=0.5,
                detector_signals=detector_sigs, spy_ret=spy_ret,
                vix_prices=vix_prices,
                portfolio_returns=None, days_since_rebalance=0,
            )
            states_for_fit.append(state)

        if len(states_for_fit) >= 30:
            self.scaler.fit(np.array(states_for_fit))
            self._scaler_fitted = True

        # 2. Compute dd_threshold from training cumulative returns
        spy_arr = spy_ret.dropna().values
        cum = np.cumsum(spy_arr)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        self._dd_threshold = float(np.percentile(dd, 10))

        # 3. Compute reward stats from training data for normalisation
        if len(spy_arr) > 63:
            train_rewards = []
            for t in range(63, len(spy_arr)):
                r = spy_arr[t]
                dd_t = cum[t] - peak[t]
                reward = self.trainer.compute_reward(r, dd_t, self._dd_threshold, 0.0)
                train_rewards.append(reward)
            self.trainer.update_reward_stats(train_rewards)

        # 4. Run PPO update if enough experience
        self._windows_completed += 1
        if self._windows_completed > config.COLD_START_WINDOWS and len(self.buffer) >= 252:
            self._is_trained = True
            self.trainer.update(self.buffer, self._pending_timestamp)

        self._days_in_window = 0

    def compute_weights(
        self,
        day_idx: int,
        log_ret: pd.DataFrame,
        current_weights: dict[str, float],
        assignments: dict,
        vol_dict: dict[str, float],
        p_stress: float,
        detector_signals: dict[str, float],
        spy_ret: pd.Series,
        vix_prices: pd.Series | None = None,
    ) -> dict[str, float]:
        """Generate portfolio weights for one trading day.

        Parameters
        ----------
        day_idx : int
            Current index in log_ret.
        Returns dict mapping ticker → weight, summing to ~1.0.
        """
        self._days_in_window += 1

        # Build state
        state = self.state_builder.build(
            day_idx=day_idx, log_ret=log_ret,
            current_weights=current_weights, assignments=assignments,
            vol_dict=vol_dict, p_stress=p_stress,
            detector_signals=detector_signals, spy_ret=spy_ret,
            vix_prices=vix_prices,
            portfolio_returns=self._portfolio_returns,
            days_since_rebalance=self._days_in_window,
        )

        # Scale
        if self._scaler_fitted:
            state_scaled = self.scaler.transform(state.reshape(1, -1)).flatten()
            state_scaled = np.clip(state_scaled, -5.0, 5.0)
        else:
            state_scaled = state

        # Store for experience recording
        self._pending_state = state_scaled.copy()
        self._pending_timestamp = day_idx

        if not self._is_trained:
            weights = self._fallback_weights(assignments, vol_dict, p_stress)
            # Record action for buffer even during fallback
            action = np.array([weights.get(t, 0.0) for t in self.tickers])
            self._pending_action = action
            self._pending_log_prob = 0.0
            self._pending_value = 0.0
            return weights

        # Neural policy inference
        state_tensor = torch.FloatTensor(state_scaled).unsqueeze(0).unsqueeze(0)
        self.policy.eval()
        with torch.no_grad():
            output = self.policy(state_tensor)
            if isinstance(self.policy, MultiHeadPolicy):
                weights_t, value_t, gate_t = output
                self.gate_weights_history.append(
                    gate_t.squeeze().numpy().copy()
                )
            else:
                weights_t, value_t = output

        weights_np = weights_t.squeeze().numpy()
        value_np = float(value_t.squeeze().item())

        # Compute log probability
        log_prob = float(
            (torch.FloatTensor(weights_np) * torch.log(weights_t.squeeze() + 1e-8)).sum().item()
        )

        self._pending_action = weights_np.copy()
        self._pending_log_prob = log_prob
        self._pending_value = value_np

        return {t: float(w) for t, w in zip(self.tickers, weights_np)}

    def record_outcome(
        self,
        portfolio_return: float,
        turnover: float,
        done: bool = False,
    ) -> None:
        """Record the outcome of the last weight decision.

        Called after each trading day to complete the experience tuple
        and add it to the replay buffer.

        Parameters
        ----------
        portfolio_return : float
            Realised daily portfolio return (net of costs).
        turnover : float
            Turnover for the trade.
        done : bool
            True at end of walk-forward test window.
        """
        # Update equity curve
        self._portfolio_returns.append(portfolio_return)
        new_equity = self._equity_curve[-1] * np.exp(portfolio_return)
        self._equity_curve.append(new_equity)
        self._peak_equity = max(self._peak_equity, new_equity)
        drawdown = (new_equity - self._peak_equity) / self._peak_equity

        # Compute shaped reward
        reward = self.trainer.compute_reward(
            portfolio_return, drawdown,
            self._dd_threshold, turnover,
        )
        reward = self.trainer.normalise_reward(reward)

        # Build next state (same as current pending state for simplicity —
        # the LSTM carries forward context)
        next_state = self._pending_state if self._pending_state is not None else np.zeros(self.state_dim)

        if self._pending_state is not None and self._pending_action is not None:
            exp = Experience(
                state=self._pending_state,
                action=self._pending_action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=self._pending_log_prob,
                value=self._pending_value,
                timestamp=self._pending_timestamp,
            )
            self.buffer.add(exp)

    def reset(self) -> None:
        """Reset LSTM hidden state (called at start of each test window)."""
        self.policy.reset_hidden(batch_size=1)
        self._days_in_window = 0

    def _fallback_weights(
        self,
        assignments: dict,
        vol_dict: dict[str, float],
        p_stress: float,
    ) -> dict[str, float]:
        """Inverse-vol weights with basket-aware stress scaling.

        Used during cold-start before the policy has enough training data.
        """
        weights: dict[str, float] = {}
        for tkr in self.tickers:
            vol = vol_dict.get(tkr, 0.2)
            inv_vol = 1.0 / max(vol, 1e-6)

            basket = self.state_builder._get_basket(assignments, tkr)
            if basket == "A":
                # Tactical: scale down with stress
                scale = max(0.0, 1.0 - p_stress)
            elif basket == "B":
                # Avoid: continuous de-risking
                scale = max(0.0, 1.0 - p_stress)
            else:
                # Core: hold
                scale = 1.0

            weights[tkr] = inv_vol * scale

        # Normalise
        total = sum(weights.values())
        if total > 0:
            weights = {t: w / total for t, w in weights.items()}
        else:
            equal = 1.0 / self.n_assets
            weights = {t: equal for t in self.tickers}

        return weights

    def save_checkpoint(self, path: str | Path) -> None:
        """Save policy, scaler, and buffer to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.policy.state_dict(), path / "policy.pt")
        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(path / "buffer.pkl", "wb") as f:
            pickle.dump(self.buffer, f)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load policy, scaler, and buffer from disk."""
        path = Path(path)
        self.policy.load_state_dict(torch.load(path / "policy.pt", weights_only=True))
        self.policy.eval()
        with open(path / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        self._scaler_fitted = True
        with open(path / "buffer.pkl", "rb") as f:
            self.buffer = pickle.load(f)
        self._is_trained = True
