"""
Proximal Policy Optimization (PPO) trainer for the regime-adaptive policy.

Reward function (per-day):
    r_t = portfolio_return_t
          − λ_dd × max(0, |drawdown_t| − dd_threshold)²
          − λ_turnover × turnover_t

References
----------
Schulman, J. et al. (2017). "Proximal policy optimization algorithms."
arXiv:1707.06347.

Schulman, J. et al. (2015). "High-dimensional continuous control using
generalized advantage estimation." arXiv:1506.02438.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neural.replay_buffer import RecencyWeightedBuffer, Experience

import config


class PPOTrainer:
    """PPO trainer for LSTM portfolio policy."""

    def __init__(
        self,
        policy: nn.Module,
        lr: float = config.POLICY_LR,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = config.PPO_CLIP_EPS,
        entropy_coef: float = config.PPO_ENTROPY_COEF,
        value_coef: float = 0.5,
        lambda_dd: float = config.LAMBDA_DD,
        lambda_turnover: float = config.LAMBDA_TURNOVER,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.policy = policy
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=lr, weight_decay=1e-5,
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.lambda_dd = lambda_dd
        self.lambda_turnover = lambda_turnover
        self.max_grad_norm = max_grad_norm

        # Running reward statistics for normalisation (training data only)
        self._reward_mean: float = 0.0
        self._reward_std: float = 1.0
        self._reward_count: int = 0

    def compute_reward(
        self,
        portfolio_return: float,
        drawdown: float,
        dd_threshold: float,
        turnover: float,
    ) -> float:
        """Compute shaped reward for one day.

        Parameters
        ----------
        portfolio_return : float
            Daily portfolio log return minus transaction cost.
        drawdown : float
            Current portfolio drawdown (negative value).
        dd_threshold : float
            Data-driven drawdown threshold (negative, from training).
        turnover : float
            Sum of absolute weight changes.

        Returns
        -------
        float
            Shaped reward.
        """
        dd_penalty = self.lambda_dd * max(0.0, abs(drawdown) - abs(dd_threshold)) ** 2
        turnover_penalty = self.lambda_turnover * turnover
        return portfolio_return - dd_penalty - turnover_penalty

    def update_reward_stats(self, rewards: list[float]) -> None:
        """Update running reward mean/std from training-window rewards."""
        if not rewards:
            return
        arr = np.array(rewards)
        self._reward_mean = float(arr.mean())
        self._reward_std = float(arr.std()) if arr.std() > 1e-8 else 1.0
        self._reward_count = len(rewards)

    def normalise_reward(self, reward: float) -> float:
        """Normalise reward by running statistics (training data only)."""
        if self._reward_std < 1e-8:
            return reward
        return (reward - self._reward_mean) / self._reward_std

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
    ) -> tuple[list[float], list[float]]:
        """Compute Generalized Advantage Estimation (Schulman et al., 2015).

        Parameters
        ----------
        rewards : list[float]
            Per-step rewards.
        values : list[float]
            Per-step V(s) estimates.
        dones : list[bool]
            Per-step done flags.

        Returns
        -------
        advantages : list[float]
        returns : list[float]
        """
        advantages = []
        returns = []
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0.0
                gae = 0.0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]

        return advantages, returns

    def update(
        self,
        buffer: RecencyWeightedBuffer,
        current_timestamp: int,
        n_epochs: int = config.PPO_N_EPOCHS,
        batch_size: int = config.PPO_BATCH_SIZE,
    ) -> dict[str, float]:
        """Run PPO update on buffered experiences.

        Parameters
        ----------
        buffer : RecencyWeightedBuffer
            Experience replay buffer.
        current_timestamp : int
            Current absolute day index.
        n_epochs : int
            Number of PPO update epochs.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        dict[str, float]
            Loss components for logging.
        """
        if len(buffer) < 252:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        self.policy.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(n_epochs):
            experiences = buffer.sample(batch_size, current_timestamp)
            if not experiences:
                continue

            # Prepare tensors
            states = torch.FloatTensor(
                np.stack([e.state for e in experiences])
            ).unsqueeze(1)  # (B, 1, state_dim)
            old_log_probs = torch.FloatTensor(
                [e.log_prob for e in experiences]
            )
            old_values = torch.FloatTensor(
                [e.value for e in experiences]
            )
            actions = torch.FloatTensor(
                np.stack([e.action for e in experiences])
            )

            # Compute GAE for this batch
            rewards_list = [e.reward for e in experiences]
            values_list = [e.value for e in experiences]
            dones_list = [e.done for e in experiences]
            advantages, returns_list = self.compute_gae(
                rewards_list, values_list, dones_list
            )
            advantages_t = torch.FloatTensor(advantages)
            returns_t = torch.FloatTensor(returns_list)

            # Normalise advantages
            if advantages_t.std() > 1e-8:
                advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            # Forward pass (reset hidden for batch processing)
            self.policy.reset_hidden(len(experiences))
            output = self.policy(states)
            if len(output) == 3:
                new_weights, new_values, _ = output
            else:
                new_weights, new_values = output
            new_weights = new_weights.squeeze(1)  # (B, n_assets)
            new_values = new_values.squeeze(1).squeeze(-1)  # (B,)

            # Compute log probabilities under current policy
            # Treat portfolio weights as a Dirichlet-like distribution
            # log π(a|s) ≈ Σ a_i * log(π_i) (categorical-like approximation)
            new_log_probs = (actions * torch.log(new_weights + 1e-8)).sum(dim=-1)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values, returns_t)

            # Entropy bonus (encourage exploration)
            entropy = -(new_weights * torch.log(new_weights + 1e-8)).sum(dim=-1).mean()

            # Total loss
            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

        self.policy.eval()

        if n_updates == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }
