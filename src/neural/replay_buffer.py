"""
Experience replay buffer with exponential recency weighting.

Sampling probability for experience at time t when current time is T:
    p(t) ∝ exp(-β × (T − t) / 252)

With β = 0.5, experiences from ~1.4 years ago have half the sampling weight.
This allows the policy to adapt to evolving market regimes while retaining
longer-term memory of rare events (e.g., COVID crash).

References
----------
Schaul, T. et al. (2015). "Prioritized experience replay." arXiv:1511.05952.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Experience:
    """Single transition in the replay buffer."""

    state: np.ndarray           # state vector at time t
    action: np.ndarray          # portfolio weights chosen
    reward: float               # shaped reward for that day
    next_state: np.ndarray      # state vector at time t+1
    done: bool                  # True at end of walk-forward window
    log_prob: float             # log π(a|s) at time of action
    value: float                # V(s) estimate at time of action
    timestamp: int              # absolute day index (for recency weighting)


class RecencyWeightedBuffer:
    """Experience replay with exponential recency weighting."""

    def __init__(self, max_size: int = 50000, beta: float = 0.5) -> None:
        """
        Parameters
        ----------
        max_size : int
            Maximum number of experiences stored.
        beta : float
            Exponential decay rate for recency weighting.
        """
        self.buffer: list[Experience] = []
        self.max_size = max_size
        self.beta = beta

    def add(self, exp: Experience) -> None:
        """Add an experience to the buffer, evicting oldest if full."""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(exp)

    def sample(
        self, batch_size: int, current_timestamp: int
    ) -> list[Experience]:
        """Sample with exponential recency weighting.

        Parameters
        ----------
        batch_size : int
            Number of experiences to sample.
        current_timestamp : int
            Current absolute day index.

        Returns
        -------
        list[Experience]
        """
        if len(self.buffer) == 0:
            return []

        n = len(self.buffer)
        weights = np.array([
            np.exp(-self.beta * (current_timestamp - e.timestamp) / 252.0)
            for e in self.buffer
        ])
        # Normalise
        total = weights.sum()
        if total < 1e-12:
            weights = np.ones(n) / n
        else:
            weights /= total

        size = min(batch_size, n)
        indices = np.random.choice(n, size=size, replace=False, p=weights)
        return [self.buffer[i] for i in indices]

    def get_recent(self, n: int) -> list[Experience]:
        """Return the most recent n experiences (for GAE computation)."""
        return self.buffer[-n:] if n <= len(self.buffer) else list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()
