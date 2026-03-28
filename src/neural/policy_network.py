"""
LSTM-based policy networks for portfolio weight generation.

Two architectures:
1. RegimeAdaptivePolicy — single LSTM → policy head + value head
2. MultiHeadPolicy — single LSTM → K gated policy heads + value head

The LSTM hidden state carries temporal context across days within each
walk-forward test window and is RESET at each rebalance point.

References
----------
Schulman, J. et al. (2017). "Proximal policy optimization algorithms."
arXiv:1707.06347.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class RegimeAdaptivePolicy(nn.Module):
    """Single-head LSTM policy network.

    Architecture:
        State (state_dim) → Linear(state_dim, 64) → ReLU
        → LSTM(64, hidden_dim, num_layers=2, dropout=0.1)
        → Policy head: Linear(hidden_dim, 64) → ReLU → Linear(64, n_assets) → Softmax
        → Value head: Linear(hidden_dim, 64) → ReLU → Linear(64, 1)
    """

    def __init__(
        self,
        state_dim: int = 99,
        n_assets: int = 11,
        hidden_dim: int = config.LSTM_HIDDEN_DIM,
        n_layers: int = config.LSTM_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_assets = n_assets

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            64, hidden_dim, num_layers=n_layers,
            dropout=0.1 if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_assets),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def reset_hidden(self, batch_size: int = 1) -> None:
        """Reset LSTM hidden state (call at start of each walk-forward window)."""
        device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        state : torch.Tensor
            Shape (batch, seq_len, state_dim).

        Returns
        -------
        weights : torch.Tensor
            Shape (batch, seq_len, n_assets) — portfolio weights summing to 1.
        values : torch.Tensor
            Shape (batch, seq_len, 1) — state value estimates.
        """
        x = self.encoder(state)
        if self.hidden is None:
            self.reset_hidden(state.size(0))
        # Detach hidden state to prevent backprop through time across episodes
        h = (self.hidden[0].detach(), self.hidden[1].detach())
        lstm_out, self.hidden = self.lstm(x, h)
        weights = F.softmax(self.policy_head(lstm_out), dim=-1)
        values = self.value_head(lstm_out)
        return weights, values


class MultiHeadPolicy(nn.Module):
    """K-head gated LSTM policy network.

    K parallel policy heads, each specialising in a different regime:
        Head 0 (Bull): momentum-riding, high-beta overweight
        Head 1 (Transition): gradual de-risking, balanced allocation
        Head 2 (Crisis): flight-to-quality, defensive sectors

    A gating network g(state) → softmax(R^K) produces mixture weights.
    Final portfolio weights = Σ_k g_k × policy_k(state).

    The gate weights are interpretable and stored for visualisation.
    """

    def __init__(
        self,
        state_dim: int = 99,
        n_assets: int = 11,
        hidden_dim: int = config.LSTM_HIDDEN_DIM,
        n_heads: int = config.N_POLICY_HEADS,
        n_layers: int = config.LSTM_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_assets = n_assets

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            64, hidden_dim, num_layers=n_layers,
            dropout=0.1 if n_layers > 1 else 0.0,
            batch_first=True,
        )

        # K separate policy heads
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, n_assets),
            )
            for _ in range(n_heads)
        ])

        # Gating network (soft regime classifier)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_heads),
        )

        # Shared value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_gate_weights: torch.Tensor | None = None

    def reset_hidden(self, batch_size: int = 1) -> None:
        """Reset LSTM hidden state."""
        device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device),
            torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        state : torch.Tensor
            Shape (batch, seq_len, state_dim).

        Returns
        -------
        weights : torch.Tensor
            Shape (batch, seq_len, n_assets).
        values : torch.Tensor
            Shape (batch, seq_len, 1).
        gate_weights : torch.Tensor
            Shape (batch, seq_len, n_heads).
        """
        x = self.encoder(state)
        if self.hidden is None:
            self.reset_hidden(state.size(0))
        h = (self.hidden[0].detach(), self.hidden[1].detach())
        lstm_out, self.hidden = self.lstm(x, h)

        # Gate → soft mixture over K heads
        gate_logits = self.gate(lstm_out)
        gate_weights = F.softmax(gate_logits, dim=-1)  # (batch, seq, K)
        self._last_gate_weights = gate_weights.detach()

        # Each head produces softmax weights independently
        head_outputs = torch.stack([
            F.softmax(head(lstm_out), dim=-1)
            for head in self.policy_heads
        ], dim=-2)  # (batch, seq, K, n_assets)

        # Mixture: sum over K heads weighted by gate
        # gate_weights: (batch, seq, K) → (batch, seq, K, 1)
        weights = (gate_weights.unsqueeze(-1) * head_outputs).sum(dim=-2)
        # weights: (batch, seq, n_assets), already sums to 1

        values = self.value_head(lstm_out)

        return weights, values, gate_weights

    @property
    def last_gate_weights(self) -> torch.Tensor | None:
        """Return last forward pass gate weights for interpretability."""
        return self._last_gate_weights
