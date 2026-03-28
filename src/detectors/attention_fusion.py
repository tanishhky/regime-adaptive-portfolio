"""
Attention-based detector fusion — replaces the Takagi-Sugeno fuzzy aggregator.

Uses a temporal cross-attention mechanism (2-layer TransformerEncoder) that
learns which detectors to trust in different market contexts and how far back
to look.

Architecture:
    Input: (lookback × 9) — 4 detector signals + 5 context features
    → Sinusoidal positional encoding
    → 2-layer TransformerEncoder (d_model=32, nhead=2, dim_ff=64, dropout=0.1)
    → Last timestep (causal: only attends to past)
    → Linear(32 → 1) → Sigmoid → p_stress ∈ [0,1]

CRITICAL: Uses causal masking so position t only attends to positions ≤ t.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (not learned)."""

    def __init__(self, d_model: int, max_len: int = 500) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class _AttentionModel(nn.Module):
    """Transformer-based detector fusion model."""

    def __init__(
        self, input_dim: int = 9, d_model: int = 32,
        nhead: int = 2, dim_ff: int = 64, n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _SinusoidalPE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
        )
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        self._last_attn_weights: np.ndarray | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, lookback, input_dim).

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) — p_stress for the last timestep.
        """
        seq_len = x.size(1)
        # Causal mask: position t attends only to positions ≤ t
        mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )

        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h, mask=mask)
        # Take last timestep only
        out = self.output_head(h[:, -1, :])
        return out


class AttentionFusion:
    """Attention-based detector fusion (drop-in replacement for FuzzyAggregator).

    Parameters
    ----------
    n_detectors : int
        Number of detector signals (4).
    n_context : int
        Number of context features (5).
    lookback : int
        Number of days of signal history for attention.
    """

    N_DETECTORS = 4

    def __init__(
        self,
        n_detectors: int = 4,
        n_context: int = 5,
        lookback: int = config.ATTENTION_LOOKBACK,
    ) -> None:
        self.n_detectors = n_detectors
        self.n_context = n_context
        self.lookback = lookback
        self.input_dim = n_detectors + n_context

        self.model = _AttentionModel(input_dim=self.input_dim)
        self._last_attention_weights: np.ndarray | None = None

    def calibrate(
        self,
        signal_matrix: np.ndarray,
        context_matrix: np.ndarray,
        returns: pd.Series,
        drawdown_pct: float = 10.0,
        forward_window: int = 21,
    ) -> None:
        """Train the attention model on the training window.

        Parameters
        ----------
        signal_matrix : np.ndarray
            Shape (T, 4) — daily detector signals.
        context_matrix : np.ndarray
            Shape (T, 5) — daily context features.
        returns : pd.Series
            Daily log returns (for drawdown target computation).
        drawdown_pct : float
            Percentile threshold for significant drawdowns.
        forward_window : int
            Days to look ahead for target computation.
        """
        # Build target: did drawdown exceed threshold within forward_window?
        ret_arr = returns.values
        prices = np.exp(np.cumsum(ret_arr))
        peak = np.maximum.accumulate(prices)
        dd = (prices - peak) / peak

        dd_threshold = np.percentile(dd, drawdown_pct)
        T = len(dd)
        target = np.zeros(T)
        for t in range(T - forward_window):
            if np.min(dd[t: t + forward_window]) < dd_threshold:
                target[t] = 1.0

        # Build windowed training samples
        n = min(len(signal_matrix), len(context_matrix), T)
        features = np.concatenate(
            [signal_matrix[:n], context_matrix[:n]], axis=1,
        )  # (n, 9)

        X_windows = []
        y_labels = []
        for t in range(self.lookback, n):
            window = features[t - self.lookback: t]
            X_windows.append(window)
            y_labels.append(target[t])

        if len(X_windows) < 30:
            return

        X = torch.FloatTensor(np.array(X_windows))    # (N, lookback, 9)
        y = torch.FloatTensor(np.array(y_labels)).unsqueeze(1)  # (N, 1)

        # Train/val split (last 20% for early stopping)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Training
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4,
        )
        criterion = nn.BCELoss()
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10
        best_state = None

        for epoch in range(50):
            # Train
            optimizer.zero_grad()
            pred = self.model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            # Validate
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def aggregate(
        self,
        signal_history: np.ndarray,
        context_history: np.ndarray,
    ) -> float:
        """Compute p_stress from signal+context history.

        Parameters
        ----------
        signal_history : np.ndarray
            Shape (lookback, 4) or shorter (will be zero-padded).
        context_history : np.ndarray
            Shape (lookback, 5) or shorter.

        Returns
        -------
        float
            P(stress) ∈ [0, 1].
        """
        # Pad if history is shorter than lookback
        sig = self._pad_to_lookback(signal_history, self.n_detectors)
        ctx = self._pad_to_lookback(context_history, self.n_context)

        features = np.concatenate([sig, ctx], axis=1)  # (lookback, 9)
        x = torch.FloatTensor(features).unsqueeze(0)   # (1, lookback, 9)

        self.model.eval()
        with torch.no_grad():
            p = self.model(x)
        return float(p.item())

    def aggregate_series(
        self,
        signal_matrix: np.ndarray,
        context_matrix: np.ndarray,
    ) -> np.ndarray:
        """Vectorised aggregation for a full series.

        Parameters
        ----------
        signal_matrix : np.ndarray
            Shape (T, 4).
        context_matrix : np.ndarray
            Shape (T, 5).

        Returns
        -------
        np.ndarray
            Shape (T,) — composite p_stress per day.
        """
        T = min(len(signal_matrix), len(context_matrix))
        result = np.zeros(T)

        for t in range(T):
            start = max(0, t + 1 - self.lookback)
            sig_hist = signal_matrix[start: t + 1]
            ctx_hist = context_matrix[start: t + 1]
            result[t] = self.aggregate(sig_hist, ctx_hist)

        return result

    def get_attention_weights(self) -> np.ndarray | None:
        """Return last attention weights for interpretability."""
        return self._last_attention_weights

    def _pad_to_lookback(self, arr: np.ndarray, width: int) -> np.ndarray:
        """Zero-pad array to (lookback, width) if shorter."""
        if len(arr) >= self.lookback:
            return arr[-self.lookback:]
        padded = np.zeros((self.lookback, width))
        padded[-len(arr):] = arr
        return padded
