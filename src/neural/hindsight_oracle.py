import numpy as np
import pandas as pd
from scipy.optimize import minimize

class HindsightOracle:
    """
    Computes the ex-post optimal portfolio weights for a past window.

    At time t, given realized returns over [t-lookback, t], solves:
        max_w  (w' @ μ_realized) - λ * w' @ Σ_realized @ w
        s.t.   Σ w_i = 1, w_i ≥ 0, w_cash ≥ 0

    This is a BACKWARD-looking quadratic program. No future data is used.
    The solution becomes the training target for imitation learning.
    """

    def __init__(self, lookback: int = 21, risk_aversion: float = 2.0,
                 max_single_weight: float = 0.35, min_cash: float = 0.0):
        self.lookback = lookback
        self.risk_aversion = risk_aversion
        self.max_single_weight = max_single_weight
        self.min_cash = min_cash

    def compute_optimal(self, realized_returns: pd.DataFrame,
                        include_cash: bool = True) -> np.ndarray:
        """
        Solve for optimal weights given REALIZED returns over the past window.

        Args:
            realized_returns: DataFrame (lookback × N_assets), daily log returns
                              that have ALREADY occurred (past data only)
            include_cash: if True, adds a cash asset with rf return

        Returns:
            optimal_weights: ndarray (N_assets + cash,) summing to 1.0
        """
        # Mean and covariance of REALIZED returns (these are historical facts, not forecasts)
        mu = realized_returns.mean().values * 252  # annualise
        Sigma = realized_returns.cov().values * 252
        n = len(mu)

        if include_cash:
            # Add cash as (n+1)-th asset with ~0 vol and rf return
            mu = np.append(mu, 0.0002)  # ~5% annual / 252
            row = np.zeros((1, n))
            col = np.zeros((n + 1, 1))
            Sigma = np.block([[Sigma, row.T], [row, np.array([[1e-8]])]])
            n += 1

        # Solve: max w'μ - λ/2 * w'Σw, s.t. Σw=1, 0 ≤ w ≤ max_weight
        def neg_utility(w):
            return -(w @ mu - 0.5 * self.risk_aversion * w @ Sigma @ w)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, self.max_single_weight)] * n
        if include_cash:
            bounds[-1] = (self.min_cash, 1.0)  # cash can go higher

        w0 = np.ones(n) / n
        result = minimize(neg_utility, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 500})

        return result.x if result.success else w0

    def generate_training_pairs(self, log_ret: pd.DataFrame,
                                feature_builder, day_indices: list[int]
                                ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate (state, oracle_weights) pairs for imitation learning.

        For each day_idx in day_indices:
          - state = feature_builder.build(day_idx - lookback)  <-- features at START of window
          - target = oracle.compute_optimal(log_ret[day_idx - lookback : day_idx])  <-- realized returns
        """
        pairs = []
        for day_idx in day_indices:
            if day_idx < self.lookback + 63:  # need enough history for features
                continue
            start = day_idx - self.lookback
            window_returns = log_ret.iloc[start:day_idx]
            if len(window_returns) < self.lookback - 5:
                continue

            # Need to pass dummy vars or extract from history since feature_builder.build takes many args
            # The strategy engine will pass these correctly when it integrates.
            # Assuming StrategyEngine intercepts and provides current weights etc.
            # BUT the interface in the prompt for this method is:
            # state = feature_builder.build(start, log_ret) # wait, prompt has 2 args, but the real one has ~11 args.
            # I will expect the engine to call build() and just pass states and returns here,
            # or we adapt this method to assume feature_builder is a callable wrapper.
            raise NotImplementedError("generate_training_pairs is handled directly within Engine to ensure state args")

        return pairs
