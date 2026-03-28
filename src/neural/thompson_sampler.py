"""
Thompson Sampling for online hyperparameter tuning.

Maintains Beta(α, β) posteriors for discrete hyperparameter choices.
At each walk-forward rebalance, samples from posteriors, selects the
configuration with the highest sample, and updates based on observed
performance (Sharpe > 0 or not).

This replaces the static grid search in BasketManager.calibrate_thresholds().
"""

from __future__ import annotations

import numpy as np


class ThompsonSampler:
    """Bayesian hyperparameter selection via Thompson sampling."""

    def __init__(self) -> None:
        self.params: dict[str, list[float]] = {
            "lambda_dd": [1.0, 2.0, 5.0, 10.0, 20.0],
            "lambda_turnover": [0.01, 0.05, 0.1, 0.5],
            "entropy_coef": [0.001, 0.005, 0.01, 0.05],
        }
        # Beta(α, β) posteriors for each (param, value) pair
        self.posteriors: dict[str, list[tuple[float, float]]] = {
            param: [(1.0, 1.0) for _ in values]
            for param, values in self.params.items()
        }

    def sample(self) -> dict[str, float]:
        """Thompson sample: draw from each Beta posterior, pick argmax.

        Returns
        -------
        dict
            Selected hyperparameter values plus internal index keys
            (prefixed with '_') for update().
        """
        selected: dict[str, float] = {}
        for param, values in self.params.items():
            samples = [
                np.random.beta(a, b)
                for a, b in self.posteriors[param]
            ]
            best_idx = int(np.argmax(samples))
            selected[param] = values[best_idx]
            selected[f"_{param}_idx"] = float(best_idx)
        return selected

    def update(self, selected: dict[str, float], sharpe_positive: bool) -> None:
        """Update Beta posteriors based on observed Sharpe.

        Parameters
        ----------
        selected : dict
            The output of sample() (contains '_*_idx' keys).
        sharpe_positive : bool
            Whether the selected configuration produced positive Sharpe.
        """
        for param in self.params:
            idx_key = f"_{param}_idx"
            if idx_key not in selected:
                continue
            idx = int(selected[idx_key])
            a, b = self.posteriors[param][idx]
            if sharpe_positive:
                self.posteriors[param][idx] = (a + 1.0, b)
            else:
                self.posteriors[param][idx] = (a, b + 1.0)

    def get_posterior_means(self) -> dict[str, list[float]]:
        """Return posterior means for each parameter value (for logging)."""
        return {
            param: [a / (a + b) for a, b in posteriors]
            for param, posteriors in self.posteriors.items()
        }
