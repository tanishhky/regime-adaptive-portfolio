"""
Hamilton (1989) Markov-Switching regime detector — medium time-scale (63-126 day).

Fits a 2-state Markov-Switching model on SPY weekly returns using
``statsmodels.tsa.regime_switching.markov_regression.MarkovRegression``.

**CRITICAL**: Uses filtered probability P(S_t = stress | r₁,…,r_t), NOT
smoothed probability (which uses future data and would introduce lookahead
bias).

References
----------
Hamilton, J.D. (1989). "A new approach to the economic analysis of
nonstationary time series and the business cycle." Econometrica, 57(2),
357-384.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


class MarkovSwitchingDetector:
    """Two-state Hamilton (1989) regime-switching detector."""

    def __init__(self) -> None:
        self.model_result = None
        self._transition_matrix: np.ndarray | None = None
        self._last_filtered_prob: float = 0.5

    def fit(self, returns: pd.Series) -> None:
        """Fit Markov-Switching model on weekly aggregated returns.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns (SPY) in the training window.
        """
        # Aggregate to weekly returns for medium-term regime detection
        daily = returns.dropna()
        weekly = daily.resample("W-FRI").sum().dropna()

        if len(weekly) < 52:
            # Not enough data for reliable estimation
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mod = MarkovRegression(
                    weekly.values,
                    k_regimes=2,
                    trend="c",
                    switching_variance=True,
                )
                self.model_result = mod.fit(maxiter=200, disp=False)
            except Exception:
                # Fallback: keep previous model or neutral signal
                return

        # Identify which state is the stress (high-vol) state
        # State with higher variance = stress
        params = self.model_result.params
        # MarkovRegression stores sigma parameters; regime with larger
        # variance is "stress"
        variances = self.model_result.sigma2_regimes if hasattr(
            self.model_result, "sigma2_regimes"
        ) else None

        # Use filtered probabilities (NO lookahead bias)
        filtered = self.model_result.filtered_marginal_probabilities
        self._transition_matrix = self.model_result.expected_durations

        # Determine stress state
        if variances is not None and len(variances) == 2:
            self._stress_state = int(np.argmax(variances))
        else:
            # Heuristic: state with higher mean filtered prob during
            # worst returns is stress
            self._stress_state = self._identify_stress_state(weekly, filtered)

        # Store last filtered probability for use in test window
        self._last_filtered_prob = float(
            filtered[self._stress_state].iloc[-1]
            if hasattr(filtered, "iloc")
            else filtered[self._stress_state][-1]
        )

    def _identify_stress_state(
        self, weekly: pd.Series, filtered: np.ndarray
    ) -> int:
        """Identify which HMM state corresponds to stress.

        Uses the correlation between filtered probability and negative
        returns: the state whose probability rises when returns are most
        negative is the stress state.
        """
        if hasattr(filtered, "values"):
            fp = filtered.values
        else:
            fp = filtered
        if fp.ndim == 2:
            corr0 = np.corrcoef(weekly.values[: fp.shape[0]], fp[:, 0])[0, 1]
            return 0 if corr0 < 0 else 1
        return 1

    @property
    def transition_matrix(self) -> np.ndarray | None:
        """Return the estimated transition matrix (for visualisation)."""
        if self.model_result is None:
            return None
        return np.array(self.model_result.regime_transition)

    def signal(self, r_t: float) -> float:
        """Return current stress probability.

        In the test window the model is not re-fit daily; instead the
        filtered probability is carried forward (a conservative approach
        that avoids lookahead).  The signal is updated at each walk-forward
        recalibration.

        Parameters
        ----------
        r_t : float
            Today's log return (unused between recalibrations, but kept
            in the API for consistency).

        Returns
        -------
        float
            P(stress) ∈ [0, 1].
        """
        return self._last_filtered_prob

    def signal_series(self, returns: pd.Series) -> pd.Series:
        """Return filtered stress probabilities for training data.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns.

        Returns
        -------
        pd.Series
            Daily P(stress) values (weekly values forward-filled to daily).
        """
        if self.model_result is None:
            return pd.Series(
                0.5, index=returns.index, name="markov_signal"
            )

        daily = returns.dropna()
        weekly = daily.resample("W-FRI").sum().dropna()

        filtered = self.model_result.filtered_marginal_probabilities
        stress_state = getattr(self, "_stress_state", 1)

        if hasattr(filtered, "iloc"):
            weekly_probs = pd.Series(
                filtered[stress_state].values,
                index=weekly.index[: len(filtered)],
            )
        else:
            weekly_probs = pd.Series(
                filtered[stress_state],
                index=weekly.index[: len(filtered[stress_state])],
            )

        # Forward-fill weekly probabilities to daily
        daily_probs = weekly_probs.reindex(daily.index, method="ffill")
        daily_probs = daily_probs.fillna(0.5)
        return daily_probs.rename("markov_signal")
