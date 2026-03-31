"""
Position sizing — inverse-volatility and risk-parity weighting.

References
----------
Kirby, C. & Ostdiek, B. (2012). "It's all in the timing: simple active
portfolio strategies that outperform naive diversification." Journal of
Financial and Quantitative Analysis, 47(2), 437-467.
"""

from __future__ import annotations

import numpy as np


def inverse_volatility_weights(
    vols: dict[str, float], tickers: list[str] | None = None
) -> dict[str, float]:
    """Compute inverse-volatility weights.

    w_i = (1/σ_i) / Σ(1/σ_j)

    Parameters
    ----------
    vols : dict[str, float]
        Annualised conditional volatility per ticker.
    tickers : list[str] | None
        Subset of tickers. Defaults to all keys in *vols*.

    Returns
    -------
    dict[str, float]
        Normalised weights summing to 1.
    """
    if tickers is None:
        tickers = list(vols.keys())

    inv = {t: 1.0 / max(vols[t], 1e-8) for t in tickers if t in vols}
    total = sum(inv.values())
    if total == 0:
        equal = 1.0 / len(tickers) if tickers else 0.0
        return {t: equal for t in tickers}
    return {t: v / total for t, v in inv.items()}


def risk_parity_weights(
    vols: dict[str, float], tickers: list[str] | None = None
) -> dict[str, float]:
    """Compute risk-parity weights: w_i * σ_i = constant for all i.

    Equivalent to: w_i = (1/σ_i) / Σ(1/σ_j) (same as inverse-vol for
    the diagonal covariance case).

    For the general case (with correlations), use numerical optimisation.
    This implementation assumes diagonal covariance (correlation = 0
    between assets), which is standard for sector-level risk parity.

    Parameters
    ----------
    vols : dict[str, float]
        Annualised conditional volatility per ticker.
    tickers : list[str] | None
        Subset of tickers.

    Returns
    -------
    dict[str, float]
        Normalised weights.

    References
    ----------
    Maillard, S., Roncalli, T. & Teïletche, J. (2010). "The properties
    of equally weighted risk contribution portfolios." Journal of
    Portfolio Management, 36(4), 60-70.
    """
    # Under diagonal covariance, risk parity = inverse volatility
    return inverse_volatility_weights(vols, tickers)
