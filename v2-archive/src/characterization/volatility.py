"""
GARCH(1,1)-t conditional volatility and VaR estimation.

For each sector ETF, fits a GARCH(1,1) model with Student-t innovations
using the ``arch`` Python library.  Extracts conditional volatility,
conditional VaR(99%), and persistence.

References
----------
Bollerslev, T. (1986). "Generalized autoregressive conditional
heteroskedasticity." Journal of Econometrics, 31(3), 307-327.

Engle, R.F. (1982). "Autoregressive conditional heteroscedasticity with
estimates of the variance of United Kingdom inflation." Econometrica,
50(4), 987-1007.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t as t_dist


@dataclass
class GARCHResult:
    """Container for GARCH estimation outputs."""

    conditional_vol: pd.Series        # annualised σ_t
    conditional_var99: pd.Series      # daily VaR(99%) in log-return units
    persistence: float                # α + β
    omega: float
    alpha: float
    beta: float
    nu: float                         # Student-t degrees of freedom
    last_vol: float                   # last annualised conditional σ


class GARCHVolatility:
    """GARCH(1,1)-t conditional volatility estimator."""

    def __init__(self) -> None:
        self._results: dict[str, GARCHResult] = {}

    def fit(self, returns: pd.Series, ticker: str = "") -> GARCHResult:
        """Fit GARCH(1,1)-t on a series of log returns.

        Parameters
        ----------
        returns : pd.Series
            Daily log returns (training window only).
        ticker : str
            Identifier for caching results.

        Returns
        -------
        GARCHResult
        """
        clean = returns.dropna()
        if len(clean) < 30:
            # Not enough data — return a neutral result
            vol_series = pd.Series(np.nan, index=clean.index)
            return GARCHResult(
                conditional_vol=vol_series, conditional_var99=vol_series,
                persistence=np.nan, omega=np.nan, alpha=np.nan, beta=np.nan,
                nu=np.nan, last_vol=np.nan,
            )

        # arch expects returns in percentage scale for numerical stability
        scaled = clean * 100.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                scaled,
                vol="Garch",
                p=1,
                q=1,
                dist="t",
                mean="Constant",
                rescale=False,
            )
            res = model.fit(disp="off", show_warning=False)

        params = res.params
        omega = float(params.get("omega", 0.0))
        alpha = float(params.get("alpha[1]", 0.0))
        beta = float(params.get("beta[1]", 0.0))
        nu = float(params.get("nu", 5.0))

        # Conditional variance series (in pct^2 → convert back)
        cond_var_pct2 = res.conditional_volatility ** 2
        cond_vol_daily = res.conditional_volatility / 100.0  # back to decimal
        cond_vol_annual = cond_vol_daily * np.sqrt(252)

        # Conditional VaR(99%) — Bollerslev (1986) with t-distribution
        # VaR = μ + σ_t * t_ν^{-1}(0.01) * sqrt((ν-2)/ν)
        mu = float(params.get("mu", 0.0)) / 100.0
        if nu > 2:
            scaling = np.sqrt((nu - 2.0) / nu)
            t_quantile = t_dist.ppf(0.01, df=nu)
        else:
            scaling = 1.0
            t_quantile = -2.326  # normal approx

        cond_var99 = mu + cond_vol_daily * t_quantile * scaling

        result = GARCHResult(
            conditional_vol=cond_vol_annual,
            conditional_var99=pd.Series(cond_var99, index=clean.index, name="VaR99"),
            persistence=alpha + beta,
            omega=omega,
            alpha=alpha,
            beta=beta,
            nu=nu,
            last_vol=float(cond_vol_annual.iloc[-1]),
        )

        if ticker:
            self._results[ticker] = result

        return result

    def get_result(self, ticker: str) -> GARCHResult | None:
        """Retrieve cached GARCH result for a ticker."""
        return self._results.get(ticker)

    def fit_all(
        self, returns_df: pd.DataFrame, tickers: list[str] | None = None
    ) -> dict[str, GARCHResult]:
        """Fit GARCH(1,1)-t on multiple tickers.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Daily log returns, columns = tickers.
        tickers : list[str] | None
            Subset of columns to fit.  Defaults to all.

        Returns
        -------
        dict[str, GARCHResult]
        """
        if tickers is None:
            tickers = list(returns_df.columns)
        for tkr in tickers:
            if tkr in returns_df.columns:
                self.fit(returns_df[tkr], ticker=tkr)
        return dict(self._results)
