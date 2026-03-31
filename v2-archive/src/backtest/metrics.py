"""
Performance metrics for backtest evaluation.

Computes annualised return, volatility, Sharpe, Sortino, Calmar,
maximum drawdown, drawdown duration, turnover, hit rate, and false
alarm rate.

References
----------
Sharpe, W.F. (1964). "Capital asset prices." Journal of Finance, 19(3),
425-442.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for backtest performance statistics."""

    annualised_return: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int    # trading days
    annualised_turnover: float
    n_regime_transitions: int
    hit_rate: float               # % stress signals → actual drawdowns
    false_alarm_rate: float       # % stress signals → no drawdown


def compute_metrics(
    returns: pd.Series,
    risk_free_daily: float = 0.0,
    turnover_series: pd.Series | None = None,
    stress_signals: pd.Series | None = None,
    stress_threshold: float = 0.5,
    drawdown_threshold: float = -0.05,
    forward_window: int = 21,
) -> PerformanceMetrics:
    """Compute full suite of performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio log returns.
    risk_free_daily : float
        Daily risk-free rate.
    turnover_series : pd.Series | None
        Daily turnover values.
    stress_signals : pd.Series | None
        Daily composite P(stress) signal.
    stress_threshold : float
        Threshold above which a day is counted as a "stress signal."
    drawdown_threshold : float
        Minimum drawdown (negative) to qualify as "actual drawdown."
    forward_window : int
        Days to look ahead for drawdown realisation.

    Returns
    -------
    PerformanceMetrics
    """
    r = returns.dropna()
    n = len(r)
    if n < 2:
        return PerformanceMetrics(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        )

    # Annualised return (geometric)
    cum_ret = r.sum()  # log returns → sum is ln(P_T/P_0)
    years = n / 252.0
    ann_ret = cum_ret / years

    # Annualised volatility
    ann_vol = float(r.std() * np.sqrt(252))

    # Sharpe ratio
    excess = r - risk_free_daily
    sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0

    # Sortino ratio (downside deviation)
    downside = excess[excess < 0]
    downside_std = float(np.sqrt((downside ** 2).mean())) if len(downside) > 0 else 1e-8
    sortino = float(excess.mean() * np.sqrt(252) / downside_std)

    # Drawdown series
    cum_log = r.cumsum()
    peak = cum_log.cummax()
    dd = cum_log - peak

    # Maximum drawdown
    max_dd = float(dd.min())

    # Maximum drawdown duration
    dd_duration = _max_drawdown_duration(dd)

    # Calmar ratio
    calmar = float(ann_ret / abs(max_dd)) if abs(max_dd) > 1e-8 else 0.0

    # Annualised turnover
    if turnover_series is not None and len(turnover_series) > 0:
        ann_turnover = float(turnover_series.sum() / years)
    else:
        ann_turnover = 0.0

    # Regime transitions and hit/false alarm rate
    n_transitions = 0
    hit_rate = 0.0
    false_alarm_rate = 0.0
    if stress_signals is not None:
        aligned = stress_signals.reindex(r.index).fillna(0)
        binary = (aligned > stress_threshold).astype(int)
        n_transitions = int((binary.diff().abs() > 0).sum())

        # Hit rate: fraction of stress signals followed by actual drawdowns
        stress_days = binary[binary == 1].index
        hits = 0
        misses = 0
        for day in stress_days:
            loc = r.index.get_loc(day)
            end = min(loc + forward_window, n)
            if dd.iloc[loc:end].min() < drawdown_threshold:
                hits += 1
            else:
                misses += 1
        total_signals = hits + misses
        hit_rate = hits / total_signals if total_signals > 0 else 0.0
        false_alarm_rate = misses / total_signals if total_signals > 0 else 0.0

    return PerformanceMetrics(
        annualised_return=float(ann_ret),
        annualised_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        annualised_turnover=ann_turnover,
        n_regime_transitions=n_transitions,
        hit_rate=hit_rate,
        false_alarm_rate=false_alarm_rate,
    )


def _max_drawdown_duration(dd: pd.Series) -> int:
    """Compute the longest drawdown duration in trading days."""
    in_dd = False
    current_duration = 0
    max_duration = 0

    for val in dd.values:
        if val < -1e-8:
            current_duration += 1
            in_dd = True
        else:
            if in_dd:
                max_duration = max(max_duration, current_duration)
                current_duration = 0
                in_dd = False

    if in_dd:
        max_duration = max(max_duration, current_duration)

    return max_duration


def metrics_to_df(
    metrics_dict: dict[str, PerformanceMetrics],
) -> pd.DataFrame:
    """Convert a dict of PerformanceMetrics to a comparison DataFrame."""
    rows = {}
    for name, m in metrics_dict.items():
        rows[name] = {
            "Ann. Return": f"{m.annualised_return:.2%}",
            "Ann. Vol": f"{m.annualised_volatility:.2%}",
            "Sharpe": f"{m.sharpe_ratio:.2f}",
            "Sortino": f"{m.sortino_ratio:.2f}",
            "Calmar": f"{m.calmar_ratio:.2f}",
            "Max DD": f"{m.max_drawdown:.2%}",
            "Max DD Duration": f"{m.max_drawdown_duration}d",
            "Ann. Turnover": f"{m.annualised_turnover:.2f}",
            "Regime Transitions": m.n_regime_transitions,
            "Hit Rate": f"{m.hit_rate:.1%}",
            "False Alarm": f"{m.false_alarm_rate:.1%}",
        }
    return pd.DataFrame(rows).T
