"""
Baseline strategy comparisons.

Implements 4 alternative strategies to benchmark the regime-adaptive framework:
1. Volatility Targeting — scale exposure inversely with trailing realized vol
2. 200-Day Moving Average Timing — trend-following on SPY
3. Simple Risk Parity — inverse-vol weighting without regime awareness
4. Drawdown Control — reduce exposure proportional to trailing drawdown

All baselines use Close prices and apply 10 bps transaction costs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.backtest.metrics import compute_metrics, PerformanceMetrics
from src.backtest.walk_forward import WalkForwardEngine

# ── Plotting style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "robustness"
COST_BPS = config.TRANSACTION_COST_BPS
COST_FRAC = COST_BPS / 10_000.0


def _get_oos_range(prices: pd.DataFrame, ff_data: pd.DataFrame) -> pd.DatetimeIndex:
    """Get the OOS date range by running a quick check on the engine config."""
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    start_idx = config.WALK_FORWARD_MIN_TRAIN
    return log_ret.index[start_idx:]


def _compute_rf_series(
    index: pd.DatetimeIndex,
    ff_data: pd.DataFrame | None,
) -> pd.Series:
    """Get daily risk-free rate series."""
    rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
    if ff_data is not None and "RF" in ff_data.columns:
        return ff_data["RF"].reindex(index).fillna(rf_daily)
    return pd.Series(rf_daily, index=index)


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE 1: Volatility Targeting
# ═══════════════════════════════════════════════════════════════════════════════

def vol_targeting_strategy(
    prices: pd.DataFrame,
    sector_tickers: list[str],
    target_vol: float = 0.12,
    lookback: int = 63,
    ff_data: pd.DataFrame | None = None,
) -> pd.Series:
    """Scale sector ETF exposure inversely with trailing realized volatility.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily Close prices.
    sector_tickers : list[str]
        Sector ETF tickers.
    target_vol : float
        Annualized target portfolio volatility (default 12%).
    lookback : int
        Rolling window for realized vol (default 63 days).
    ff_data : pd.DataFrame | None
        Fama-French data for risk-free rate.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    available = [t for t in sector_tickers if t in log_ret.columns]
    sector_ret = log_ret[available]

    # Equal-weight portfolio returns
    ew_ret = sector_ret.mean(axis=1)

    # Rolling realized vol (annualized)
    rolling_vol = ew_ret.rolling(lookback).std() * np.sqrt(252)

    # OOS period
    oos_start = config.WALK_FORWARD_MIN_TRAIN
    oos_idx = log_ret.index[oos_start:]
    rf = _compute_rf_series(oos_idx, ff_data)

    port_returns = []
    prev_scale = 1.0

    for day in oos_idx:
        if day not in rolling_vol.index or np.isnan(rolling_vol.loc[day]):
            port_returns.append(0.0)
            continue

        rv = rolling_vol.loc[day]
        scale = target_vol / rv if rv > 1e-6 else 1.0
        scale = np.clip(scale, 0.0, 1.5)

        # Transaction cost on scale changes
        cost = abs(scale - prev_scale) * COST_FRAC

        # Portfolio return
        equity_ret = float(ew_ret.loc[day]) if day in ew_ret.index else 0.0
        cash_weight = max(0.0, 1.0 - scale)
        rf_day = float(rf.loc[day]) if day in rf.index else 0.0
        day_ret = scale * equity_ret + cash_weight * rf_day - cost

        port_returns.append(day_ret)
        prev_scale = scale

    return pd.Series(port_returns, index=oos_idx, name="vol_targeting")


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE 2: 200-Day Moving Average Timing
# ═══════════════════════════════════════════════════════════════════════════════

def ma_timing_strategy(
    prices: pd.DataFrame,
    sector_tickers: list[str],
    ma_window: int = 200,
    ff_data: pd.DataFrame | None = None,
) -> pd.Series:
    """Invest in sectors when SPY > 200-day SMA, move to cash when below.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily Close prices.
    sector_tickers : list[str]
        Sector ETF tickers.
    ma_window : int
        Moving average window (default 200 days).
    ff_data : pd.DataFrame | None
        Fama-French data for risk-free rate.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    available = [t for t in sector_tickers if t in log_ret.columns]
    sector_ret = log_ret[available]
    ew_ret = sector_ret.mean(axis=1)

    spy_price = prices[config.BENCHMARK]
    spy_sma = spy_price.rolling(ma_window).mean()

    oos_start = config.WALK_FORWARD_MIN_TRAIN
    oos_idx = log_ret.index[oos_start:]
    rf = _compute_rf_series(oos_idx, ff_data)

    port_returns = []
    prev_invested = True  # assume invested at start

    for day in oos_idx:
        if day not in spy_price.index or day not in spy_sma.index:
            port_returns.append(0.0)
            continue

        price = spy_price.loc[day]
        sma = spy_sma.loc[day]

        if np.isnan(sma):
            invested = prev_invested
        else:
            invested = price > sma

        # Transaction cost on regime changes
        cost = 0.0
        if invested != prev_invested:
            cost = COST_FRAC  # full portfolio turnover

        equity_ret = float(ew_ret.loc[day]) if day in ew_ret.index else 0.0
        rf_day = float(rf.loc[day]) if day in rf.index else 0.0

        if invested:
            day_ret = equity_ret - cost
        else:
            day_ret = rf_day - cost

        port_returns.append(day_ret)
        prev_invested = invested

    return pd.Series(port_returns, index=oos_idx, name="ma_timing")


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE 3: Simple Risk Parity
# ═══════════════════════════════════════════════════════════════════════════════

def risk_parity_strategy(
    prices: pd.DataFrame,
    sector_tickers: list[str],
    vol_lookback: int = 63,
    ff_data: pd.DataFrame | None = None,
) -> pd.Series:
    """Inverse-vol weighting with NO regime awareness.

    Rebalances quarterly (every 63 days) to match the main strategy cadence.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily Close prices.
    sector_tickers : list[str]
        Sector ETF tickers.
    vol_lookback : int
        Lookback for trailing vol (default 63 days).
    ff_data : pd.DataFrame | None
        Fama-French data for risk-free rate.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    available = [t for t in sector_tickers if t in log_ret.columns]
    sector_ret = log_ret[available]

    oos_start = config.WALK_FORWARD_MIN_TRAIN
    oos_idx = log_ret.index[oos_start:]

    port_returns = []
    current_weights = np.ones(len(available)) / len(available)
    rebal_counter = 0

    for day in oos_idx:
        day_loc = log_ret.index.get_loc(day)

        # Rebalance every 63 days
        if rebal_counter % 63 == 0:
            # Compute trailing vol
            window = sector_ret.iloc[max(0, day_loc - vol_lookback):day_loc]
            if len(window) >= 20:
                vols = window.std() * np.sqrt(252)
                vols = vols.fillna(vols.mean())
                inv_vol = 1.0 / np.maximum(vols.values, 1e-6)
                new_weights = inv_vol / inv_vol.sum()
            else:
                new_weights = np.ones(len(available)) / len(available)

            # Transaction cost
            turnover = np.abs(new_weights - current_weights).sum()
            cost = turnover * COST_FRAC
            current_weights = new_weights
        else:
            cost = 0.0

        # Portfolio return (always fully invested)
        day_rets = sector_ret.loc[day].values if day in sector_ret.index else np.zeros(len(available))
        day_ret = float(np.nansum(current_weights * day_rets)) - cost

        port_returns.append(day_ret)
        rebal_counter += 1

    return pd.Series(port_returns, index=oos_idx, name="risk_parity")


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE 4: Drawdown Control
# ═══════════════════════════════════════════════════════════════════════════════

def drawdown_control_strategy(
    prices: pd.DataFrame,
    sector_tickers: list[str],
    dd_threshold: float = -0.05,
    dd_floor: float = -0.15,
    ff_data: pd.DataFrame | None = None,
) -> pd.Series:
    """Reduce exposure proportionally when trailing drawdown exceeds threshold.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily Close prices.
    sector_tickers : list[str]
        Sector ETF tickers.
    dd_threshold : float
        Start reducing exposure at this drawdown (default -5%).
    dd_floor : float
        Fully in cash at this drawdown (default -15%).
    ff_data : pd.DataFrame | None
        Fama-French data for risk-free rate.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    available = [t for t in sector_tickers if t in log_ret.columns]
    sector_ret = log_ret[available]
    ew_ret = sector_ret.mean(axis=1)

    oos_start = config.WALK_FORWARD_MIN_TRAIN
    oos_idx = log_ret.index[oos_start:]
    rf = _compute_rf_series(oos_idx, ff_data)

    port_returns = []
    # Track drawdown of the BASE portfolio (equal-weight), not the
    # drawdown-controlled portfolio itself. Using the portfolio's own DD
    # creates a self-referential trap: once in cash, the portfolio can't
    # recover, so it stays in cash indefinitely.
    ew_cum = 0.0
    ew_peak = 0.0
    prev_exposure = 1.0

    for day in oos_idx:
        # Track equal-weight portfolio's drawdown (external signal)
        equity_ret = float(ew_ret.loc[day]) if day in ew_ret.index else 0.0
        ew_cum += equity_ret
        ew_peak = max(ew_peak, ew_cum)
        dd = ew_cum - ew_peak if ew_peak > 0 else 0.0

        # Compute exposure
        if dd > dd_threshold:
            exposure = 1.0
        elif dd < dd_floor:
            exposure = 0.0
        else:
            # Linear interpolation between threshold and floor
            exposure = (dd - dd_floor) / (dd_threshold - dd_floor)

        # Transaction cost on exposure changes
        cost = abs(exposure - prev_exposure) * COST_FRAC

        rf_day = float(rf.loc[day]) if day in rf.index else 0.0
        cash_weight = max(0.0, 1.0 - exposure)
        day_ret = exposure * equity_ret + cash_weight * rf_day - cost

        port_returns.append(day_ret)
        prev_exposure = exposure

    return pd.Series(port_returns, index=oos_idx, name="drawdown_control")


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_baselines(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run all baseline strategies and the regime-adaptive strategy.

    Returns
    -------
    pd.DataFrame
        Comparison table of all strategies.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sector_tickers = [t for t in config.SECTOR_ETFS.keys() if t in prices.columns]

    # Run regime-adaptive (main strategy)
    print("    Running regime-adaptive strategy...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine = WalkForwardEngine()
        results = engine.run(prices, ff_data=ff_data)

    ra_ret = results["strategy_returns"]
    oos_idx = ra_ret.index

    rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
    if ff_data is not None and "RF" in ff_data.columns:
        rf_daily = float(ff_data["RF"].reindex(oos_idx).mean())

    # Run baselines
    print("    Running vol targeting...")
    vt_ret = vol_targeting_strategy(prices, sector_tickers, ff_data=ff_data)
    print("    Running MA timing...")
    ma_ret = ma_timing_strategy(prices, sector_tickers, ff_data=ff_data)
    print("    Running risk parity...")
    rp_ret = risk_parity_strategy(prices, sector_tickers, ff_data=ff_data)
    print("    Running drawdown control...")
    dc_ret = drawdown_control_strategy(prices, sector_tickers, ff_data=ff_data)

    # SPY benchmark
    spy_ret = np.log(prices[config.BENCHMARK] / prices[config.BENCHMARK].shift(1))
    spy_ret = spy_ret.reindex(oos_idx).fillna(0)

    strategies = {
        "Regime-Adaptive": ra_ret,
        "Vol Targeting": vt_ret.reindex(oos_idx).fillna(0),
        "MA Timing (200d)": ma_ret.reindex(oos_idx).fillna(0),
        "Risk Parity": rp_ret.reindex(oos_idx).fillna(0),
        "Drawdown Control": dc_ret.reindex(oos_idx).fillna(0),
        "SPY (Buy & Hold)": spy_ret,
    }

    # Compute metrics for each
    rows = []
    for name, ret_series in strategies.items():
        turnover = results["turnover"] if name == "Regime-Adaptive" else None
        stress = results["stress_signals"] if name == "Regime-Adaptive" else None
        m = compute_metrics(ret_series, rf_daily,
                            turnover_series=turnover, stress_signals=stress)
        rows.append({
            "strategy": name,
            "ann_return": m.annualised_return,
            "ann_vol": m.annualised_volatility,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "calmar": m.calmar_ratio,
            "max_dd": m.max_drawdown,
            "max_dd_duration": m.max_drawdown_duration,
            "turnover": m.annualised_turnover,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)
    print(f"\n    Saved: {OUTPUT_DIR / 'baseline_comparison.csv'}")

    # Generate plots
    _plot_baseline_equity_curves(strategies)
    _plot_baseline_metrics_bars(df)

    return df


def _plot_baseline_equity_curves(strategies: dict[str, pd.Series]) -> None:
    """Plot cumulative returns for all strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "Regime-Adaptive": "#1a5276",
        "Vol Targeting": "#b03a2e",
        "MA Timing (200d)": "#27ae60",
        "Risk Parity": "#8e44ad",
        "Drawdown Control": "#d4ac0d",
        "SPY (Buy & Hold)": "#7d8c8e",
    }

    for name, ret_series in strategies.items():
        cum = ret_series.cumsum().apply(np.exp)
        style = "--" if name == "SPY (Buy & Hold)" else "-"
        lw = 2.0 if name == "Regime-Adaptive" else 1.5
        ax.plot(cum.index, cum.values, color=colors.get(name, "#333"),
                linewidth=lw, label=name, linestyle=style)

    ax.set_title("Strategy Comparison: Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_equity_curves.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'baseline_equity_curves.png'}")


def _plot_baseline_metrics_bars(df: pd.DataFrame) -> None:
    """Plot Sharpe / Calmar / Max DD as grouped bar charts."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = [
        ("sharpe", "Sharpe Ratio", "#1a5276"),
        ("calmar", "Calmar Ratio", "#27ae60"),
        ("max_dd", "Max Drawdown", "#b03a2e"),
    ]

    for ax, (col, title, color) in zip(axes, metrics_to_plot):
        bars = ax.barh(df["strategy"], df[col], color=color, alpha=0.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels
        for bar, val in zip(bars, df[col]):
            fmt = f"{val:.2f}" if col != "max_dd" else f"{val:.1%}"
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    fmt, va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_metrics_bars.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'baseline_metrics_bars.png'}")
