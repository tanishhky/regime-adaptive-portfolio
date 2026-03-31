"""
Regime generalization tests — check whether the framework overfits the sample.

5a: Sub-period analysis — report metrics for distinct market regimes
5b: Detector contribution stability — track fuzzy aggregator weights over time
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.backtest.walk_forward import WalkForwardEngine
from src.backtest.metrics import compute_metrics, PerformanceMetrics

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

PERIODS = {
    "2009-2012 (post-GFC recovery)": ("2009-01-01", "2012-12-31"),
    "2013-2016 (low-vol bull)": ("2013-01-01", "2016-12-31"),
    "2017-2019 (late cycle)": ("2017-01-01", "2019-12-31"),
    "2020-2022 (COVID + rate hikes)": ("2020-01-01", "2022-12-31"),
    "2023-2025 (recovery)": ("2023-01-01", "2025-12-31"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 5a: Sub-Period Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_period_analysis(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run the full walk-forward and report metrics by sub-period.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily Close prices.
    ff_data : pd.DataFrame
        Fama-French factor data.

    Returns
    -------
    pd.DataFrame
        Metrics for each sub-period.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run the full walk-forward backtest once
    print("    Running full walk-forward backtest...")
    engine = WalkForwardEngine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = engine.run(prices, ff_data=ff_data)

    port_ret = results["strategy_returns"]
    bench_ret = results["benchmark_returns"]

    rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
    if ff_data is not None and "RF" in ff_data.columns:
        rf_daily = float(ff_data["RF"].reindex(port_ret.index).mean())

    rows = []
    for period_name, (start, end) in PERIODS.items():
        mask = (port_ret.index >= start) & (port_ret.index <= end)
        if mask.sum() == 0:
            continue

        strat_sub = port_ret[mask]
        bench_sub = bench_ret[mask]

        m_strat = compute_metrics(strat_sub, rf_daily)
        m_bench = compute_metrics(bench_sub, rf_daily)

        rows.append({
            "period": period_name,
            "strategy_return": m_strat.annualised_return,
            "strategy_vol": m_strat.annualised_volatility,
            "strategy_sharpe": m_strat.sharpe_ratio,
            "strategy_max_dd": m_strat.max_drawdown,
            "spy_return": m_bench.annualised_return,
            "spy_vol": m_bench.annualised_volatility,
            "spy_sharpe": m_bench.sharpe_ratio,
            "spy_max_dd": m_bench.max_drawdown,
            "sharpe_diff": m_strat.sharpe_ratio - m_bench.sharpe_ratio,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "period_analysis.csv", index=False)
    print(f"\n    Saved: {OUTPUT_DIR / 'period_analysis.csv'}")

    # Plot
    _plot_period_analysis(df)

    return df


def _plot_period_analysis(df: pd.DataFrame) -> None:
    """Grouped bar chart: strategy vs SPY Sharpe by period."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df["strategy_sharpe"], width,
                    label="Regime-Adaptive", color="#1a5276")
    bars2 = ax.bar(x + width / 2, df["spy_sharpe"], width,
                    label="SPY", color="#7d8c8e")

    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sub-Period Sharpe Ratio: Strategy vs SPY")

    # Shorten period labels for readability
    labels = [p.split(" (")[0] for p in df["period"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "period_analysis.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'period_analysis.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5b: Detector Contribution Stability
# ═══════════════════════════════════════════════════════════════════════════════

class WeightTrackingEngine(WalkForwardEngine):
    """Walk-forward engine that records fuzzy aggregator weights at each rebalance."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight_history: list[dict] = []

    def run(
        self,
        prices: pd.DataFrame,
        benchmark_ticker: str = config.BENCHMARK,
        sector_tickers: list[str] | None = None,
        ff_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run walk-forward and record fuzzy weights at each rebalance.

        Hooks into the fuzzy aggregator's calibrate method to capture weights.
        """
        self.weight_history.clear()

        original_calibrate = self.fuzzy.calibrate

        def tracking_calibrate(signal_matrix, returns, **kwargs):
            original_calibrate(signal_matrix, returns, **kwargs)
            # Record weights after calibration
            self.weight_history.append({
                "cusum": float(self.fuzzy.weights[0]),
                "ewma": float(self.fuzzy.weights[1]),
                "markov": float(self.fuzzy.weights[2]),
                "structural": float(self.fuzzy.weights[3]),
            })

        self.fuzzy.calibrate = tracking_calibrate

        try:
            results = super().run(prices, benchmark_ticker, sector_tickers, ff_data)
        finally:
            self.fuzzy.calibrate = original_calibrate

        return results


def run_weight_stability(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> pd.DataFrame:
    """Track fuzzy aggregator weights across all rebalance dates.

    Returns
    -------
    pd.DataFrame
        Weight history indexed by rebalance date.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("    Running weight-tracking backtest...")
    engine = WeightTrackingEngine()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = engine.run(prices, ff_data=ff_data)

    # Build weight DataFrame
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    n_days = len(log_ret)
    rebalance_points = list(range(config.WALK_FORWARD_MIN_TRAIN, n_days, config.WALK_FORWARD_STEP))

    # Match weights to rebalance dates
    dates = [log_ret.index[rp] for rp in rebalance_points[:len(engine.weight_history)]]
    weight_df = pd.DataFrame(engine.weight_history, index=pd.DatetimeIndex(dates))

    print(f"    Tracked {len(weight_df)} rebalance windows")
    print(f"    Mean weights: {weight_df.mean().to_dict()}")

    # Plot stacked area chart
    _plot_weight_stability(weight_df)

    return weight_df


def _plot_weight_stability(weight_df: pd.DataFrame) -> None:
    """Stacked area chart of detector weights over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {
        "cusum": "#1a5276",
        "ewma": "#b03a2e",
        "markov": "#27ae60",
        "structural": "#8e44ad",
    }

    ax.stackplot(
        weight_df.index,
        weight_df["cusum"].values,
        weight_df["ewma"].values,
        weight_df["markov"].values,
        weight_df["structural"].values,
        labels=["CUSUM", "EWMA", "Markov", "Structural"],
        colors=[colors["cusum"], colors["ewma"],
                colors["markov"], colors["structural"]],
        alpha=0.8,
    )

    ax.set_ylabel("Fuzzy Aggregator Weight")
    ax.set_title("Detector Weight Stability Across Rebalance Windows")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weight_stability.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'weight_stability.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_regime_tests(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> None:
    """Run all regime generalization tests."""
    print("\n  [5a] Sub-period analysis...")
    period_df = run_period_analysis(prices, ff_data)
    print("\n" + period_df.to_string(index=False))

    print("\n  [5b] Detector weight stability...")
    weight_df = run_weight_stability(prices, ff_data)
    print("\n" + weight_df.to_string())
