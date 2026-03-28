"""
Side-by-side comparison: Original BasketManager vs Neural Policy vs benchmarks.

Runs BOTH the original and neural pipelines on the same data, producing
a comparison table and overlay equity curves.

Usage:
    python run_comparison.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import config
from src.backtest.walk_forward import WalkForwardEngine
from src.backtest.metrics import compute_metrics, metrics_to_df


def main() -> None:
    print("=" * 60)
    print("Regime-Adaptive Portfolio — Comparison Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/5] Loading data...")
    try:
        prices = pd.read_csv("data/raw/etf_prices.csv", index_col=0, parse_dates=True)
        ff_data = pd.read_csv("data/raw/ff5_daily.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        from src.data.fetcher import fetch_etf_prices, fetch_fama_french
        prices = fetch_etf_prices()
        ff_data = fetch_fama_french()

    # Run original pipeline
    print("\n[2/5] Running original (fuzzy + basket) pipeline...")
    engine_orig = WalkForwardEngine(use_neural=False)
    results_orig = engine_orig.run(prices, ff_data=ff_data)

    # Run neural pipeline
    print("\n[3/5] Running neural (attention + LSTM-PPO) pipeline...")
    engine_neural = WalkForwardEngine(use_neural=True)
    results_neural = engine_neural.run(prices, ff_data=ff_data)

    # Compute metrics
    print("\n[4/5] Computing metrics...")
    rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
    if ff_data is not None and "RF" in ff_data.columns:
        idx = results_orig["strategy_returns"].index
        rf_daily = float(ff_data["RF"].reindex(idx).mean())

    metrics = {
        "Original (Fuzzy+Basket)": compute_metrics(
            results_orig["strategy_returns"], rf_daily,
            turnover_series=results_orig["turnover"],
            stress_signals=results_orig["stress_signals"],
        ),
        "Neural (Attention+PPO)": compute_metrics(
            results_neural["strategy_returns"], rf_daily,
            turnover_series=results_neural["turnover"],
            stress_signals=results_neural["stress_signals"],
        ),
        "SPY (Buy & Hold)": compute_metrics(
            results_orig["benchmark_returns"], rf_daily,
        ),
        "Equal-Weight Sector": compute_metrics(
            results_orig["equal_weight_returns"], rf_daily,
        ),
    }

    table = metrics_to_df(metrics)
    print("\n" + table.to_string())

    # Save
    output_dir = Path("output/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "comparison_metrics.csv")

    # Overlay equity curves
    print("\n[5/5] Generating comparison figures...")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    series_config = [
        (results_orig["strategy_returns"], "Original (Fuzzy+Basket)", "#e74c3c", 1.5),
        (results_neural["strategy_returns"], "Neural (Attention+PPO)", "#2c3e50", 1.8),
        (results_orig["benchmark_returns"], "SPY (Buy & Hold)", "#7f8c8d", 1.0),
        (results_orig["equal_weight_returns"], "Equal-Weight Sector", "#bdc3c7", 1.0),
    ]

    for series, label, color, lw in series_config:
        cum = series.cumsum()
        ax1.plot(cum.index, np.exp(cum.values) - 1,
                 label=label, linewidth=lw, color=color)

    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Walk-Forward Out-of-Sample: Original vs Neural")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(alpha=0.3)

    # Annotate metrics
    m_orig = metrics["Original (Fuzzy+Basket)"]
    m_neural = metrics["Neural (Attention+PPO)"]
    text = (
        f"Original — Sharpe: {m_orig.sharpe_ratio:.2f}, DD: {m_orig.max_drawdown:.1%}\n"
        f"Neural  — Sharpe: {m_neural.sharpe_ratio:.2f}, DD: {m_neural.max_drawdown:.1%}"
    )
    ax1.text(0.98, 0.05, text, transform=ax1.transAxes,
             fontsize=10, verticalalignment="bottom",
             horizontalalignment="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    # Drawdown comparison
    for series, label, color, _ in series_config[:2]:
        cum = series.cumsum()
        dd = cum - cum.cummax()
        ax2.plot(dd.index, dd.values, label=label, linewidth=1.2, color=color)

    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(alpha=0.3)

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=200)
    plt.close(fig)

    # Rolling Sharpe comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    for series, label, color, _ in series_config[:2]:
        rolling_mean = series.rolling(252).mean()
        rolling_std = series.rolling(252).std()
        rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252)).dropna()
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                label=label, linewidth=1.2, color=color)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Rolling 1-Year Sharpe")
    ax.set_title("Rolling Sharpe Ratio: Original vs Neural")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(output_dir / "rolling_sharpe_comparison.png", dpi=200)
    plt.close(fig)

    print(f"\n  Figures saved to {output_dir}/")
    print("=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
