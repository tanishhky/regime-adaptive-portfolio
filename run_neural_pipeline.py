"""
Neural-enhanced pipeline runner.

Runs the walk-forward backtest with the LSTM-PPO policy, attention-based
detector fusion, and Thompson sampling for hyperparameter tuning.

Usage:
    python run_neural_pipeline.py
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


def _plot_gate_weights(gate_weights: list[np.ndarray], dates: pd.DatetimeIndex,
                       output_dir: Path) -> None:
    """Plot regime gate weights over time."""
    if not gate_weights:
        return
    n = min(len(gate_weights), len(dates))
    gw = np.array(gate_weights[:n])
    if gw.ndim != 2:
        return
    fig, ax = plt.subplots(figsize=(16, 4))
    labels = ["Bull", "Transition", "Crisis"]
    colors = ["#27ae60", "#f39c12", "#e74c3c"]
    for k in range(gw.shape[1]):
        label = labels[k] if k < len(labels) else f"Head {k}"
        color = colors[k] if k < len(colors) else None
        ax.fill_between(dates[:n], 0, gw[:, k],
                        alpha=0.4 if k == 0 else 0.5,
                        label=label, color=color)
    ax.set_ylabel("Gate Weight")
    ax.set_xlabel("Date")
    ax.set_title("Regime-Conditioned Policy Head Activation")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(output_dir / "gate_weights.png", dpi=200)
    plt.close(fig)


def _plot_weight_evolution(daily_weights: list[dict], dates: pd.DatetimeIndex,
                           tickers: list[str], output_dir: Path) -> None:
    """Stacked area chart of portfolio weights over time."""
    n = min(len(daily_weights), len(dates))
    df = pd.DataFrame(daily_weights[:n], index=dates[:n])
    # Keep only sector tickers
    cols = [c for c in tickers if c in df.columns]
    if not cols:
        return
    df = df[cols].fillna(0)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.stackplot(df.index, df.values.T, labels=cols, alpha=0.8)
    ax.set_ylabel("Portfolio Weight")
    ax.set_xlabel("Date")
    ax.set_title("Portfolio Weight Evolution (Neural Policy)")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(output_dir / "weight_evolution.png", dpi=200)
    plt.close(fig)


def _plot_reward_components(portfolio_returns: pd.Series, turnover: pd.Series,
                            output_dir: Path) -> None:
    """Visualise reward components over time."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)

    # Returns
    cum_ret = portfolio_returns.cumsum()
    axes[0].plot(cum_ret.index, np.exp(cum_ret.values) - 1, color="#2c3e50", linewidth=1)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].set_title("Neural Policy: Return Components")
    axes[0].grid(alpha=0.3)

    # Drawdown
    peak = cum_ret.cummax()
    dd = cum_ret - peak
    axes[1].fill_between(dd.index, 0, dd.values, alpha=0.5, color="#e74c3c")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)

    # Turnover
    roll_turnover = turnover.rolling(21).mean()
    axes[2].plot(roll_turnover.index, roll_turnover.values, color="#3498db", linewidth=1)
    axes[2].set_ylabel("21-Day Rolling\nAvg Turnover")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)

    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(output_dir / "reward_components.png", dpi=200)
    plt.close(fig)


def main() -> None:
    print("=" * 60)
    print("Neural Regime-Adaptive Portfolio — Pipeline")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    try:
        prices = pd.read_csv("data/raw/etf_prices.csv", index_col=0, parse_dates=True)
        ff_data = pd.read_csv("data/raw/ff5_daily.csv", index_col=0, parse_dates=True)
        print("  Loaded cached data.")
    except FileNotFoundError:
        from src.data.fetcher import fetch_etf_prices, fetch_fama_french
        prices = fetch_etf_prices()
        ff_data = fetch_fama_french()

    # Run neural walk-forward
    print("\n[2/4] Running neural walk-forward backtest...")
    engine = WalkForwardEngine(use_neural=True)
    results = engine.run(prices, ff_data=ff_data)

    port_ret = results["strategy_returns"]
    bench_ret = results["benchmark_returns"]
    ew_ret = results["equal_weight_returns"]
    print(f"  Period: {port_ret.index[0].date()} to {port_ret.index[-1].date()}")
    print(f"  OOS days: {len(port_ret)}")

    # Metrics
    print("\n[3/4] Computing performance metrics...")
    rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
    if ff_data is not None and "RF" in ff_data.columns:
        rf_daily = float(ff_data["RF"].reindex(port_ret.index).mean())

    m_neural = compute_metrics(
        port_ret, rf_daily,
        turnover_series=results["turnover"],
        stress_signals=results["stress_signals"],
    )
    m_bench = compute_metrics(bench_ret, rf_daily)
    m_ew = compute_metrics(ew_ret, rf_daily)

    table = metrics_to_df({
        "Neural Policy": m_neural,
        "SPY (Buy & Hold)": m_bench,
        "Equal-Weight Sector": m_ew,
    })
    print("\n" + table.to_string())

    # Save
    output_dir = Path("output/neural")
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "performance_metrics.csv")

    # Figures
    print("\n[4/4] Generating neural-specific figures...")

    sector_tickers = sorted(config.SECTOR_ETFS.keys())
    _plot_gate_weights(
        results.get("gate_weights_history", []),
        port_ret.index,
        output_dir,
    )
    _plot_weight_evolution(
        results["daily_weights"], port_ret.index,
        sector_tickers, output_dir,
    )
    _plot_reward_components(port_ret, results["turnover"], output_dir)

    # Backtest equity curve
    from src.visualization.plots import plot_backtest
    fig = plot_backtest(port_ret, bench_ret, ew_ret, m_neural, save=False)
    fig.savefig(output_dir / "backtest.png", dpi=200)
    plt.close(fig)

    print(f"\n  Figures saved to {output_dir}/")
    print("=" * 60)
    print("Neural pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
