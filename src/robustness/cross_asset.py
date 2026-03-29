"""
Cross-asset validation — test whether the regime-adaptive framework
generalizes beyond US sector ETFs.

Runs the walk-forward backtest on 3 different asset universes:
1. Multi-Asset (Stocks + Bonds + Commodities + Gold)
2. International Equity ETFs
3. Factor ETFs

The framework is asset-agnostic — it uses SPY as the stress signal
and applies basket classification to whatever ETFs are provided.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

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


MULTI_ASSET_ETFS = {
    "SPY": "US Equities",
    "TLT": "Long Treasuries",
    "IEF": "Intermediate Treasuries",
    "GLD": "Gold",
    "DBC": "Commodities",
    "VNQ": "Real Estate",
    "EFA": "International Developed",
    "EEM": "Emerging Markets",
}

INTL_ETFS = {
    "EFA": "EAFE",
    "EEM": "Emerging Markets",
    "EWJ": "Japan",
    "EWG": "Germany",
    "EWU": "UK",
    "FXI": "China",
    "EWZ": "Brazil",
    "EWY": "South Korea",
}

FACTOR_ETFS = {
    "MTUM": "Momentum",
    "VLUE": "Value",
    "QUAL": "Quality",
    "SIZE": "Size",
    "USMV": "Min Vol",
}

UNIVERSES = [
    ("Multi-Asset", MULTI_ASSET_ETFS, 504, "SPY"),
    ("International", INTL_ETFS, 504, "SPY"),
    ("Factor", FACTOR_ETFS, 252, "SPY"),
]


def _download_universe(
    universe_tickers: dict[str, str],
    benchmark: str,
    start: str = "2007-01-01",
    end: str = config.DATA_END_DATE,
) -> pd.DataFrame:
    """Download prices for a universe, including benchmark and VIX."""
    all_tickers = list(universe_tickers.keys())

    # Ensure benchmark and VIX are included
    if benchmark not in all_tickers:
        all_tickers.append(benchmark)
    vix = config.VIX_TICKER
    if vix not in all_tickers:
        all_tickers.append(vix)

    print(f"      Downloading: {', '.join(all_tickers)}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(all_tickers, start=start, end=end, auto_adjust=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Drop columns with too many NaNs
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))

    return prices


def _run_universe(
    name: str,
    universe_tickers: dict[str, str],
    min_train: int,
    benchmark: str,
    ff_data: pd.DataFrame,
) -> dict:
    """Run the walk-forward backtest on a specific universe."""
    print(f"\n    [{name}] Downloading data...")
    prices = _download_universe(universe_tickers, benchmark)

    # Filter to available tickers
    available = [t for t in universe_tickers.keys() if t in prices.columns]
    if not available:
        print(f"    [{name}] No tickers available, skipping.")
        return None

    print(f"    [{name}] Available: {', '.join(available)}")
    print(f"    [{name}] Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"    [{name}] Running walk-forward (min_train={min_train})...")

    engine = WalkForwardEngine(min_train=min_train, step=config.WALK_FORWARD_STEP)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            results = engine.run(
                prices,
                benchmark_ticker=benchmark,
                sector_tickers=available,
                ff_data=ff_data,
            )
        except Exception as e:
            print(f"    [{name}] Error: {e}")
            return None

    port_ret = results["strategy_returns"]
    if len(port_ret) == 0:
        print(f"    [{name}] No OOS data produced, skipping.")
        return None

    rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
    if ff_data is not None and "RF" in ff_data.columns:
        rf_daily = float(ff_data["RF"].reindex(port_ret.index).mean())

    m = compute_metrics(
        port_ret, rf_daily,
        turnover_series=results["turnover"],
        stress_signals=results["stress_signals"],
    )

    # Equal-weight benchmark for this universe
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
    ew_ret = log_ret[available].reindex(port_ret.index).mean(axis=1).fillna(0)
    m_ew = compute_metrics(ew_ret, rf_daily)

    # SPY benchmark
    bench_ret = results["benchmark_returns"]
    m_bench = compute_metrics(bench_ret, rf_daily)

    print(f"    [{name}] Strategy Sharpe: {m.sharpe_ratio:.2f}, "
          f"EW Sharpe: {m_ew.sharpe_ratio:.2f}, "
          f"SPY Sharpe: {m_bench.sharpe_ratio:.2f}")

    return {
        "name": name,
        "metrics": m,
        "ew_metrics": m_ew,
        "bench_metrics": m_bench,
        "strategy_returns": port_ret,
        "ew_returns": ew_ret,
        "bench_returns": bench_ret,
    }


def run_all_cross_asset(
    ff_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run cross-asset validation on all 3 universes.

    Parameters
    ----------
    ff_data : pd.DataFrame
        Fama-French factor data (for risk-free rate).

    Returns
    -------
    pd.DataFrame
        Results table.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    universe_results = {}

    for name, tickers, min_train, benchmark in UNIVERSES:
        result = _run_universe(name, tickers, min_train, benchmark, ff_data)
        if result is None:
            continue

        m = result["metrics"]
        m_ew = result["ew_metrics"]
        rows.append({
            "universe": name,
            "strategy_sharpe": m.sharpe_ratio,
            "strategy_return": m.annualised_return,
            "strategy_vol": m.annualised_volatility,
            "strategy_max_dd": m.max_drawdown,
            "strategy_calmar": m.calmar_ratio,
            "strategy_turnover": m.annualised_turnover,
            "ew_sharpe": m_ew.sharpe_ratio,
            "ew_return": m_ew.annualised_return,
            "ew_max_dd": m_ew.max_drawdown,
            "benchmark_sharpe": result["bench_metrics"].sharpe_ratio,
        })
        universe_results[name] = result

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "cross_asset_results.csv", index=False)
    print(f"\n    Saved: {OUTPUT_DIR / 'cross_asset_results.csv'}")

    if universe_results:
        _plot_cross_asset_equity_curves(universe_results)

    return df


def _plot_cross_asset_equity_curves(universe_results: dict[str, dict]) -> None:
    """One subplot per universe showing strategy vs EW vs SPY."""
    n = len(universe_results)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, (name, result) in zip(axes, universe_results.items()):
        # Strategy
        strat_cum = result["strategy_returns"].cumsum().apply(np.exp)
        ax.plot(strat_cum.index, strat_cum.values, color="#1a5276",
                linewidth=1.8, label="Regime-Adaptive")

        # Equal weight
        ew_cum = result["ew_returns"].cumsum().apply(np.exp)
        ax.plot(ew_cum.index, ew_cum.values, color="#b03a2e",
                linewidth=1.2, linestyle="--", label="Equal Weight")

        # SPY
        bench_cum = result["bench_returns"].cumsum().apply(np.exp)
        ax.plot(bench_cum.index, bench_cum.values, color="#7d8c8e",
                linewidth=1.2, linestyle=":", label="SPY")

        ax.set_title(f"{name}")
        ax.set_ylabel("Growth of $1")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Asset Generalization", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cross_asset_equity_curves.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'cross_asset_equity_curves.png'}")
