"""
Full pipeline runner — downloads data, runs walk-forward backtest,
generates figures and performance tables.

Usage:
    python run_pipeline.py
"""

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

import config
from src.data.fetcher import fetch_etf_prices, fetch_fama_french, compute_log_returns
from src.backtest.walk_forward import WalkForwardEngine
from src.backtest.metrics import compute_metrics, metrics_to_df
from src.visualization.plots import generate_all_figures
from src.characterization.volatility import GARCHVolatility
from src.characterization.recovery import RecoveryEstimator
from src.characterization.classifier import BasketClassifier


def main() -> None:
    print("=" * 60)
    print("Regime-Adaptive Portfolio — Full Pipeline")
    print("=" * 60)

    # ── 1. Data acquisition ───────────────────────────────────────────
    print("\n[1/5] Fetching data...")
    try:
        prices = pd.read_csv("data/raw/etf_prices.csv", index_col=0, parse_dates=True)
        ff_data = pd.read_csv("data/raw/ff5_daily.csv", index_col=0, parse_dates=True)
        print("  Loaded cached data.")
    except FileNotFoundError:
        prices = fetch_etf_prices()
        ff_data = fetch_fama_french()
        print(f"  Downloaded {prices.shape[0]} days × {prices.shape[1]} tickers.")

    log_ret = compute_log_returns(prices)
    sector_tickers = [t for t in config.SECTOR_ETFS.keys() if t in prices.columns]

    # ── 2. Walk-forward backtest ──────────────────────────────────────
    print("\n[2/5] Running walk-forward backtest...")
    engine = WalkForwardEngine()
    results = engine.run(prices, ff_data=ff_data)

    port_ret = results["strategy_returns"]
    bench_ret = results["benchmark_returns"]
    print(f"  Backtest period: {port_ret.index[0].date()} to {port_ret.index[-1].date()}")
    print(f"  Total OOS days: {len(port_ret)}")

    # ── 3. Compute metrics ────────────────────────────────────────────
    print("\n[3/5] Computing performance metrics...")
    rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
    if ff_data is not None and "RF" in ff_data.columns:
        rf_daily = float(ff_data["RF"].reindex(port_ret.index).mean())

    m_strategy = compute_metrics(
        port_ret, rf_daily,
        turnover_series=results["turnover"],
        stress_signals=results["stress_signals"],
    )
    m_benchmark = compute_metrics(bench_ret, rf_daily)

    metrics_table = metrics_to_df({
        "Regime-Adaptive": m_strategy,
        "SPY (Buy & Hold)": m_benchmark,
    })
    print("\n" + metrics_table.to_string())

    # Save table
    from pathlib import Path
    tables_dir = Path("output/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(tables_dir / "performance_metrics.csv")

    # ── 4. Asset characterization (for figures) ───────────────────────
    print("\n[4/5] Characterizing assets for visualization...")
    # Use full training window up to start of OOS for final characterization
    train_end = min(config.WALK_FORWARD_MIN_TRAIN, len(log_ret))
    train_ret = log_ret.iloc[:train_end]

    garch = GARCHVolatility()
    avail = [t for t in sector_tickers if train_ret[t].dropna().shape[0] >= 30]
    garch_results = garch.fit_all(train_ret, avail)

    recovery = RecoveryEstimator()
    recovery_results = recovery.estimate_all(train_ret, avail)

    classifier = BasketClassifier()
    assignments = classifier.assign(garch_results, recovery_results)

    # Compute boundaries for scatter plot
    half_lives = [r.half_life for r in recovery_results.values() if np.isfinite(r.half_life)]
    half_life_median = float(np.median(half_lives)) if half_lives else 5.0
    var99s = [float(g.conditional_var99.iloc[-1]) for g in garch_results.values()]
    cvar_75 = float(np.percentile(var99s, 25)) if var99s else -0.05

    # ── 5. Generate figures ───────────────────────────────────────────
    print("\n[5/5] Generating figures...")
    generate_all_figures(
        results, prices,
        fuzzy_aggregator=engine.fuzzy,
        markov_detector=engine.markov,
        assignments=assignments,
        half_life_median=half_life_median,
        cvar_75=cvar_75,
        sector_tickers=sector_tickers,
    )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Figures: output/figures/")
    print(f"  Tables:  output/tables/")
    print("=" * 60)


if __name__ == "__main__":
    main()
