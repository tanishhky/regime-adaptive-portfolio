"""
Run all robustness experiments.

Usage:
    python run_robustness.py              # Run everything
    python run_robustness.py ablation     # Run only ablation
    python run_robustness.py baselines    # Run only baselines
    python run_robustness.py sensitivity  # Run only sensitivity
    python run_robustness.py cross_asset  # Run only cross-asset
    python run_robustness.py regimes      # Run only regime generalization
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd

import config
from src.data.fetcher import fetch_etf_prices, fetch_fama_french


def _load_data():
    """Load cached prices and Fama-French data."""
    print("[0/6] Loading data...")
    try:
        prices = pd.read_csv("data/raw/etf_prices.csv", index_col=0, parse_dates=True)
        ff_data = pd.read_csv("data/raw/ff5_daily.csv", index_col=0, parse_dates=True)
        print(f"  Loaded cached data: {prices.shape[0]} days × {prices.shape[1]} tickers.")
    except FileNotFoundError:
        prices = fetch_etf_prices()
        ff_data = fetch_fama_french()
        print(f"  Downloaded {prices.shape[0]} days × {prices.shape[1]} tickers.")
    return prices, ff_data


def run_ablation(prices, ff_data):
    """Phase 1: Ablation study."""
    print("\n" + "=" * 60)
    print("PHASE 1: ABLATION STUDY")
    print("=" * 60)
    from src.robustness.ablation import run_all_ablations
    t0 = time.time()
    df = run_all_ablations(prices, ff_data)
    print(f"\n{df.to_string(index=False)}")
    print(f"\n  Phase 1 completed in {time.time() - t0:.0f}s")
    return df


def run_baselines(prices, ff_data):
    """Phase 2: Baseline comparisons."""
    print("\n" + "=" * 60)
    print("PHASE 2: BASELINE COMPARISONS")
    print("=" * 60)
    from src.robustness.baselines import run_all_baselines
    t0 = time.time()
    df = run_all_baselines(prices, ff_data)
    print(f"\n{df.to_string(index=False)}")
    print(f"\n  Phase 2 completed in {time.time() - t0:.0f}s")
    return df


def run_sensitivity(prices, ff_data):
    """Phase 3: Sensitivity analysis."""
    print("\n" + "=" * 60)
    print("PHASE 3: SENSITIVITY ANALYSIS")
    print("=" * 60)
    from src.robustness.sensitivity import run_all_sensitivity
    t0 = time.time()
    run_all_sensitivity(prices, ff_data)
    print(f"\n  Phase 3 completed in {time.time() - t0:.0f}s")


def run_cross_asset(ff_data):
    """Phase 4: Cross-asset validation."""
    print("\n" + "=" * 60)
    print("PHASE 4: CROSS-ASSET VALIDATION")
    print("=" * 60)
    from src.robustness.cross_asset import run_all_cross_asset
    t0 = time.time()
    df = run_all_cross_asset(ff_data)
    print(f"\n{df.to_string(index=False)}")
    print(f"\n  Phase 4 completed in {time.time() - t0:.0f}s")
    return df


def run_regimes(prices, ff_data):
    """Phase 5: Regime generalization."""
    print("\n" + "=" * 60)
    print("PHASE 5: REGIME GENERALIZATION")
    print("=" * 60)
    from src.robustness.regime_generalization import run_all_regime_tests
    t0 = time.time()
    run_all_regime_tests(prices, ff_data)
    print(f"\n  Phase 5 completed in {time.time() - t0:.0f}s")


def main():
    t_start = time.time()

    print("=" * 60)
    print("Regime-Adaptive Portfolio — Robustness Analysis")
    print("=" * 60)

    prices, ff_data = _load_data()

    # Determine which phases to run
    args = sys.argv[1:]
    run_all = len(args) == 0

    if run_all or "ablation" in args:
        run_ablation(prices, ff_data)

    if run_all or "baselines" in args:
        run_baselines(prices, ff_data)

    if run_all or "sensitivity" in args:
        run_sensitivity(prices, ff_data)

    if run_all or "cross_asset" in args:
        run_cross_asset(ff_data)

    if run_all or "regimes" in args:
        run_regimes(prices, ff_data)

    print("\n" + "=" * 60)
    print(f"All robustness experiments complete! ({time.time() - t_start:.0f}s)")
    print(f"  Results: output/robustness/")
    print("=" * 60)


if __name__ == "__main__":
    main()
