"""
Ablation study — quantify the contribution of each detector to ensemble performance.

Runs the walk-forward backtest with different detector subsets active.
Inactive detectors still calibrate (to avoid changing the code path) but their
signals are zeroed before fuzzy aggregation, and the fuzzy weights are
re-optimised for the active subset.

This is implemented via an AblationEngine that subclasses WalkForwardEngine
and overrides signal paths by monkey-patching detector .signal() methods to
return 0.0 for inactive detectors.
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


# ── Plotting style (matches src/visualization/plots.py) ──────────────────────
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

DETECTOR_NAMES = ["cusum", "ewma", "markov", "structural"]

ABLATION_CONFIGS: list[tuple[str, list[str]]] = [
    ("full_ensemble",   ["cusum", "ewma", "markov", "structural"]),
    ("cusum_only",      ["cusum"]),
    ("ewma_only",       ["ewma"]),
    ("markov_only",     ["markov"]),
    ("structural_only", ["structural"]),
    ("no_cusum",        ["ewma", "markov", "structural"]),
    ("no_ewma",         ["cusum", "markov", "structural"]),
    ("no_markov",       ["cusum", "ewma", "structural"]),
    ("no_structural",   ["cusum", "ewma", "markov"]),
    ("cusum_ewma",      ["cusum", "ewma"]),
]


class AblationEngine(WalkForwardEngine):
    """Walk-forward engine that masks inactive detector signals.

    Hybrid approach to avoid numerical issues with all-zero signal columns:
    1. All detectors calibrate and produce training signals normally
    2. Fuzzy aggregator calibrates on full signal matrix (stable optimization)
    3. After calibration, inactive detector weights are zeroed and redistributed
    4. During OOS, inactive detector signals are zeroed before aggregation

    This ensures the fuzzy optimizer always has a well-conditioned landscape
    while correctly excluding inactive detectors from the final P(stress).
    """

    def __init__(
        self,
        active_detectors: list[str],
        min_train: int = config.WALK_FORWARD_MIN_TRAIN,
        step: int = config.WALK_FORWARD_STEP,
    ) -> None:
        super().__init__(min_train=min_train, step=step)
        self.active_detectors = set(active_detectors)

    def run(
        self,
        prices: pd.DataFrame,
        benchmark_ticker: str = config.BENCHMARK,
        sector_tickers: list[str] | None = None,
        ff_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run walk-forward with masked detectors."""
        # Save original methods
        orig_signal = {
            "cusum": self.cusum.signal,
            "ewma": self.ewma.signal,
            "markov": self.markov.signal,
            "structural": self.structural.signal,
        }
        orig_calibrate = self.fuzzy.calibrate

        def _make_zero_signal(original):
            def zero_signal(r_t):
                original(r_t)
                return 0.0
            return zero_signal

        def patched_calibrate(signal_matrix, returns, **kwargs):
            """Calibrate normally, then zero inactive weights."""
            try:
                orig_calibrate(signal_matrix, returns, **kwargs)
            except (ValueError, np.linalg.LinAlgError):
                # Scipy L-BFGS-B can fail on older versions with certain
                # signal patterns. Fall back to default parameters.
                self.fuzzy.sigmoid_params = np.tile(
                    [10.0, 0.5], (self.fuzzy.N_DETECTORS, 1)
                )
                self.fuzzy.weights = np.ones(self.fuzzy.N_DETECTORS) / self.fuzzy.N_DETECTORS
            # Zero out inactive detector weights and redistribute
            for i, name in enumerate(DETECTOR_NAMES):
                if name not in self.active_detectors:
                    self.fuzzy.weights[i] = 0.0
            total = self.fuzzy.weights.sum()
            if total > 0:
                self.fuzzy.weights /= total
            else:
                for i, name in enumerate(DETECTOR_NAMES):
                    if name in self.active_detectors:
                        self.fuzzy.weights[i] = 1.0 / len(self.active_detectors)

        try:
            # Patch: zero OOS signals for inactive detectors
            for name in DETECTOR_NAMES:
                if name not in self.active_detectors:
                    detector = getattr(self, name)
                    detector.signal = _make_zero_signal(orig_signal[name])

            # Patch: redistribute weights after calibration
            self.fuzzy.calibrate = patched_calibrate

            return super().run(
                prices,
                benchmark_ticker=benchmark_ticker,
                sector_tickers=sector_tickers,
                ff_data=ff_data,
            )
        finally:
            for name in DETECTOR_NAMES:
                getattr(self, name).signal = orig_signal[name]
            self.fuzzy.calibrate = orig_calibrate


def run_ablation(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
    active_detectors: list[str],
    label: str = "",
) -> dict:
    """Run walk-forward backtest with only the specified detectors active.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily Close prices.
    ff_data : pd.DataFrame
        Fama-French factor data.
    active_detectors : list[str]
        Subset of ["cusum", "ewma", "markov", "structural"].
    label : str
        Label for this configuration.

    Returns
    -------
    dict
        Dictionary with metrics and return series.
    """
    print(f"    Running ablation: {label} "
          f"(active: {', '.join(active_detectors)})...")

    engine = AblationEngine(active_detectors=active_detectors)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = engine.run(prices, ff_data=ff_data)

    port_ret = results["strategy_returns"]
    rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
    if ff_data is not None and "RF" in ff_data.columns:
        rf_daily = float(ff_data["RF"].reindex(port_ret.index).mean())

    metrics = compute_metrics(
        port_ret, rf_daily,
        turnover_series=results["turnover"],
        stress_signals=results["stress_signals"],
    )

    return {
        "label": label,
        "metrics": metrics,
        "strategy_returns": port_ret,
        "results": results,
    }


def run_all_ablations(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run all 10 ablation configurations and compile results.

    Returns
    -------
    pd.DataFrame
        Results table with one row per configuration.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    equity_curves = {}

    for label, active in ABLATION_CONFIGS:
        result = run_ablation(prices, ff_data, active, label)
        m: PerformanceMetrics = result["metrics"]
        rows.append({
            "config": label,
            "ann_return": m.annualised_return,
            "ann_vol": m.annualised_volatility,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "calmar": m.calmar_ratio,
            "max_dd": m.max_drawdown,
            "max_dd_duration": m.max_drawdown_duration,
            "turnover": m.annualised_turnover,
        })
        equity_curves[label] = result["strategy_returns"]

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    print(f"\n    Saved: {OUTPUT_DIR / 'ablation_results.csv'}")

    # Generate equity curve plot
    _plot_ablation_equity_curves(equity_curves, prices)

    return df


def _plot_ablation_equity_curves(
    equity_curves: dict[str, pd.Series],
    prices: pd.DataFrame,
) -> None:
    """Plot cumulative returns for key ablation configs vs SPY."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # SPY benchmark
    oos_index = equity_curves["full_ensemble"].index
    spy_ret = np.log(prices[config.BENCHMARK] / prices[config.BENCHMARK].shift(1))
    spy_ret = spy_ret.reindex(oos_index).fillna(0)
    spy_cum = spy_ret.cumsum().apply(np.exp)
    ax.plot(spy_cum.index, spy_cum.values, color="#7d8c8e",
            linewidth=1.5, label="SPY", linestyle="--")

    colors = {
        "full_ensemble": "#1a5276",
        "cusum_only": "#b03a2e",
        "no_cusum": "#27ae60",
        "cusum_ewma": "#8e44ad",
    }
    for label in ["full_ensemble", "cusum_only", "no_cusum", "cusum_ewma"]:
        if label in equity_curves:
            cum = equity_curves[label].cumsum().apply(np.exp)
            ax.plot(cum.index, cum.values, color=colors.get(label, "#333"),
                    linewidth=1.8, label=label.replace("_", " ").title())

    ax.set_title("Ablation Study: Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ablation_equity_curves.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'ablation_equity_curves.png'}")
