"""Ablation study: quantify what the 2-of-4 consensus buys over its parts.

Rebuilt 2026-07-04 for the current architecture. The previous version
ablated a detector roster (EWMA / Markov / structural) and a weighted
aggregator that no longer exist; the live system is CUSUM / correlation /
breadth / skewness feeding a WEIGHTLESS max-of-top-2 fuzzy consensus.

Mechanics
---------
Masking happens at the aggregator, not the detectors: every detector still
calibrates and produces signals (identical code path to production), but a
``MaskedFuzzyAggregator`` ignores inactive detectors' memberships when
forming the composite. Because both the training-time ``aggregate_series``
(used to calibrate allocation thresholds) and the day-by-day OOS
``aggregate`` go through the aggregator, the mask is consistent end to end.

The consensus-order subtlety, handled explicitly: with the production
max-of-top-2 rule a SINGLE active detector could never act (the
second-largest of one membership is zero), so single-detector arms would
be vacuously flat. Single arms therefore run top-1 (their own membership
drives allocation directly), which is the FAVORABLE interpretation for
them, while multi-detector arms keep the production 2-of-K consensus. Any
win the full ensemble shows is thus not an artifact of crippling the
singles.

Outputs
-------
output/robustness/ablation_results.csv    per-arm performance table
output/robustness/ablation_equity_curves.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.backtest.walk_forward import WalkForwardEngine
from src.backtest.metrics import compute_metrics
from src.detectors.fuzzy_aggregator import FuzzyAggregator

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "robustness"

# Column order of the engine's signal matrix (walk_forward.py builds it as
# [cusum, correlation, breadth, skewness]).
DETECTOR_NAMES = ["cusum", "correlation", "breadth", "skewness"]

ABLATION_CONFIGS: list[tuple[str, list[str]]] = [
    ("full_ensemble",    DETECTOR_NAMES),
    ("cusum_only",       ["cusum"]),
    ("correlation_only", ["correlation"]),
    ("breadth_only",     ["breadth"]),
    ("skewness_only",    ["skewness"]),
    ("no_cusum",         ["correlation", "breadth", "skewness"]),
    ("no_correlation",   ["cusum", "breadth", "skewness"]),
    ("no_breadth",       ["cusum", "correlation", "skewness"]),
    ("no_skewness",      ["cusum", "correlation", "breadth"]),
]


class MaskedFuzzyAggregator(FuzzyAggregator):
    """FuzzyAggregator that ignores inactive detectors.

    Composite = k-th largest of the ACTIVE memberships with
    k = min(2, n_active): the production 2-of-K consensus whenever two or
    more detectors are active, top-1 for single-detector arms.
    """

    def __init__(self, active_idx: list[int]) -> None:
        super().__init__()
        self.active_idx = np.asarray(sorted(active_idx), dtype=int)
        self.k = min(2, len(self.active_idx))

    def aggregate(self, signals) -> float:
        s = np.asarray(signals, dtype=float)
        memberships = np.zeros(len(self.active_idx))
        for j, i in enumerate(self.active_idx):
            a, c = self.sigmoid_params[i]
            memberships[j] = self._sigmoid(s[i : i + 1], a, c)[0]
        sorted_m = np.sort(memberships)[::-1]
        return float(np.clip(sorted_m[self.k - 1], 0.0, 1.0))


def _make_engine(active: list[str]) -> WalkForwardEngine:
    engine = WalkForwardEngine()
    idx = [DETECTOR_NAMES.index(n) for n in active]
    engine.fuzzy = MaskedFuzzyAggregator(idx)
    return engine


def run_ablation(prices: pd.DataFrame,
                 ff_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the walk-forward backtest for every ablation arm."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    curves: dict[str, pd.Series] = {}

    for name, active in ABLATION_CONFIGS:
        print(f"  [ablation] {name} (active: {', '.join(active)}, "
              f"rule: top-{min(2, len(active))})")
        engine = _make_engine(active)
        result = engine.run(prices, ff_data=ff_data)
        rets = result["strategy_returns"]
        stress = result["stress_signals"]
        m = compute_metrics(rets, stress_signals=stress,
                            turnover_series=result["turnover"])
        rows.append({
            "config": name,
            "n_active": len(active),
            "rule": f"top-{min(2, len(active))}",
            "sharpe": round(m.sharpe_ratio, 3),
            "sortino": round(m.sortino_ratio, 3),
            "ann_return_pct": round(m.annualised_return * 100, 2),
            "ann_vol_pct": round(m.annualised_volatility * 100, 2),
            "max_drawdown_pct": round(m.max_drawdown * 100, 2),
            "calmar": round(m.calmar_ratio, 3),
            "pct_days_stressed": round(float((stress > 0.5).mean()) * 100, 1),
            "false_alarm_rate": round(m.false_alarm_rate, 3),
        })
        curves[name] = rets

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    print(f"    Saved: {OUTPUT_DIR / 'ablation_results.csv'}")

    _plot_ablation_equity_curves(curves, prices)
    print(df.to_string(index=False))
    return df


def _plot_ablation_equity_curves(curves: dict[str, pd.Series],
                                 prices: pd.DataFrame) -> None:
    """Full ensemble vs singles (dashed) vs leave-one-outs (dotted) vs SPY."""
    fig, ax = plt.subplots(figsize=(12, 6))

    oos_index = curves["full_ensemble"].index
    spy_ret = np.log(prices[config.BENCHMARK] / prices[config.BENCHMARK].shift(1))
    spy_cum = spy_ret.reindex(oos_index).fillna(0).cumsum().apply(np.exp)
    ax.plot(spy_cum.index, spy_cum.values, color="#7d8c8e", lw=1.4,
            ls="-", alpha=0.6, label="SPY")

    for name, rets in curves.items():
        cum = rets.cumsum().apply(np.exp)
        if name == "full_ensemble":
            ax.plot(cum.index, cum.values, lw=2.4, color="black",
                    label="full ensemble (2-of-4)")
        elif name.endswith("_only"):
            ax.plot(cum.index, cum.values, lw=1.1, ls="--", alpha=0.85,
                    label=name.replace("_only", " only (top-1)"))
        else:
            ax.plot(cum.index, cum.values, lw=1.1, ls=":", alpha=0.85,
                    label=name.replace("no_", "without "))

    ax.set_title("Ablation: detector subsets, walk-forward out of sample")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ablation_equity_curves.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'ablation_equity_curves.png'}")
