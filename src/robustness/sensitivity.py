"""
Sensitivity analysis — test robustness to parameter choices.

3a: Transaction cost sensitivity (0-50 bps)
3b: Walk-forward window sensitivity
3c: Stress threshold sensitivity (Basket C scaling)
3d: Regime-conditional transaction costs (VIX-scaled)

References
----------
Brunnermeier, M.K. & Pedersen, L.H. (2009). "Market liquidity and
funding liquidity." Review of Financial Studies, 22(6), 2201-2238.

Hameed, A., Kang, W. & Viswanathan, S. (2010). "Stock market
declines and liquidity." Journal of Finance, 65(1), 257-293.
"""

from __future__ import annotations

import copy
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
from src.portfolio.execution import ExecutionModel, ExecutionLog


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


def _run_engine(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
    min_train: int = config.WALK_FORWARD_MIN_TRAIN,
    step: int = config.WALK_FORWARD_STEP,
    cost_bps: int = config.TRANSACTION_COST_BPS,
) -> dict[str, Any]:
    """Run the walk-forward engine with custom parameters."""
    engine = WalkForwardEngine(min_train=min_train, step=step)
    engine.execution = ExecutionModel(cost_bps=cost_bps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return engine.run(prices, ff_data=ff_data)


def _metrics_from_results(
    results: dict[str, Any],
    ff_data: pd.DataFrame,
) -> PerformanceMetrics:
    """Extract metrics from engine results."""
    port_ret = results["strategy_returns"]
    rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
    if ff_data is not None and "RF" in ff_data.columns:
        rf_daily = float(ff_data["RF"].reindex(port_ret.index).mean())
    return compute_metrics(
        port_ret, rf_daily,
        turnover_series=results["turnover"],
        stress_signals=results["stress_signals"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3a: Transaction Cost Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def run_cost_sensitivity(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> pd.DataFrame:
    """Run the strategy at different transaction cost levels."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cost_levels_bps = [0, 5, 10, 15, 20, 30, 50]
    rows = []

    for bps in cost_levels_bps:
        print(f"    Cost sensitivity: {bps} bps...")
        results = _run_engine(prices, ff_data, cost_bps=bps)
        m = _metrics_from_results(results, ff_data)
        rows.append({
            "cost_bps": bps,
            "ann_return": m.annualised_return,
            "ann_vol": m.annualised_volatility,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "max_dd": m.max_drawdown,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "cost_sensitivity.csv", index=False)
    print(f"\n    Saved: {OUTPUT_DIR / 'cost_sensitivity.csv'}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["cost_bps"], df["sharpe"], "o-", color="#1a5276", linewidth=2)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio vs Transaction Cost")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color="#b03a2e", linestyle="--", alpha=0.5, label="Sharpe = 0.5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cost_sensitivity.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'cost_sensitivity.png'}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3b: Walk-Forward Window Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def run_window_sensitivity(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> pd.DataFrame:
    """Test different training window / OOS step combinations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        (252, 63),   # 1yr train, quarterly
        (504, 63),   # 2yr train, quarterly (current)
        (756, 63),   # 3yr train, quarterly
        (504, 21),   # 2yr train, monthly
        (504, 126),  # 2yr train, semi-annual
    ]

    rows = []
    for train, step in configs:
        label = f"train={train},step={step}"
        print(f"    Window sensitivity: {label}...")
        results = _run_engine(prices, ff_data, min_train=train, step=step)
        m = _metrics_from_results(results, ff_data)
        rows.append({
            "train_window": train,
            "oos_step": step,
            "ann_return": m.annualised_return,
            "ann_vol": m.annualised_volatility,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "calmar": m.calmar_ratio,
            "max_dd": m.max_drawdown,
            "turnover": m.annualised_turnover,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "window_sensitivity.csv", index=False)
    print(f"\n    Saved: {OUTPUT_DIR / 'window_sensitivity.csv'}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3c: Stress Threshold Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

def run_threshold_sensitivity(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> pd.DataFrame:
    """Test the effect of different Basket C stress scaling thresholds.

    The default threshold is 0.7 (hardcoded in BasketManager._compute_weights).
    We create a subclass that overrides this.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from src.portfolio.basket_manager import BasketManager

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []

    for thresh in thresholds:
        print(f"    Threshold sensitivity: {thresh:.1f}...")

        engine = WalkForwardEngine()

        # Monkey-patch the basket manager's _compute_weights to use custom threshold
        original_compute = engine.basket_mgr._compute_weights

        def patched_compute(
            tickers, assignments, vol_dict, p_stress,
            entry_th, exit_th, liquidated, structural_break,
            _thresh=thresh, _orig=original_compute,
        ):
            """Patched weight computation with custom Basket C threshold."""
            n = len(tickers)

            # Step 1: base inverse-vol weights
            base = np.zeros(n)
            for i, tkr in enumerate(tickers):
                if tkr not in assignments:
                    continue
                vol = vol_dict.get(tkr, 0.2)
                base[i] = 1.0 / max(vol, 1e-6)
            base_total = base.sum()
            if base_total > 0:
                base /= base_total

            # Step 2: basket-specific scaling with custom threshold
            scaled = np.zeros(n)
            for i, tkr in enumerate(tickers):
                if tkr not in assignments:
                    continue
                ba = assignments[tkr]

                if ba.basket == "A":
                    if p_stress > entry_th:
                        liquidated.add(tkr)
                        scaled[i] = 0.0
                    elif p_stress < exit_th and tkr in liquidated:
                        liquidated.discard(tkr)
                        scaled[i] = base[i]
                    elif tkr in liquidated:
                        scaled[i] = 0.0
                    else:
                        scaled[i] = base[i]
                elif ba.basket == "B":
                    scaled[i] = base[i] * (1.0 - p_stress)
                elif ba.basket == "C":
                    if structural_break:
                        scaled[i] = 0.0
                    elif p_stress > _thresh:
                        scale = max(0.0, (1.0 - p_stress) / (1.0 - _thresh))
                        scaled[i] = base[i] * scale
                    else:
                        scaled[i] = base[i]

            total = scaled.sum()
            if total > 1.0 + 1e-8:
                scaled /= total
            return scaled

        engine.basket_mgr._compute_weights = patched_compute

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = engine.run(prices, ff_data=ff_data)

        m = _metrics_from_results(results, ff_data)
        rows.append({
            "basket_c_threshold": thresh,
            "ann_return": m.annualised_return,
            "ann_vol": m.annualised_volatility,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "calmar": m.calmar_ratio,
            "max_dd": m.max_drawdown,
            "turnover": m.annualised_turnover,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "threshold_sensitivity.csv", index=False)
    print(f"\n    Saved: {OUTPUT_DIR / 'threshold_sensitivity.csv'}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3d: Regime-Conditional Transaction Costs (VIX-Scaled)
# ═══════════════════════════════════════════════════════════════════════════════

class RegimeAwareExecutionModel(ExecutionModel):
    """Transaction cost model that scales with VIX.

    cost_t = base_cost × (VIX_t / VIX_median)

    where VIX_median is computed from the training window at each
    walk-forward step (no lookahead).

    Capped at max_multiplier × base cost to prevent extreme outliers.

    References
    ----------
    Brunnermeier, M.K. & Pedersen, L.H. (2009). "Market liquidity and
    funding liquidity." Review of Financial Studies, 22(6), 2201-2238.

    Hameed, A., Kang, W. & Viswanathan, S. (2010). "Stock market
    declines and liquidity." Journal of Finance, 65(1), 257-293.
    """

    def __init__(
        self,
        base_cost_bps: int = 10,
        max_multiplier: float = 5.0,
    ) -> None:
        super().__init__(cost_bps=base_cost_bps)
        self.base_cost_bps = base_cost_bps
        self.max_multiplier = max_multiplier
        self._vix_median: float = 17.0
        self._current_vix: float = 17.0
        self._effective_costs: list[float] = []

    def set_vix_median(self, vix_series: pd.Series) -> None:
        """Set VIX median from training window (call at each rebalance)."""
        clean = vix_series.dropna()
        if len(clean) > 0:
            self._vix_median = float(clean.median())

    def set_current_vix(self, vix: float) -> None:
        """Set today's VIX (call daily in the OOS loop)."""
        self._current_vix = vix if not np.isnan(vix) else self._vix_median

    def execute(
        self,
        old_weights: dict[str, float],
        new_weights: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        """Execute with VIX-scaled cost."""
        multiplier = min(
            self._current_vix / max(self._vix_median, 1.0),
            self.max_multiplier,
        )
        self.cost_frac = (self.base_cost_bps * multiplier) / 10_000.0
        effective_bps = self.base_cost_bps * multiplier
        self._effective_costs.append(effective_bps)
        return super().execute(old_weights, new_weights)

    def reset(self) -> None:
        super().reset()
        self._effective_costs.clear()


class VIXScaledWalkForwardEngine(WalkForwardEngine):
    """Walk-forward engine with VIX-scaled transaction costs.

    At each rebalance, updates the VIX median from training data.
    At each daily step, updates the current VIX before execution.
    """

    def __init__(
        self,
        base_cost_bps: int = 10,
        max_multiplier: float = 5.0,
        min_train: int = config.WALK_FORWARD_MIN_TRAIN,
        step: int = config.WALK_FORWARD_STEP,
    ) -> None:
        super().__init__(min_train=min_train, step=step)
        self.execution = RegimeAwareExecutionModel(
            base_cost_bps=base_cost_bps,
            max_multiplier=max_multiplier,
        )

    def run(
        self,
        prices: pd.DataFrame,
        benchmark_ticker: str = config.BENCHMARK,
        sector_tickers: list[str] | None = None,
        ff_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run walk-forward with VIX-scaled costs.

        Overrides the parent run() to inject VIX data at appropriate points.
        This is done by wrapping the execution model's behavior.
        """
        if sector_tickers is None:
            sector_tickers = list(config.SECTOR_ETFS.keys())

        vix_col = config.VIX_TICKER
        if vix_col not in prices.columns:
            print(f"    Warning: {vix_col} not in prices, falling back to flat cost")
            return super().run(prices, benchmark_ticker, sector_tickers, ff_data)

        vix_prices = prices[vix_col].copy()
        log_ret = np.log(prices / prices.shift(1)).dropna(how="all")
        n_days = len(log_ret)

        # We need to hook into the rebalance and daily loops.
        # Strategy: override the execution model to look up VIX by date,
        # and set the median at each rebalance point.

        # Pre-compute VIX series aligned with log_ret index
        vix_aligned = vix_prices.reindex(log_ret.index).ffill()

        # Store VIX data on the execution model for self-service lookup
        exec_model: RegimeAwareExecutionModel = self.execution
        self._vix_aligned = vix_aligned
        self._log_ret_index = log_ret.index

        # Patch: before each execute() call, the engine sets current weights.
        # We wrap execute to auto-set VIX from the current date.
        original_execute = exec_model.execute
        self._day_counter = 0
        self._rebalance_points = list(range(self.min_train, n_days, self.step))
        self._current_rp_idx = 0

        # Track which OOS day we're on
        oos_day_idx = [0]  # mutable counter

        def patched_execute(old_weights, new_weights):
            # Figure out current day from execution log length
            day_num = len(exec_model.log.daily_costs)
            # Map to absolute index: min_train + day_num
            abs_idx = self.min_train + day_num
            if abs_idx < len(self._log_ret_index):
                day = self._log_ret_index[abs_idx]
                if day in self._vix_aligned.index:
                    exec_model.set_current_vix(float(self._vix_aligned.loc[day]))
            return original_execute(old_weights, new_weights)

        exec_model.execute = patched_execute

        # At each rebalance, set VIX median from training window
        original_reset = exec_model.reset

        def patched_reset():
            original_reset()
            # Will be called at start; median set in calibrate hook below

        exec_model.reset = patched_reset

        # We also need to set VIX median at each rebalance.
        # Patch calibrate_thresholds on basket_mgr to also set VIX median.
        original_calibrate = self.basket_mgr.calibrate_thresholds

        def patched_calibrate(p_stress_series, returns_df, assignments,
                              vol_dict, risk_free_daily):
            # The training window ends at the start of the current OOS window.
            # p_stress_series index gives us the training dates.
            if len(p_stress_series) > 0:
                train_end_date = p_stress_series.index[-1]
                train_vix = vix_aligned.loc[:train_end_date].dropna()
                exec_model.set_vix_median(train_vix)
            return original_calibrate(
                p_stress_series, returns_df, assignments,
                vol_dict, risk_free_daily,
            )

        self.basket_mgr.calibrate_thresholds = patched_calibrate

        try:
            results = super().run(prices, benchmark_ticker, sector_tickers, ff_data)
        finally:
            # Restore originals
            exec_model.execute = original_execute
            exec_model.reset = original_reset
            self.basket_mgr.calibrate_thresholds = original_calibrate

        # Attach effective cost series
        results["effective_costs_bps"] = pd.Series(
            exec_model._effective_costs[:len(results["strategy_returns"])],
            index=results["strategy_returns"].index,
            name="effective_cost_bps",
        )

        return results


def run_regime_costs(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compare flat vs VIX-scaled transaction cost models."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ("flat_10bps", 10, None),        # Flat cost
        ("vix_scaled", 10, 5.0),         # VIX-scaled, cap 5x
        ("vix_scaled_conservative", 15, 5.0),  # Conservative VIX-scaled
    ]

    rows = []
    all_results = {}

    for label, base_bps, max_mult in configs:
        print(f"    Regime costs: {label}...")

        if max_mult is None:
            # Flat cost
            results = _run_engine(prices, ff_data, cost_bps=base_bps)
        else:
            engine = VIXScaledWalkForwardEngine(
                base_cost_bps=base_bps,
                max_multiplier=max_mult,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = engine.run(prices, ff_data=ff_data)

        m = _metrics_from_results(results, ff_data)
        rows.append({
            "cost_model": label,
            "base_bps": base_bps,
            "ann_return": m.annualised_return,
            "ann_vol": m.annualised_volatility,
            "sharpe": m.sharpe_ratio,
            "sortino": m.sortino_ratio,
            "calmar": m.calmar_ratio,
            "max_dd": m.max_drawdown,
            "turnover": m.annualised_turnover,
        })
        all_results[label] = results

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "regime_costs.csv", index=False)
    print(f"\n    Saved: {OUTPUT_DIR / 'regime_costs.csv'}")

    # Plot timeseries of effective costs
    _plot_regime_cost_timeseries(all_results, prices)
    _plot_regime_cost_comparison(all_results)

    return df


def _plot_regime_cost_timeseries(
    all_results: dict[str, dict],
    prices: pd.DataFrame,
) -> None:
    """Dual-axis chart: effective cost (bps) and VIX over time."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # VIX on right axis
    ax2 = ax1.twinx()

    if "vix_scaled" in all_results and "effective_costs_bps" in all_results["vix_scaled"]:
        eff_costs = all_results["vix_scaled"]["effective_costs_bps"]
        ax1.plot(eff_costs.index, eff_costs.values, color="#b03a2e",
                 linewidth=0.8, alpha=0.7, label="Effective Cost (bps)")
        ax1.axhline(y=10, color="#1a5276", linestyle="--", alpha=0.5,
                     label="Flat 10 bps")

    # VIX
    vix_col = config.VIX_TICKER
    if vix_col in prices.columns:
        oos_idx = all_results.get("vix_scaled", all_results.get("flat_10bps", {}))
        if "strategy_returns" in oos_idx:
            idx = oos_idx["strategy_returns"].index
            vix = prices[vix_col].reindex(idx).ffill()
            ax2.fill_between(vix.index, 0, vix.values, alpha=0.15,
                             color="#7d8c8e", label="VIX")
            ax2.set_ylabel("VIX", color="#7d8c8e")

    ax1.set_ylabel("Effective Cost (bps)")
    ax1.set_title("VIX-Scaled Transaction Costs Over Time")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regime_costs_timeseries.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'regime_costs_timeseries.png'}")


def _plot_regime_cost_comparison(all_results: dict[str, dict]) -> None:
    """Equity curves for flat vs VIX-scaled costs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "flat_10bps": "#1a5276",
        "vix_scaled": "#b03a2e",
        "vix_scaled_conservative": "#d4ac0d",
    }
    labels = {
        "flat_10bps": "Flat 10 bps",
        "vix_scaled": "VIX-Scaled (10 bps base)",
        "vix_scaled_conservative": "VIX-Scaled Conservative (15 bps base)",
    }

    for name, results in all_results.items():
        ret = results["strategy_returns"]
        cum = ret.cumsum().apply(np.exp)
        ax.plot(cum.index, cum.values, color=colors.get(name, "#333"),
                linewidth=1.8, label=labels.get(name, name))

    ax.set_title("Impact of VIX-Scaled Transaction Costs")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "regime_costs_comparison.png")
    plt.close(fig)
    print(f"    Saved: {OUTPUT_DIR / 'regime_costs_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# Combined Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_sensitivity(
    prices: pd.DataFrame,
    ff_data: pd.DataFrame,
) -> None:
    """Run all sensitivity analyses."""
    print("\n  [3a] Transaction cost sensitivity...")
    cost_df = run_cost_sensitivity(prices, ff_data)
    print("\n" + cost_df.to_string(index=False))

    print("\n  [3b] Walk-forward window sensitivity...")
    window_df = run_window_sensitivity(prices, ff_data)
    print("\n" + window_df.to_string(index=False))

    print("\n  [3c] Stress threshold sensitivity...")
    thresh_df = run_threshold_sensitivity(prices, ff_data)
    print("\n" + thresh_df.to_string(index=False))

    print("\n  [3d] Regime-conditional costs (VIX-scaled)...")
    regime_df = run_regime_costs(prices, ff_data)
    print("\n" + regime_df.to_string(index=False))
