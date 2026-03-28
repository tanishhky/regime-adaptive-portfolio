"""
Publication-quality figure generation.

All figures use font sizes ≥ 12 for axes and ≥ 14 for titles.
Saved at 200 DPI minimum.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────────
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

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output" / "figures"


def _ensure_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Figure 1: Multi-panel regime detection ────────────────────────────────────

def plot_regime_detection(
    prices: pd.Series,
    detector_df: pd.DataFrame,
    p_stress: pd.Series,
    portfolio_returns: pd.Series,
    save: bool = True,
) -> plt.Figure:
    """4-panel stacked figure: price + shading, detectors, composite, drawdown."""
    _ensure_dir()
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # Panel 1: Price with regime shading
    ax = axes[0]
    price_aligned = prices.reindex(p_stress.index).ffill()
    ax.plot(price_aligned.index, price_aligned.values, color="k", linewidth=0.8)
    ax.set_ylabel("SPY Price")
    ax.set_title("Multi-Scale Regime Detection System")

    # Shade background by P(stress) level
    for i in range(len(p_stress) - 1):
        p = p_stress.iloc[i]
        if p < 0.3:
            color = "green"
        elif p < 0.6:
            color = "gold"
        else:
            color = "red"
        ax.axvspan(p_stress.index[i], p_stress.index[i + 1],
                   alpha=0.15, color=color, linewidth=0)

    # Panel 2: Individual detector signals
    ax = axes[1]
    colors = {"cusum": "#1f77b4", "ewma": "#ff7f0e",
              "markov": "#2ca02c", "structural": "#d62728"}
    for col in detector_df.columns:
        ax.plot(detector_df.index, detector_df[col], label=col.upper(),
                alpha=0.8, linewidth=0.7, color=colors.get(col, "gray"))
    ax.set_ylabel("Signal [0, 1]")
    ax.legend(loc="upper left", ncol=4, framealpha=0.8)
    ax.set_ylim(-0.05, 1.05)

    # Panel 3: Composite P(stress)
    ax = axes[2]
    ax.fill_between(p_stress.index, 0, p_stress.values, alpha=0.4, color="crimson")
    ax.plot(p_stress.index, p_stress.values, color="crimson", linewidth=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("P(stress)")
    ax.set_ylim(-0.05, 1.05)

    # Panel 4: Portfolio drawdown
    ax = axes[3]
    cum_ret = portfolio_returns.cumsum()
    peak = cum_ret.cummax()
    dd = cum_ret - peak
    ax.fill_between(dd.index, 0, dd.values, alpha=0.5, color="steelblue")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "regime_detection.png")
    return fig


# ── Figure 2: Fuzzy membership functions ──────────────────────────────────────

def plot_membership_functions(
    sigmoid_params: np.ndarray,
    detector_names: list[str] | None = None,
    save: bool = True,
) -> plt.Figure:
    """2×2 subplot of calibrated sigmoid membership functions."""
    _ensure_dir()
    if detector_names is None:
        detector_names = ["CUSUM", "EWMA", "Markov-Switching", "Structural Break"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.linspace(0, 1, 200)

    for i, ax in enumerate(axes.flat):
        if i >= len(sigmoid_params):
            ax.set_visible(False)
            continue
        a, c = sigmoid_params[i]
        z = -a * (x - c)
        mu_high = 1.0 / (1.0 + np.exp(np.clip(z, -500, 500)))
        mu_low = 1.0 - mu_high

        ax.plot(x, mu_low, label=r"$\mu_{\mathrm{normal}}$", color="green", linewidth=2)
        ax.plot(x, mu_high, label=r"$\mu_{\mathrm{stress}}$", color="red", linewidth=2)
        ax.axvline(c, color="gray", linestyle="--", linewidth=0.8, label=f"c = {c:.2f}")
        ax.set_title(detector_names[i])
        ax.set_xlabel("Signal value")
        ax.set_ylabel("Membership degree")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Calibrated Fuzzy Membership Functions", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "membership_functions.png")
    return fig


# ── Figure 3: Asset characterization scatter ──────────────────────────────────

def plot_asset_scatter(
    assignments: dict,
    half_life_median: float,
    cvar_75: float,
    save: bool = True,
) -> plt.Figure:
    """Scatter: VaR(99%) vs half-life, colored by basket."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 7))

    colors_map = {"A": "#e74c3c", "B": "#f39c12", "C": "#27ae60"}
    labels_map = {"A": "Tactical", "B": "Avoid", "C": "Core"}

    for tkr, ba in assignments.items():
        hl = ba.half_life if np.isfinite(ba.half_life) else 100
        ax.scatter(
            abs(ba.cond_var99) * 100, hl,
            c=colors_map.get(ba.basket, "gray"), s=120, zorder=5,
            edgecolors="k", linewidth=0.5,
        )
        ax.annotate(tkr, (abs(ba.cond_var99) * 100, hl),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)

    # Decision boundaries
    ax.axhline(half_life_median, color="gray", linestyle="--", linewidth=1,
               label=f"Half-life median = {half_life_median:.1f}d")
    ax.axvline(abs(cvar_75) * 100, color="gray", linestyle=":", linewidth=1,
               label=f"|VaR(99%)| 75th pctile = {abs(cvar_75)*100:.2f}%")

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=f"Basket {k}: {labels_map[k]}")
               for k, c in colors_map.items()]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0],
              loc="upper right", fontsize=10)

    ax.set_xlabel("Conditional |VaR(99%)| (%)")
    ax.set_ylabel("OU Half-Life (trading days)")
    ax.set_title("Asset Characterization: Basket Assignment")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "asset_scatter.png")
    return fig


# ── Figure 4: Portfolio backtest ──────────────────────────────────────────────

def plot_backtest(
    strategy: pd.Series,
    benchmark: pd.Series,
    metrics_strategy: Any = None,
    save: bool = True,
) -> plt.Figure:
    """Cumulative return plot with drawdown subplot."""
    _ensure_dir()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    # Cumulative returns
    for series, label, color in [
        (strategy, "Regime-Adaptive", "#2c3e50"),
        (benchmark, "SPY (Buy & Hold)", "#7f8c8d"),
    ]:
        cum = series.cumsum()
        ax1.plot(cum.index, np.exp(cum.values) - 1, label=label,
                 linewidth=1.5 if "Regime" in label else 1.0, color=color)

    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Walk-Forward Out-of-Sample Backtest")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(alpha=0.3)

    # Annotate metrics
    if metrics_strategy is not None:
        text = (
            f"Sharpe: {metrics_strategy.sharpe_ratio:.2f}\n"
            f"Max DD: {metrics_strategy.max_drawdown:.1%}\n"
            f"Calmar: {metrics_strategy.calmar_ratio:.2f}"
        )
        ax1.text(0.98, 0.05, text, transform=ax1.transAxes,
                 fontsize=11, verticalalignment="bottom",
                 horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    # Drawdown subplot
    cum = strategy.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    ax2.fill_between(dd.index, 0, dd.values, alpha=0.5, color="#e74c3c")
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "backtest.png")
    return fig


# ── Figure 5: Detector contribution heatmap ──────────────────────────────────

def plot_detector_heatmap(
    detector_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Heatmap: time × detector, color = signal intensity."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(16, 4))

    # Resample to weekly for readability
    weekly = detector_df.resample("W").mean()
    data = weekly.T.values

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(weekly.columns)))
    ax.set_yticklabels([c.upper() for c in weekly.columns])

    # X-axis: show years
    n_weeks = data.shape[1]
    year_positions = []
    year_labels = []
    for i, date in enumerate(weekly.index):
        if date.month == 1 and date.day <= 7:
            year_positions.append(i)
            year_labels.append(str(date.year))
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels)

    ax.set_title("Detector Contribution Over Time")
    ax.set_xlabel("Date")
    plt.colorbar(im, ax=ax, label="Signal Intensity", shrink=0.8)

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "detector_heatmap.png")
    return fig


# ── Figure 6: HMM transition matrix ──────────────────────────────────────────

def plot_transition_matrix(
    transition_matrix: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Annotated heatmap of Markov-Switching transition probabilities."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(6, 5))

    # Reshape if needed
    tm = np.array(transition_matrix)
    if tm.ndim == 3:
        tm = tm[:, :, 0]
    if tm.shape != (2, 2):
        tm = tm.T if tm.shape == (2, 2) else tm[:2, :2]

    sns.heatmap(tm, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=["Bull", "Stress"],
                yticklabels=["Bull", "Stress"],
                ax=ax, square=True, linewidths=1)
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title("Hamilton Regime-Switching Transition Matrix")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "transition_matrix.png")
    return fig


# ── Figure 7: Rolling OOS Sharpe ─────────────────────────────────────────────

def plot_rolling_sharpe(
    strategy: pd.Series,
    benchmark: pd.Series,
    window: int = 252,
    save: bool = True,
) -> plt.Figure:
    """Rolling 1-year Sharpe of strategy vs benchmark."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(14, 5))

    for series, label, color in [
        (strategy, "Regime-Adaptive", "#2c3e50"),
        (benchmark, "SPY", "#7f8c8d"),
    ]:
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252)).dropna()
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                label=label, linewidth=1.2, color=color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("Rolling 1-Year Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.set_title("Walk-Forward Out-of-Sample Sharpe Ratio")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "rolling_sharpe.png")
    return fig


# ── Figure 8: Recovery analysis ──────────────────────────────────────────────

def plot_recovery_analysis(
    prices_df: pd.DataFrame,
    assignments: dict,
    events: dict[str, tuple[str, str]] | None = None,
    save: bool = True,
) -> plt.Figure:
    """Recovery paths for selected drawdown events, colored by basket."""
    _ensure_dir()
    if events is None:
        events = {
            "COVID-19 (2020)": ("2020-02-19", "2020-09-01"),
            "2022 Rate Hikes": ("2022-01-03", "2022-12-30"),
        }

    colors_map = {"A": "#e74c3c", "B": "#f39c12", "C": "#27ae60"}
    n_events = len(events)
    fig, axes = plt.subplots(1, n_events, figsize=(8 * n_events, 6))
    if n_events == 1:
        axes = [axes]

    for ax, (event_name, (start, end)) in zip(axes, events.items()):
        mask = (prices_df.index >= start) & (prices_df.index <= end)
        period = prices_df.loc[mask]

        if len(period) == 0:
            ax.set_title(f"{event_name} (no data)")
            continue

        for tkr in period.columns:
            if tkr not in assignments:
                continue
            basket = assignments[tkr].basket if hasattr(assignments[tkr], "basket") else assignments[tkr]
            normalised = period[tkr] / period[tkr].iloc[0] - 1
            ax.plot(normalised.index, normalised.values * 100,
                    color=colors_map.get(basket, "gray"),
                    alpha=0.7, linewidth=1.2, label=tkr)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(event_name)
        ax.set_ylabel("Return from Peak (%)")
        ax.set_xlabel("Date")
        ax.legend(fontsize=8, ncol=3, loc="lower right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    fig.suptitle("Sector Recovery Paths by Basket Assignment", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "recovery_analysis.png")
    return fig


def generate_all_figures(results: dict, prices: pd.DataFrame, **kwargs) -> None:
    """Generate all 8 publication figures from backtest results."""
    _ensure_dir()

    # 1. Regime detection
    spy_prices = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
    plot_regime_detection(
        spy_prices,
        results["detector_signals"],
        results["stress_signals"],
        results["strategy_returns"],
    )

    # 2. Membership functions
    fuzzy = kwargs.get("fuzzy_aggregator")
    if fuzzy is not None:
        plot_membership_functions(fuzzy.sigmoid_params)

    # 3. Asset scatter
    assignments = kwargs.get("assignments")
    if assignments is not None:
        half_life_median = kwargs.get("half_life_median", 5.0)
        cvar_75 = kwargs.get("cvar_75", -0.05)
        plot_asset_scatter(assignments, half_life_median, cvar_75)

    # 4. Backtest
    from src.backtest.metrics import compute_metrics
    m = compute_metrics(
        results["strategy_returns"],
        stress_signals=results["stress_signals"],
        turnover_series=results["turnover"],
    )
    plot_backtest(
        results["strategy_returns"],
        results["benchmark_returns"],
        metrics_strategy=m,
    )

    # 5. Detector heatmap
    plot_detector_heatmap(results["detector_signals"])

    # 6. Transition matrix
    markov = kwargs.get("markov_detector")
    if markov is not None and markov.transition_matrix is not None:
        plot_transition_matrix(markov.transition_matrix)

    # 7. Rolling Sharpe
    plot_rolling_sharpe(
        results["strategy_returns"],
        results["benchmark_returns"],
    )

    # 8. Recovery analysis
    sector_tickers = kwargs.get("sector_tickers", [])
    sector_prices = prices[[t for t in sector_tickers if t in prices.columns]]
    if assignments is not None and len(sector_prices.columns) > 0:
        plot_recovery_analysis(sector_prices, assignments)

    plt.close("all")
    print(f"All figures saved to {OUTPUT_DIR}")
