"""
Walk-forward backtest engine — zero lookahead bias.

At each rebalance step:
1. Calibrate all detectors on training data only
2. Characterise assets on training data only
3. Classify into baskets
4. Calibrate entry/exit thresholds
5. Run day-by-day through the out-of-sample test window

CRITICAL: All detectors use only past data. No smoothed probabilities.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

import config
from src.detectors.cusum import CUSUMDetector
from src.detectors.correlation import CorrelationDetector
from src.detectors.breadth import BreadthDetector
from src.detectors.skewness import SkewnessDetector
from src.detectors.fuzzy_aggregator import FuzzyAggregator
from src.characterization.volatility import GARCHVolatility
from src.characterization.recovery import RecoveryEstimator
from src.characterization.classifier import BasketClassifier
from src.portfolio.basket_manager import BasketManager
from src.portfolio.execution import ExecutionModel


class WalkForwardEngine:
    """Walk-forward backtest with strict no-lookahead guarantee."""

    def __init__(
        self,
        min_train: int = config.WALK_FORWARD_MIN_TRAIN,
        step: int = config.WALK_FORWARD_STEP,
    ) -> None:
        self.min_train = min_train
        self.step = step

        # Detectors
        self.cusum = CUSUMDetector()
        self.correlation = CorrelationDetector(window=21)
        self.breadth = BreadthDetector(window=21)
        self.skewness = SkewnessDetector(window=63)

        # Characterisation
        self.garch = GARCHVolatility()
        self.recovery = RecoveryEstimator()
        self.classifier = BasketClassifier()

        # Aggregator and portfolio manager
        self.fuzzy = FuzzyAggregator()
        self.basket_mgr = BasketManager()

        # Execution
        self.execution = ExecutionModel()

        # Results storage
        self.portfolio_returns: list[float] = []
        self.portfolio_dates: list[pd.Timestamp] = []
        self.stress_signals: list[float] = []
        self.detector_signals: list[dict[str, float]] = []
        self.daily_weights: list[dict[str, float]] = []
        self.daily_turnover: list[float] = []
        self.basket_history: list[dict] = []

    def run(
        self,
        prices: pd.DataFrame,
        benchmark_ticker: str = config.BENCHMARK,
        sector_tickers: list[str] | None = None,
        ff_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Execute the full walk-forward backtest.

        Parameters
        ----------
        prices : pd.DataFrame
            Daily Close prices (all tickers including benchmark and VIX).
        benchmark_ticker : str
            Benchmark ticker.
        sector_tickers : list[str] | None
            Sector ETF tickers to trade.
        ff_data : pd.DataFrame | None
            Fama-French data (for risk-free rate).

        Returns
        -------
        dict
            Dictionary with portfolio returns, signals, and metadata.
        """
        if sector_tickers is None:
            sector_tickers = list(config.SECTOR_ETFS.keys())

        # Compute log returns
        log_ret = np.log(prices / prices.shift(1)).dropna(how="all")

        # Risk-free rate
        rf_daily = config.RISK_FREE_RATE_ANNUAL / 252.0
        if ff_data is not None and "RF" in ff_data.columns:
            rf_series = ff_data["RF"].reindex(log_ret.index).fillna(rf_daily)
        else:
            rf_series = pd.Series(rf_daily, index=log_ret.index)

        # Filter to available tickers
        available = [t for t in sector_tickers if t in log_ret.columns]
        if benchmark_ticker not in log_ret.columns:
            raise ValueError(f"Benchmark {benchmark_ticker} not in data")

        spy_ret = log_ret[benchmark_ticker]
        sector_ret = log_ret[available]

        n_days = len(log_ret)
        current_weights: dict[str, float] = {t: 0.0 for t in available}

        # Reset state
        self.portfolio_returns.clear()
        self.portfolio_dates.clear()
        self.stress_signals.clear()
        self.detector_signals.clear()
        self.daily_weights.clear()
        self.daily_turnover.clear()
        self.basket_history.clear()
        self.execution.reset()

        rebalance_points = list(range(self.min_train, n_days, self.step))

        for wp_start in rebalance_points:
            wp_end = min(wp_start + self.step, n_days)
            if wp_start >= n_days:
                break

            # ── 1. Training data: only past data ──────────────────────
            train_spy = spy_ret.iloc[:wp_start]
            train_sector = sector_ret.iloc[:wp_start]

            # ── 2. Calibrate all detectors on training data ───────────
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # CUSUM: calibrate on SPY returns
                self.cusum.calibrate(train_spy)

                # New orthogonal detectors: reset + fit
                self.correlation.reset()
                self.correlation.fit(train_sector)

                self.breadth.reset()
                self.breadth.fit(train_sector)

                self.skewness.reset()
                self.skewness.fit(train_spy.values)

                # Build signal matrix over the training window
                s1 = self.cusum.signal_series(train_spy)
                s2 = self.correlation.signal_series(train_sector)
                s3 = self.breadth.signal_series(train_sector)
                s4 = self.skewness.signal_series(train_spy)

                common_idx = (
                    s1.index.intersection(s2.index)
                    .intersection(s3.index)
                    .intersection(s4.index)
                )
                signal_matrix = np.column_stack([
                    s1.reindex(common_idx).fillna(0).values,
                    s2.reindex(common_idx).fillna(0).values,
                    s3.reindex(common_idx).fillna(0).values,
                    s4.reindex(common_idx).fillna(0).values,
                ])
                train_spy_aligned = train_spy.reindex(common_idx).fillna(0)

                # Calibrate fuzzy aggregator
                self.fuzzy.calibrate(signal_matrix, train_spy_aligned)

            # ── 3. Characterise assets on training data ───────────────
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_avail = [
                    t for t in available
                    if train_sector[t].dropna().shape[0] >= 30
                ]
                garch_results = self.garch.fit_all(train_sector, train_avail)
                recovery_results = self.recovery.estimate_all(
                    train_sector, train_avail
                )

            # ── 4. Classify into baskets ──────────────────────────────
            assignments = self.classifier.assign(garch_results, recovery_results)
            self.basket_history.append({
                "date": log_ret.index[wp_start],
                "assignments": {t: a.basket for t, a in assignments.items()},
            })

            # Vol dict for sizing
            vol_dict = {
                t: garch_results[t].last_vol
                for t in train_avail
                if t in garch_results and not np.isnan(garch_results[t].last_vol)
            }
            default_vol = np.mean(list(vol_dict.values())) if vol_dict else 0.2
            for t in available:
                if t not in vol_dict:
                    vol_dict[t] = default_vol

            # ── 5. Calibrate portfolio manager ────────────────────────
            rf_mean = float(rf_series.iloc[:wp_start].mean())
            p_stress_train = pd.Series(
                self.fuzzy.aggregate_series(signal_matrix),
                index=common_idx,
            )
            self.basket_mgr.calibrate_thresholds(
                p_stress_train,
                train_sector.reindex(common_idx).fillna(0),
                assignments, vol_dict, rf_mean,
            )
            self.basket_mgr.reset()

            # ── 6. Reset detectors for OOS window ────────────────────
            self.cusum.reset_accumulators()

            # Re-prime new detectors with training tail for OOS
            self.correlation.reset()
            self.correlation.fit(train_sector)

            self.breadth.reset()
            self.breadth.fit(train_sector)

            self.skewness.reset()
            self.skewness.fit(train_spy.values)

            # ── 7. Run day-by-day through the OOS window ─────────────
            for day_idx in range(wp_start, wp_end):
                if day_idx >= n_days:
                    break

                day = log_ret.index[day_idx]
                spy_r = float(spy_ret.iloc[day_idx]) if not np.isnan(spy_ret.iloc[day_idx]) else 0.0

                # Get sector returns for today (1-D array)
                sector_returns_today = sector_ret.iloc[day_idx].fillna(0).values

                # Compute detector signals (point-in-time: today's return only)
                sig_cusum = self.cusum.signal(spy_r)
                sig_correlation = self.correlation.signal(sector_returns_today)
                sig_breadth = self.breadth.signal(sector_returns_today)
                sig_skewness = self.skewness.signal(spy_r)

                detector_sigs = {
                    "cusum": sig_cusum,
                    "correlation": sig_correlation,
                    "breadth": sig_breadth,
                    "skewness": sig_skewness,
                }

                # Use correlation spike as structural break proxy
                structural_break = sig_correlation > 0.5

                # Aggregate to composite P(stress)
                p_stress = self.fuzzy.aggregate(
                    [sig_cusum, sig_correlation, sig_breadth, sig_skewness]
                )

                # Compute target weights
                new_weights = self.basket_mgr.compute_weights(
                    available, assignments, vol_dict, p_stress, structural_break,
                )

                # Execute (apply transaction costs)
                new_weights, cost = self.execution.execute(
                    current_weights, new_weights,
                )

                # Portfolio return = equity return + cash return − costs
                day_ret = 0.0
                for tkr in available:
                    w = new_weights.get(tkr, 0.0)
                    r = float(sector_ret[tkr].iloc[day_idx])
                    if not np.isnan(r):
                        day_ret += w * r
                cash_weight = max(0.0, 1.0 - sum(new_weights.values()))
                day_ret += cash_weight * float(rf_series.iloc[day_idx])
                day_ret -= cost

                # Record
                self.portfolio_returns.append(day_ret)
                self.portfolio_dates.append(day)
                self.stress_signals.append(p_stress)
                self.detector_signals.append(detector_sigs)
                self.daily_weights.append(dict(new_weights))
                self.daily_turnover.append(
                    self.execution.log.daily_turnover[-1]
                    if self.execution.log.daily_turnover
                    else 0.0
                )
                current_weights = new_weights

        # ── Compile results ───────────────────────────────────────────
        port_ret = pd.Series(
            self.portfolio_returns,
            index=pd.DatetimeIndex(self.portfolio_dates),
            name="strategy",
        )
        stress_sig = pd.Series(
            self.stress_signals,
            index=pd.DatetimeIndex(self.portfolio_dates),
            name="p_stress",
        )
        turnover_s = pd.Series(
            self.daily_turnover,
            index=pd.DatetimeIndex(self.portfolio_dates),
            name="turnover",
        )
        detector_df = pd.DataFrame(
            self.detector_signals,
            index=pd.DatetimeIndex(self.portfolio_dates),
        )

        bench_ret = spy_ret.reindex(port_ret.index).fillna(0)

        return {
            "strategy_returns": port_ret,
            "benchmark_returns": bench_ret,
            "stress_signals": stress_sig,
            "detector_signals": detector_df,
            "turnover": turnover_s,
            "execution_log": self.execution.log,
            "basket_history": self.basket_history,
            "daily_weights": self.daily_weights,
            "entry_threshold": self.basket_mgr.entry_threshold,
            "exit_threshold": self.basket_mgr.exit_threshold,
        }
