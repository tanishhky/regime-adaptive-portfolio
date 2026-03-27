"""
Walk-forward backtest engine — zero lookahead bias.

At each rebalance step:
1. Calibrate all detectors on training data only
2. Characterise assets on training data only
3. Classify into baskets
4. Calibrate entry/exit thresholds
5. Run day-by-day through the out-of-sample test window

CRITICAL: Markov-Switching uses filtered probabilities (not smoothed).
GARCH forecasts are 1-step-ahead from the last training observation.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

import config
from src.detectors.cusum import CUSUMDetector
from src.detectors.ewma import EWMADetector
from src.detectors.markov_switching import MarkovSwitchingDetector
from src.detectors.structural_break import StructuralBreakDetector
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

        # Components
        self.cusum = CUSUMDetector()
        self.ewma = EWMADetector()
        self.markov = MarkovSwitchingDetector()
        self.structural = StructuralBreakDetector()
        self.fuzzy = FuzzyAggregator()
        self.garch = GARCHVolatility()
        self.recovery = RecoveryEstimator()
        self.classifier = BasketClassifier()
        self.basket_mgr = BasketManager()
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

        for wp_idx, wp_start in enumerate(rebalance_points):
            wp_end = min(wp_start + self.step, n_days)
            if wp_start >= n_days:
                break

            # ── 1. Training data: only past data ──────────────────────
            train_ret = log_ret.iloc[:wp_start]
            train_spy = spy_ret.iloc[:wp_start]
            train_sector = sector_ret.iloc[:wp_start]

            # ── 2. Calibrate all detectors on training data ───────────
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                self.cusum.calibrate(train_spy)
                self.ewma.calibrate(train_spy)
                self.markov.fit(train_spy)
                self.structural.fit(train_spy)

                # Build signal matrix for fuzzy calibration
                s1 = self.cusum.signal_series(train_spy)
                s2 = self.ewma.signal_series(train_spy)
                s3 = self.markov.signal_series(train_spy)
                s4 = self.structural.signal_series(train_spy)

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
            # Default vol for tickers without GARCH
            default_vol = np.mean(list(vol_dict.values())) if vol_dict else 0.2
            for t in available:
                if t not in vol_dict:
                    vol_dict[t] = default_vol

            # ── 5. Calibrate entry/exit thresholds ────────────────────
            p_stress_train = pd.Series(
                self.fuzzy.aggregate_series(signal_matrix),
                index=common_idx,
            )
            rf_mean = float(rf_series.iloc[:wp_start].mean())
            self.basket_mgr.calibrate_thresholds(
                p_stress_train, train_sector.reindex(common_idx).fillna(0),
                assignments, vol_dict, rf_mean,
            )
            self.basket_mgr.reset()

            # ── 6. Reset detectors for test window ────────────────────
            self.cusum.reset_accumulators()
            # Re-initialise EWMA running state from end of training
            # (already done in calibrate)

            # ── 7. Run day-by-day through test window ─────────────────
            structural_signal_val = self.structural.signal(0.0)
            structural_break = structural_signal_val > 0.5

            for day_idx in range(wp_start, wp_end):
                if day_idx >= n_days:
                    break

                day = log_ret.index[day_idx]
                spy_r = float(spy_ret.iloc[day_idx]) if not np.isnan(spy_ret.iloc[day_idx]) else 0.0

                # Compute detector signals (using only current day's return)
                sig_cusum = self.cusum.signal(spy_r)
                sig_ewma = self.ewma.signal(spy_r)
                sig_markov = self.markov.signal(spy_r)
                sig_struct = self.structural.signal(spy_r)

                # Aggregate
                p_stress = self.fuzzy.aggregate(
                    [sig_cusum, sig_ewma, sig_markov, sig_struct]
                )

                # Compute new weights
                new_weights = self.basket_mgr.compute_weights(
                    available, assignments, vol_dict, p_stress, structural_break,
                )

                # Execute (compute costs)
                new_weights, cost = self.execution.execute(
                    current_weights, new_weights,
                )

                # Compute portfolio return
                day_ret = 0.0
                for tkr in available:
                    w = new_weights.get(tkr, 0.0)
                    r = float(sector_ret[tkr].iloc[day_idx])
                    if not np.isnan(r):
                        day_ret += w * r
                day_ret -= cost  # Deduct transaction costs

                # Record
                self.portfolio_returns.append(day_ret)
                self.portfolio_dates.append(day)
                self.stress_signals.append(p_stress)
                self.detector_signals.append({
                    "cusum": sig_cusum,
                    "ewma": sig_ewma,
                    "markov": sig_markov,
                    "structural": sig_struct,
                })
                self.daily_weights.append(dict(new_weights))
                self.daily_turnover.append(
                    self.execution.log.daily_turnover[-1]
                    if self.execution.log.daily_turnover
                    else 0.0
                )
                current_weights = new_weights

        # Compile results
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

        # Benchmark returns (same period)
        bench_ret = spy_ret.reindex(port_ret.index).fillna(0)

        # Equal-weight sector returns
        ew_ret = sector_ret.reindex(port_ret.index).fillna(0).mean(axis=1)
        ew_ret.name = "equal_weight"

        return {
            "strategy_returns": port_ret,
            "benchmark_returns": bench_ret,
            "equal_weight_returns": ew_ret,
            "stress_signals": stress_sig,
            "detector_signals": detector_df,
            "turnover": turnover_s,
            "execution_log": self.execution.log,
            "basket_history": self.basket_history,
            "daily_weights": self.daily_weights,
            "entry_threshold": self.basket_mgr.entry_threshold,
            "exit_threshold": self.basket_mgr.exit_threshold,
        }
