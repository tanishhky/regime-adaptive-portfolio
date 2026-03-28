#!/usr/bin/env python3
"""
Regime-Adaptive Portfolio V2 Pipeline

Runs the full end-to-end walk-forward backtest using the V2 architecture:
- Hindsight optimal oracle calibration
- Imitation learning
- Neural policy weighting with regime gating
- Put option overlay
- Strategy books & dynamic cash manager
"""

import logging
import pandas as pd
import yfinance as yf
from src.backtest.walk_forward import WalkForwardEngine
from src.backtest.metrics import compute_metrics
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def get_data() -> pd.DataFrame:
    """Download market data."""
    tickers = list(config.SECTOR_ETFS.keys()) + [config.BENCHMARK, config.VIX_TICKER]
    logging.info(f"Downloading data for: {', '.join(tickers)}")
    df = yf.download(
        tickers, start=config.DATA_START_DATE, end=config.DATA_END_DATE,
        group_by='ticker', auto_adjust=False, progress=False
    )
    # Extract Close prices
    close_prices = pd.DataFrame({tkr: df[tkr]['Close'] for tkr in tickers})
    return close_prices.dropna(how='all')

def main() -> None:
    prices = get_data()
    
    # We ideally pull Fama-French data, but for brevity/simplicity we use the fallback rf
    # Provide an empty FF dataframe to force fallback
    ff_data = pd.DataFrame()
    
    logging.info("Initializing Walk-Forward Engine (V2 mode)...")
    engine = WalkForwardEngine(
        min_train=config.WALK_FORWARD_MIN_TRAIN,
        step=config.WALK_FORWARD_STEP,
        use_neural=True,  # This activates V2 StrategyEngine
    )
    
    logging.info("Running parallel walk-forward simulation...")
    results = engine.run(prices, ff_data=ff_data)
    
    strat_ret = results["strategy_returns"]
    bench_ret = results["benchmark_returns"]
    ew_ret = results["equal_weight_returns"]
    cash_hist = results["cash_weight_history"]
    ledger = results["trade_ledger"]
    
    m_strat = compute_metrics(strat_ret, risk_free_daily=config.RISK_FREE_RATE_ANNUAL / 252.0, turnover_series=results.get('turnover'))
    
    logging.info("=" * 40)
    logging.info("V2 PERFORMANCE SUMMARY")
    logging.info("=" * 40)
    logging.info(f"Sharpe Ratio:     {m_strat.sharpe_ratio:.2f}")
    logging.info(f"Annual Return:    {m_strat.annualised_return*100:.1f}%")
    logging.info(f"Max Drawdown:     {m_strat.max_drawdown*100:.1f}%")
    logging.info(f"Win Rate (Days):  {(strat_ret > 0).mean()*100:.1f}%")
    logging.info("-" * 40)
    if isinstance(ledger, pd.DataFrame):
        logging.info(f"Total Trades:     {len(ledger)}")
    if cash_hist:
        avg_cash = sum(cash_hist) / max(1, len(cash_hist)) * 100
        logging.info(f"Avg Cash Yield:   {avg_cash:.2f} bps/period")
    logging.info("=" * 40)
    
    # Save ledger
    if isinstance(ledger, pd.DataFrame) and not ledger.empty:
        ledger.to_csv("v2_trade_ledger.csv", index=False)
        logging.info("Saved V2 trade ledger to v2_trade_ledger.csv")

if __name__ == "__main__":
    main()
