# Regime-Adaptive Sector Portfolio Management

Multi-scale regime detection with fuzzy aggregation for adaptive sector portfolio management. Walk-forward validated with zero lookahead bias.

## Architecture

```
                    ┌─────────────────────────────────┐
                    │        DATA LAYER               │
                    │  yfinance (11 Sector ETFs + SPY) │
                    │  Fama-French 5 Factors           │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │     DETECTION LAYER              │
                    │                                  │
                    │  ┌────────┐  ┌──────┐           │
                    │  │ CUSUM  │  │ EWMA │  5-42 day │
                    │  │ (5-10d)│  │(21-42)│           │
                    │  └───┬────┘  └──┬───┘           │
                    │      │          │                │
                    │  ┌───▼──────────▼───┐           │
                    │  │   Fuzzy Agg.     │ Takagi-   │
                    │  │   (Brier-opt)    │ Sugeno    │
                    │  └───┬──────────┬───┘           │
                    │      │          │                │
                    │  ┌───▼────┐  ┌──▼──────┐        │
                    │  │ Markov │  │Structural│ 63-252+│
                    │  │(63-126)│  │  Break   │  day   │
                    │  └────────┘  └─────────┘        │
                    └────────────┬────────────────────┘
                                 │ P(stress) ∈ [0,1]
                    ┌────────────▼────────────────────┐
                    │   CHARACTERIZATION LAYER         │
                    │                                  │
                    │  GARCH(1,1)-t → Cond. Vol, VaR  │
                    │  Ornstein-Uhlenbeck → Half-Life  │
                    │  Classifier → Baskets A/B/C      │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────▼────────────────────┐
                    │     PORTFOLIO LAYER              │
                    │                                  │
                    │  Basket Manager (adaptive alloc) │
                    │  Inverse-Vol / Risk-Parity Sizing│
                    │  Execution Model (10 bps cost)   │
                    │  Walk-Forward Engine (504d/63d)   │
                    └─────────────────────────────────┘
```

## Key Results

| Metric | Regime-Adaptive | SPY (Buy & Hold) | Equal-Weight Sector |
|---|---|---|---|
| Ann. Return | 9.03% | 11.79% | 8.52% |
| Ann. Volatility | 16.37% | 17.98% | 16.05% |
| Sharpe Ratio | 0.47 | 0.58 | 0.45 |
| Sortino Ratio | 0.44 | 0.54 | 0.42 |
| Calmar Ratio | 0.20 | 0.28 | 0.18 |
| Max Drawdown | -44.37% | -41.71% | -46.40% |
| Hit Rate | 94.9% | -- | -- |
| False Alarm Rate | 5.1% | -- | -- |

## Installation

```bash
git clone https://github.com/tanishkyadav/regime-adaptive-portfolio.git
cd regime-adaptive-portfolio
pip install -r requirements.txt
```

## Usage

Run the full pipeline (data download, backtest, figures):

```bash
python run_pipeline.py
```

This will:
1. Download sector ETF prices and Fama-French data
2. Run the walk-forward backtest (2009-2025)
3. Compute performance metrics
4. Generate all publication figures to `output/figures/`

Run tests:
```bash
python -m pytest tests/ -v
```

## Methodology

**Multi-Scale Detection**: Four detectors target complementary time-scales. CUSUM (Page, 1954) detects rapid mean shifts in 5-10 days. EWMA volatility crossover (RiskMetrics, 1996) captures short-term volatility regime changes over 21-42 days. Hamilton (1989) Markov-switching identifies medium-term bull/bear states over 63-126 days. PELT structural break detection (Killick et al., 2012) flags long-term regime shifts over 252+ days.

**Fuzzy Aggregation**: A Takagi-Sugeno (1985) fuzzy inference system combines detector outputs through calibrated sigmoid membership functions. Weights and parameters are optimized by minimizing the Brier score against realized drawdowns, producing a composite P(stress) signal.

**Asset Characterization**: Each sector ETF is characterized by GARCH(1,1)-t (Bollerslev, 1986) conditional volatility and VaR(99%), plus Ornstein-Uhlenbeck (1930) mean-reversion half-life estimated from drawdown episodes.

**Basket Classification**: Sectors are assigned to Tactical (fast recovery, high vol), Avoid (slow recovery, high vol), or Core (low vol) baskets using data-driven boundaries recomputed at each rebalance.

**Walk-Forward Validation**: All parameters are re-estimated at each quarterly step using only past data. No hardcoded thresholds. Zero lookahead bias.

## Citation

```bibtex
@article{yadav2025regime,
  title={Multi-Scale Regime Detection with Fuzzy Aggregation for Adaptive Sector Portfolio Management},
  author={Yadav, Tanishk},
  year={2025},
  institution={NYU Tandon School of Engineering}
}
```

## License

MIT
