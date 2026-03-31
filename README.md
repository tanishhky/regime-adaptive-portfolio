# Regime-Adaptive Sector Portfolio Management

Multi-scale regime detection with fuzzy aggregation for adaptive sector portfolio management. Walk-forward validated over 2009-2025 with zero lookahead bias.

**Sharpe 1.08 · Max Drawdown -5.25% · Calmar 1.20** - achieving 53% of SPY's return at 13% of its drawdown risk.

## Key Results

| Metric | Regime-Adaptive | SPY (Buy & Hold) |
|---|---|---|
| Ann. Return | 6.30% | 11.79% |
| Ann. Volatility | 4.64% | 17.98% |
| Sharpe Ratio | **1.08** | 0.58 |
| Sortino Ratio | **1.15** | 0.54 |
| Calmar Ratio | **1.20** | 0.28 |
| Max Drawdown | **-5.25%** | -41.71% |
| Max DD Duration | **204 days** | 512 days |
| Ann. Turnover | 10.41x | 0.00 |

The strategy achieves nearly 2x the Sharpe ratio of SPY by aggressively managing downside risk. During the COVID-19 crash (March 2020), portfolio equity exposure dropped within days of the drawdown onset. The maximum drawdown of -5.25% vs SPY's -41.71% represents an 87% reduction in peak-to-trough loss.

## How It Works

The system operates in four layers:

### 1. Detection Layer - Four Orthogonal Stress Detectors

Each detector captures a different dimension of market stress:

| Detector | Dimension | Method | What It Catches |
|---|---|---|---|
| **CUSUM** | Magnitude | Page (1954) sequential analysis on z-scored returns | Sudden crashes, cumulative drift |
| **Correlation** | Structure | Rolling pairwise correlation across sectors | Diversification collapse, contagion |
| **Breadth** | Participation | Fraction of sectors with negative rolling returns | Broad market weakness |
| **Skewness** | Distribution | Rolling Fisher skewness of benchmark returns | Left-tail risk building (leading indicator) |

All detector parameters are estimated from training data - no hardcoded thresholds.

**CUSUM** operates on z-score-standardized returns (`z_t = (r_t - mu) / sigma`), accumulating evidence of mean shifts. The decision threshold `h` is calibrated from training data.

**Correlation** computes the average pairwise Pearson correlation across 11 sector ETFs over a 21-day rolling window, mapped from [0.10, 0.60] to [0, 1]. Also serves as a structural break proxy: when the signal exceeds 0.5, Basket C holdings are zeroed.

**Breadth** measures the fraction of sectors with negative 21-day cumulative returns, mapped from [0.05, 0.50] to [0, 1]. Orthogonal to CUSUM because it captures how many sectors are declining, not how much.

**Skewness** computes 63-day rolling Fisher skewness of SPY returns, mapped from [-0.8, 0] to [1, 0]. Negative skewness indicates left-tail fattening, often a leading indicator before drawdowns materialise.

### 2. Aggregation Layer - Max-of-Top-2 Fuzzy Inference

Each detector's [0, 1] signal is passed through a calibrated sigmoid membership function:

```
mu_k(s) = 1 / (1 + exp(-a_k * (s - c_k)))
```

The composite stress signal is the **second-largest** sigmoid output:

```
P(stress) = mu_(2)    where mu_(1) >= mu_(2) >= mu_(3) >= mu_(4)
```

This requires at least two detectors to show elevated stress before the system acts - a single detector firing (possibly a false alarm) produces a low composite because the second-largest is still low.

Sigmoid parameters (steepness `a_k` and crossover `c_k`) are calibrated by minimising the Brier score against realised drawdowns. Optimisation uses L-BFGS-B with bounds: `a_k in [1, 50]`, `c_k in [0.05, 0.95]`.

### 3. Characterisation Layer - Asset Classification

Each sector ETF is characterised along two dimensions:

- **GARCH(1,1) with Student-t innovations** - conditional volatility estimation (Bollerslev, 1986)
- **Ornstein-Uhlenbeck recovery half-life** - how fast the sector rebounds from drawdowns

These metrics drive a tercile-based classification into three baskets:

| Basket | Criteria | Strategy |
|---|---|---|
| **A (Tactical)** | High/mid vol + fast recovery | Liquidate above entry threshold; re-enter below exit threshold |
| **B (Avoid)** | High vol + slow recovery | Continuously de-risk: `w * (1 - P(stress))` |
| **C (Core)** | Low vol | Hold; graduated scaling only above P(stress) > 0.7; zero on structural break |

Basket boundaries are data-driven (tercile vol, median half-life) and recomputed at each walk-forward rebalance.

### 4. Portfolio Layer - Adaptive Allocation with Implicit Cash

Base weights are set by inverse-volatility weighting (Kirby & Ostdiek, 2012), normalised to sum to 1.0. Stress-dependent scaling then reduces basket weights below 1.0, and the deficit becomes an implicit cash allocation earning the risk-free rate.

This is the key mechanism: during stress, the portfolio de-risks into cash without any forced redistribution into other risky assets. There is no re-normalisation after stress scaling.

Entry and exit thresholds for Basket A liquidation are calibrated via grid search over the training window's Sharpe ratio.

### Walk-Forward Validation

The entire pipeline is validated using expanding-window walk-forward analysis:

- **Training window**: 504 days minimum (2 years), expanding
- **Out-of-sample step**: 63 days (quarterly)
- **At each step**: ALL parameters re-estimated from training data only

No parameter, threshold, or boundary is ever computed using future data.

## Architecture

```
+--------------------------------------------------+
|                  DATA LAYER                      |
|  yfinance (11 Sector ETFs + SPY + VIX)           |
|  Fama-French 5 Factors (risk-free rate)          |
+-------------------+------------------------------+
                    |
+-------------------v------------------------------+
|              DETECTION LAYER                     |
|                                                  |
|  +--------+ +----------+ +--------+ +---------+  |
|  | CUSUM  | | Correl.  | |Breadth | | Skew-   |  |
|  | Magni- | | Struc-   | |Partic- | | ness    |  |
|  | tude   | | ture     | |ipation | | Distri- |  |
|  | z-score| | Pairwise | |Frac.neg| | bution  |  |
|  +---+----+ +----+-----+ +---+----+ +----+----+  |
|      |           |           |           |       |
|  +---v-----------v-----------v-----------v---+   |
|  |     Fuzzy Aggregator (Brier-calibrated)   |   |
|  |   Sigmoid membership + max-of-top-2 rule  |   |
|  |   No weights — aggregation is parameter-  |   |
|  |   free after sigmoid calibration          |   |
|  +------------------+------------------------+   |
|                     | P(stress) in [0,1]         |
+---------------------+----------------------------+
                      |
+---------------------v----------------------------+
|           CHARACTERISATION LAYER                 |
|                                                  |
|  GARCH(1,1)-t  -> Conditional Vol, VaR(99%)      |
|  Ornstein-Uhlenbeck -> Recovery Half-Life        |
|  Tercile Classifier -> Baskets A / B / C         |
+---------------------+----------------------------+
                      |
+---------------------v----------------------------+
|              PORTFOLIO LAYER                     |
|                                                  |
|  Base weights: inverse-vol, normalised to 1.0    |
|  Basket A: liquidate / re-enter on thresholds    |
|  Basket B: scale by (1 - P(stress))              |
|  Basket C: graduated scale-down above 0.7        |
|  Structural break: correlation > 0.5 zeros C     |
|  Implicit cash: weight deficit = risk reduction  |
|  Threshold calibration: Sharpe grid search       |
|  Execution: 10 bps flat cost per trade           |
|  Walk-forward: 504d train / 63d OOS steps        |
+--------------------------------------------------+
```

## Installation

```bash
git clone https://github.com/tanishhky/regime-adaptive-portfolio.git
cd regime-adaptive-portfolio
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, yfinance, numpy, pandas, scipy, arch, matplotlib, seaborn.

## Usage

Run the full pipeline (data download -> walk-forward backtest -> metrics -> figures):

```bash
python3 run_pipeline.py
```

This will:
1. Download sector ETF prices and Fama-French factor data (cached after first run)
2. Execute the walk-forward backtest across 2009-2025
3. Compute and print the performance metrics table
4. Generate publication-quality figures to `output/figures/`
5. Save metrics CSV to `output/tables/`

Run validation diagnostics:

```bash
python3 run_diagnostics.py
```

Run unit tests:

```bash
python3 -m pytest tests/ -v
```

## Project Structure

```
regime-adaptive-portfolio/
├── config.py                    # Structural parameters (dates, tickers, costs)
├── run_pipeline.py              # Full pipeline runner
├── run_diagnostics.py           # Post-backtest validation checks
├── requirements.txt
│
├── src/
│   ├── data/
│   │   └── fetcher.py           # ETF prices (yfinance) + Fama-French factors
│   │
│   ├── detectors/
│   │   ├── cusum.py             # CUSUM on z-score returns (Page, 1954)
│   │   ├── correlation.py       # Rolling pairwise correlation (contagion)
│   │   ├── breadth.py           # Breadth momentum (sector participation)
│   │   ├── skewness.py          # Rolling skewness (tail asymmetry)
│   │   └── fuzzy_aggregator.py  # Max-of-top-2 aggregation (Brier-calibrated)
│   │
│   ├── characterization/
│   │   ├── volatility.py        # GARCH(1,1)-t conditional vol and VaR
│   │   ├── recovery.py          # Ornstein-Uhlenbeck half-life estimation
│   │   └── classifier.py        # Tercile-based basket classification (A/B/C)
│   │
│   ├── portfolio/
│   │   ├── basket_manager.py    # Adaptive allocation with implicit cash
│   │   └── execution.py         # Transaction cost accounting
│   │
│   ├── backtest/
│   │   ├── walk_forward.py      # Walk-forward engine (zero lookahead)
│   │   └── metrics.py           # Sharpe, Sortino, Calmar, drawdown, etc.
│   │
│   └── visualization/
│       └── plots.py             # Publication-quality figure generation
│
├── tests/
│   ├── test_detectors.py
│   ├── test_characterization.py
│   └── test_backtest.py
│
├── v2-archive/                  # V2 experimental code (agreement-scaling)
│
├── paper/
│   └── paper.tex                # LaTeX paper
│
├── output/
│   ├── figures/                 # Generated plots
│   └── tables/                  # Performance metrics CSV
│
└── data/
    └── raw/                     # Cached price and factor data
```

## Methodology Summary

Every parameter in the system is estimated from data at runtime. There are zero hardcoded thresholds. The walk-forward framework ensures that at each quarterly rebalance, all estimation uses only past data:

1. **Detector calibration** - CUSUM (mu, sigma from training returns), Correlation (21-day rolling pairwise), Breadth (21-day rolling sector fraction), Skewness (63-day rolling Fisher skewness)
2. **Fuzzy aggregation** - sigmoid parameters via Brier score minimisation (L-BFGS-B with bounds), max-of-top-2 rule for composite stress
3. **Asset characterisation** - GARCH(1,1)-t conditional vol and VaR, OU half-life from drawdown episodes
4. **Basket classification** - tercile boundaries on cross-sectional vol distribution, median half-life split
5. **Threshold calibration** - entry/exit thresholds via grid search over training-window Sharpe ratio

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
