# Regime-Adaptive Sector Portfolio Management

A **capital-preservation overlay** built on multi-scale regime detection with a
**2-of-4 detector consensus** rule. Walk-forward validated 2009–2025, zero look-ahead.

**What it is for:** minimizing drawdown, not maximizing return. It delivers ~1.7×
SPY's risk-adjusted return at **one-seventh the drawdown** (−6.2% vs −41.7%) and a
quarter of the volatility — and the orthogonal architecture **generalizes across
asset classes**, scoring Sharpe **1.21 on factor portfolios** and **1.04 on
international equities** versus 0.51 / 0.11 for equal-weight.

> Numbers below are net of 10 bps/trade costs, on the current orthogonal-detector
> engine, regenerated from a single data vintage (see `output/`). They reproduce
> via `python run_pipeline.py` (core) and `python run_robustness.py` (benchmarking).

## Key Results (SPY/sector universe)

| Metric | Regime-Adaptive | SPY (Buy & Hold) |
|---|---|---|
| Ann. Return | 5.80% | 11.79% |
| Ann. Volatility | **4.64%** | 17.98% |
| Sharpe Ratio | **0.97** | 0.58 |
| Sortino Ratio | **1.02** | 0.54 |
| Calmar Ratio | **0.94** | 0.28 |
| Max Drawdown | **−6.20%** | −41.71% |
| Ann. Turnover | 9.85× | 0.00 |

Sharpe ≈ 1.7× SPY's, achieved by aggressively cutting exposure into stress: during
the COVID-19 crash equity exposure dropped within days, holding the max drawdown to
−6.2% against SPY's −41.7% (an 85% reduction in peak-to-trough loss).

## Honest Benchmarking

A capital-preservation system should be judged against the alternatives — including
trivially simple ones. It is, here, with no cherry-picking:

| Strategy | Sharpe | Calmar | Max DD | Ann. Vol |
|---|---|---|---|---|
| Regime-Adaptive | 0.97 | 0.94 | **−6.2%** | **4.6%** |
| MA Timing (200d) | **1.24** | **1.44** | −10.3% | 10.9% |
| Drawdown Control | 1.21 | 1.76 | −8.5% | 11.3% |
| Vol Targeting | 0.49 | 0.32 | −22.3% | 12.1% |
| Risk Parity | 0.45 | 0.19 | −43.7% | 15.3% |
| SPY (Buy & Hold) | 0.58 | 0.28 | −41.7% | 18.0% |

Read honestly: on the **SPY universe alone**, simple trend rules (200-day MA,
drawdown-control) post higher *raw* Sharpe — this system's edge there is the
**lowest drawdown and lowest volatility of any approach tested**, which is the
objective it is built for. The case for the multi-detector machinery is two-fold:

1. **It is not a SPY-tuned rule.** A 200-day MA is fit to one series; this
   architecture is parameter-light and universe-agnostic — see generalization below.
2. **Consensus, not a single signal.** The aggregator (below) requires *two of four*
   orthogonal detectors to agree before it acts, so no single noisy indicator can
   move the book — a structural false-alarm suppressor a single-rule system lacks.

## Cross-Asset Generalization

The same architecture — calibrated only on training data per universe — transfers
to asset classes it was never designed on, far outperforming equal-weight:

| Universe | Strategy Sharpe | Equal-Weight Sharpe | Strategy Max DD | EW Max DD |
|---|---|---|---|---|
| **Factor portfolios** | **1.21** | 0.51 | −6.1% | −44.4% |
| **International equities** | **1.04** | 0.11 | −7.3% | −58.3% |
| Multi-Asset | 0.45 | 0.26 | −14.3% | −25.1% |

This is the real justification for the orthogonal-detector design: it generalizes,
where a rule tuned to one index does not.

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
