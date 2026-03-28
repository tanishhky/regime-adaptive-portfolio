# Regime-Adaptive Sector Portfolio Management

Multi-scale regime detection with fuzzy aggregation for adaptive sector portfolio management. Walk-forward validated over 2009–2025 with zero lookahead bias.

**Sharpe 0.78 · Max Drawdown −19.6% · Calmar 0.59** — capturing 98% of SPY's return at 53% of its drawdown.

## Key Results

| Metric | Regime-Adaptive | SPY (Buy & Hold) | Equal-Weight Sector |
|---|---|---|---|
| Ann. Return | 11.61% | 11.79% | 8.52% |
| Ann. Volatility | 13.22% | 17.98% | 16.05% |
| Sharpe Ratio | **0.78** | 0.58 | 0.45 |
| Sortino Ratio | **0.75** | 0.54 | 0.42 |
| Calmar Ratio | **0.59** | 0.28 | 0.18 |
| Max Drawdown | **−19.60%** | −41.71% | −46.40% |
| Max DD Duration | **226 days** | 512 days | 530 days |
| Hit Rate | 91.6% | — | — |

The strategy matches SPY's return while cutting volatility by 26% and max drawdown by 53%. During the COVID-19 crash (March 2020), portfolio equity exposure dropped to 21% — the system moved to 79% cash within days of the drawdown onset and re-entered as recovery signals appeared.

## How It Works

The system operates in four layers:

### 1. Detection Layer — Four Complementary Detectors

Each detector targets a different time-scale of market stress:

| Detector | Time-Scale | Method | What It Catches |
|---|---|---|---|
| **CUSUM** | 5–10 days | Page (1954) sequential analysis on z-score returns | Sudden crashes, flash events |
| **EWMA Crossover** | 21–42 days | Dual EWMA volatility ratio via empirical CDF | Volatility regime shifts |
| **Markov-Switching** | 63–126 days | Hamilton (1989) 2-state model with recursive filter | Bull/bear state transitions |
| **Structural Break** | 252+ days | PELT algorithm (Killick et al., 2012) on cumulative returns | Secular trend changes |

All detector parameters are estimated from training data — no hardcoded thresholds.

**CUSUM** operates on z-score-standardized returns (`z_t = (r_t − μ) / σ`), which makes the decision threshold scale-invariant. A single −5% day produces a signal of ~0.55; three consecutive −2% days reach ~0.54.

**Markov-Switching** runs a recursive Hamilton filter during out-of-sample windows, updating the stress probability every 5 trading days (pseudo-weekly) using the fitted transition matrix and emission parameters. This means the signal evolves in real-time rather than being frozen at the training endpoint.

### 2. Aggregation Layer — Takagi-Sugeno Fuzzy Inference

Detector signals are combined through calibrated sigmoid membership functions into a single composite stress probability P(stress) ∈ [0, 1]:

```
P(stress) = Σ wᵢ · σ(aᵢ · (xᵢ − cᵢ))
```

The weights `wᵢ`, steepness `aᵢ`, and crossover points `cᵢ` are jointly optimized by minimizing the Brier score against realized drawdowns. A 5% minimum weight floor ensures no detector is zeroed out. Optimization uses L-BFGS-B with bounds on all parameters.

### 3. Characterization Layer — Asset Classification

Each sector ETF is characterized along two dimensions:

- **GARCH(1,1)-t conditional volatility** — forward-looking risk estimate
- **Ornstein-Uhlenbeck recovery half-life** — how fast the sector bounces back from drawdowns

These metrics drive a tercile-based classification into three baskets:

| Basket | Criteria | Strategy |
|---|---|---|
| **A (Tactical)** | High/mid vol + fast recovery | Liquidate above entry threshold; re-enter on recovery |
| **B (Avoid)** | High vol + slow recovery | Continuously de-risk: `w × (1 − P(stress))` |
| **C (Core)** | Low vol | Hold; graduated scaling above P(stress) > 0.7 |

Basket boundaries are data-driven (33rd/67th percentile of vol, median half-life) and recomputed at each rebalance.

### 4. Portfolio Layer — Adaptive Allocation with Implicit Cash

Base weights are set by inverse-volatility weighting (Kirby & Ostdiek, 2012), normalized to sum to 1.0. Stress-dependent scaling then reduces basket weights below 1.0, and the deficit becomes an implicit cash allocation earning the risk-free rate. This is the key mechanism: during stress, the portfolio de-risks into cash without any forced redistribution into other risky assets.

Entry and exit thresholds for Basket A liquidation are calibrated via grid search over the training window's Sharpe ratio.

### Walk-Forward Validation

The entire pipeline is validated using expanding-window walk-forward analysis:

- **Training window**: 504 days minimum (2 years), expanding
- **Out-of-sample step**: 63 days (quarterly)
- **At each step**: ALL parameters re-estimated from training data only

No parameter, threshold, or boundary is ever computed using future data.

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  DATA LAYER                      │
│  yfinance (11 Sector ETFs + SPY + VIX)           │
│  Fama-French 5 Factors (risk-free rate)          │
└───────────────────┬──────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────┐
│              DETECTION LAYER                     │
│                                                  │
│  ┌────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐  │
│  │ CUSUM  │ │  EWMA    │ │ Markov │ │ Struct. │  │
│  │ 5-10d  │ │ 21-42d   │ │ 63-126d│ │  252+d  │  │
│  │z-score │ │ MLE-cal. │ │Hamilton│ │  PELT   │  │
│  └───┬────┘ └────┬─────┘ └───┬────┘ └────┬────┘  │
│      │           │           │            │      │
│  ┌───▼───────────▼───────────▼────────────▼───┐  │
│  │        Fuzzy Aggregator (Brier-opt)        │  │
│  │   Sigmoid membership + L-BFGS-B weights    │  │
│  │   5% minimum weight floor per detector     │  │
│  └──────────────────┬────────────────────────┘   │ 
│                     │ P(stress) ∈ [0,1]          │
└─────────────────────┼────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│           CHARACTERIZATION LAYER                 │
│                                                  │
│  GARCH(1,1)-t  → Conditional Vol, VaR(99%)       │
│  Ornstein-Uhlenbeck → Recovery Half-Life         │
│  Tercile Classifier → Baskets A / B / C          │
└─────────────────────┬────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────┐
│              PORTFOLIO LAYER                     │
│                                                  │
│  Base weights: inverse-vol, normalized to 1.0    │
│  Basket A: liquidate / re-enter on thresholds    │
│  Basket B: scale by (1 − P(stress))              │
│  Basket C: graduated scale-down above 0.7        │
│  Implicit cash: weight deficit = risk reduction  │
│  Threshold calibration: Sharpe grid search       │
│  Execution: 10 bps flat cost per trade           │
│  Walk-forward: 504d train / 63d OOS steps        │
└──────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/tanishhky/regime-adaptive-portfolio.git
cd regime-adaptive-portfolio
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, yfinance, numpy, pandas, scipy, statsmodels, arch, ruptures, hmmlearn, matplotlib, seaborn.

## Usage

Run the full pipeline (data download → walk-forward backtest → metrics → figures):

```bash
python run_pipeline.py
```

This will:
1. Download sector ETF prices and Fama-French factor data (cached after first run)
2. Execute the walk-forward backtest across 2009–2025
3. Compute and print the performance metrics table
4. Generate publication-quality figures to `output/figures/`
5. Save metrics CSV to `output/tables/`

Run validation diagnostics:

```bash
python run_diagnostics.py
```

Run unit tests:

```bash
python -m pytest tests/ -v
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
│   │   ├── ewma.py              # Dual-EWMA volatility crossover (RiskMetrics)
│   │   ├── markov_switching.py  # Hamilton 2-state model + recursive filter
│   │   ├── structural_break.py  # PELT change-point detection (Killick, 2012)
│   │   └── fuzzy_aggregator.py  # Takagi-Sugeno aggregation (Brier-optimized)
│   │
│   ├── characterization/
│   │   ├── volatility.py        # GARCH(1,1)-t conditional vol and VaR
│   │   ├── recovery.py          # Ornstein-Uhlenbeck half-life estimation
│   │   └── classifier.py        # Tercile-based basket classification (A/B/C)
│   │
│   ├── portfolio/
│   │   ├── basket_manager.py    # Adaptive allocation with implicit cash
│   │   ├── sizing.py            # Inverse-volatility and risk-parity weights
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

1. **Detector calibration** — CUSUM (μ, σ from training returns), EWMA (λ via MLE), Markov (transition matrix and emissions via EM), Structural Break (BIC penalty from training variance)
2. **Fuzzy aggregation** — sigmoid parameters and detector weights via Brier score minimization (L-BFGS-B with bounds)
3. **Asset characterization** — GARCH(1,1)-t conditional vol and VaR, OU half-life from drawdown episodes
4. **Basket classification** — tercile boundaries on cross-sectional vol distribution, median half-life split
5. **Threshold calibration** — entry/exit thresholds via grid search over training-window Sharpe ratio

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