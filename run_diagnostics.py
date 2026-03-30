"""
Diagnostic script — validates sanity checks for orthogonal detector ensemble.

Usage:
    python run_diagnostics.py
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from collections import Counter
import config
from src.backtest.walk_forward import WalkForwardEngine
from src.backtest.metrics import compute_metrics

prices = pd.read_csv('data/raw/etf_prices.csv', index_col=0, parse_dates=True)
ff_data = pd.read_csv('data/raw/ff5_daily.csv', index_col=0, parse_dates=True)

engine = WalkForwardEngine()
results = engine.run(prices, ff_data=ff_data)

port_ret = results['strategy_returns']
bench_ret = results['benchmark_returns']
stress = results['stress_signals']

rf_daily = float(ff_data['RF'].reindex(port_ret.index).mean())
m = compute_metrics(port_ret, rf_daily, turnover_series=results['turnover'], stress_signals=stress)
m_bench = compute_metrics(bench_ret, rf_daily)

print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

checks = []

# 1. Max DD — strategy must be shallower than SPY (less negative)
strat_dd = m.max_drawdown
bench_dd = m_bench.max_drawdown
ok = strat_dd > bench_dd
checks.append(('Max DD < SPY', ok, f'Strategy={strat_dd:.4f} vs SPY={bench_dd:.4f}'))

# 2. Sharpe
sharpe = m.sharpe_ratio
checks.append(('Sharpe >= 0.55', sharpe >= 0.55, f'{sharpe:.3f}'))

# 3. Calmar
calmar = m.calmar_ratio
checks.append(('Calmar >= 0.30', calmar >= 0.30, f'{calmar:.3f}'))

# 4. P(stress) distribution — print empirical distribution instead of hard thresholds
print("\n── P(stress) Distribution ──────────────────────────────────────")
print(f"  P(stress) > 0.3: {(stress > 0.3).mean():.1%} of days")
print(f"  P(stress) > 0.5: {(stress > 0.5).mean():.1%} of days")
print(f"  P(stress) > 0.7: {(stress > 0.7).mean():.1%} of days")
print(f"  P(stress) > 0.9: {(stress > 0.9).mean():.1%} of days")
print()

# 5. COVID cash allocation
covid_start = pd.Timestamp('2020-02-20')
covid_end = pd.Timestamp('2020-04-30')
weights_df = pd.DataFrame(results['daily_weights'], index=port_ret.index)
covid_weights = weights_df.loc[covid_start:covid_end]
if len(covid_weights) > 0:
    min_sum = covid_weights.sum(axis=1).min()
    checks.append(('COVID cash (sum<0.6)', min_sum < 0.6, f'min weight sum={min_sum:.3f}'))
else:
    checks.append(('COVID cash', False, 'No COVID data in OOS window'))

# 6. Sigmoid crossover check — no degenerate sigmoids
sigmoid_ok = True
sigmoid_detail = []
for i, name in enumerate(['cusum', 'correlation', 'breadth', 'skewness']):
    c = engine.fuzzy.sigmoid_params[i, 1]
    sigmoid_detail.append(f'{name}={c:.3f}')
    if c <= 0.06 or c >= 0.94:
        sigmoid_ok = False
checks.append(('No degenerate sigmoids', sigmoid_ok,
               '[' + ', '.join(sigmoid_detail) + ']'))

# 7. Basket distribution — every window must have A>=1, B>=1.
basket_ok = True
first_failure = ''
for bh in results['basket_history']:
    bc = Counter(bh['assignments'].values())
    if bc.get('A', 0) < 1 or bc.get('B', 0) < 1:
        basket_ok = False
        first_failure = f"date={bh['date'].date()} A={bc.get('A',0)} B={bc.get('B',0)}"
        break
checks.append(('Basket: A>=1, B>=1 always', basket_ok,
               '' if basket_ok else f'first failure: {first_failure}'))

print()
for name, passed, detail in checks:
    status = 'PASS' if passed else 'FAIL'
    print(f'  [{status}] {name}: {detail}')

n_pass = sum(1 for _, p, _ in checks if p)
print(f'\n{n_pass}/{len(checks)} checks passed.')

# ── Additional context ────────────────────────────────────────────────────────
print()
print("── Key metrics ──────────────────────────────────────────────────")
print(f"  Sharpe:      Strategy={m.sharpe_ratio:.3f}   SPY={m_bench.sharpe_ratio:.3f}")
print(f"  Max DD:      Strategy={m.max_drawdown:.2%}  SPY={m_bench.max_drawdown:.2%}")
print(f"  Calmar:      Strategy={m.calmar_ratio:.3f}   SPY={m_bench.calmar_ratio:.3f}")
print(f"  Ann Return:  Strategy={m.annualised_return:.2%}  SPY={m_bench.annualised_return:.2%}")
print(f"  Turnover:    {m.annualised_turnover:.2f}x annualised")
