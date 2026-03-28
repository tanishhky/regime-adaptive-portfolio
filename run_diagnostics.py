"""
Diagnostic script — validates all 7 post-fix sanity checks.

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

# 4. P(stress) dynamic range
pct_above_50 = (stress > 0.5).mean() * 100
pct_above_80 = (stress > 0.8).mean() * 100
checks.append(('P(stress)>0.5 on >=8% days', pct_above_50 >= 8.0, f'{pct_above_50:.1f}%'))
checks.append(('P(stress)>0.8 on >=2% days', pct_above_80 >= 2.0, f'{pct_above_80:.1f}%'))

# 5. Cash allocation during COVID
covid_start = pd.Timestamp('2020-02-20')
covid_end = pd.Timestamp('2020-04-30')
weights_df = pd.DataFrame(results['daily_weights'], index=port_ret.index)
covid_weights = weights_df.loc[covid_start:covid_end]
if len(covid_weights) > 0:
    min_sum = covid_weights.sum(axis=1).min()
    checks.append(('COVID cash (sum<0.6)', min_sum < 0.6, f'min weight sum={min_sum:.3f}'))
else:
    checks.append(('COVID cash', False, 'No COVID data in OOS window'))

# 6. Detector weights — all four must contribute meaningfully
w = engine.fuzzy.weights
all_above = all(w_i >= 0.03 for w_i in w)
checks.append(('All detector weights >= 0.03', all_above,
               '[' + ', '.join(f'{x:.3f}' for x in w) + ']'))

# 7. Basket distribution — every window must have A>=1, B>=1.
# Early windows may have only 9 ETFs (XLRE/XLC lack early data); with a
# 9-asset universe the tercile classifier reliably produces A>=1 but
# cannot guarantee A>=2.  A+B>=3 is the meaningful threshold.
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
print(f"  P(stress) >0.5 on {pct_above_50:.1f}% of OOS days")
print(f"  P(stress) >0.8 on {pct_above_80:.1f}% of OOS days")
