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
from src.data.fetcher import compute_log_returns
from src.backtest.walk_forward import WalkForwardEngine
from src.backtest.metrics import compute_metrics, metrics_to_df

prices = pd.read_csv('data/raw/etf_prices.csv', index_col=0, parse_dates=True)
ff_data = pd.read_csv('data/raw/ff5_daily.csv', index_col=0, parse_dates=True)

engine = WalkForwardEngine()
results = engine.run(prices, ff_data=ff_data)

port_ret = results['strategy_returns']
bench_ret = results['benchmark_returns']
ew_ret = results['equal_weight_returns']
stress = results['stress_signals']

rf_daily = float(ff_data['RF'].reindex(port_ret.index).mean())
m = compute_metrics(port_ret, rf_daily, turnover_series=results['turnover'], stress_signals=stress)
m_bench = compute_metrics(bench_ret, rf_daily)

print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

checks = []

# 1. Max DD
strat_dd = m.get('max_drawdown', m.get('Max Drawdown', -1.0))
bench_dd = m_bench.get('max_drawdown', m_bench.get('Max Drawdown', -1.0))
ok = strat_dd > bench_dd  # less negative = better
checks.append(('Max DD < SPY', ok, f'Strategy={strat_dd:.4f} vs SPY={bench_dd:.4f}'))

# 2. Sharpe
sharpe = m.get('sharpe', m.get('Sharpe Ratio', 0))
checks.append(('Sharpe >= 0.55', sharpe >= 0.55, f'{sharpe:.3f}'))

# 3. Calmar
calmar = m.get('calmar', m.get('Calmar Ratio', 0))
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

# 6. Detector weights
w = engine.fuzzy.weights
all_above = all(w_i >= 0.03 for w_i in w)
checks.append(('All detector weights >= 0.03', all_above, f'{w}'))

# 7. Basket distribution
basket_ok = True
for bh in results['basket_history']:
    bc = Counter(bh['assignments'].values())
    if bc.get('A', 0) < 2 or bc.get('B', 0) < 1:
        basket_ok = False
        break
checks.append(('Basket: A>=2, B>=1 always', basket_ok, ''))

for name, passed, detail in checks:
    status = 'PASS' if passed else 'FAIL'
    print(f'  [{status}] {name}: {detail}')

n_pass = sum(1 for _, p, _ in checks if p)
print(f'\n{n_pass}/{len(checks)} checks passed.')
