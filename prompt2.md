# Orthogonal Ensemble v2 — Agreement Scaling, Global Optimization, Sharpe Objective

## Context

You are working on `regime-adaptive-portfolio`. The previous orthogonal detector redesign (correlation, breadth, skewness replacing EWMA, Markov, structural break alongside CUSUM) produced:

- Sharpe: 1.08, Max DD: -5.25%, Return: 6.30%, Vol: 4.64%

The Sharpe and DD look great but the system is **chronically under-invested** — it captures only 53% of SPY's return. The root causes:

1. **Max-of-top-2 aggregation bug**: As implemented, it takes the maximum of all signals (not second-largest), so any single detector leaking residual signal triggers de-risking.
2. **Brier score objective**: Optimizes for stress prediction accuracy, not financial performance. The optimizer found that being permanently defensive produces good Brier scores.
3. **L-BFGS-B local optimizer**: Gets stuck in local minima. All three new detector sigmoids hit the boundary at c=0.950 — the optimizer couldn't explore the full parameter space.
4. **Rolling window training**: 504-day rolling window forgets older regimes. The system doesn't accumulate knowledge from all past experience.

This prompt fixes all four with three interconnected changes:
- **Agreement-scaled response** replaces max-of-top-2
- **Differential evolution** replaces L-BFGS-B
- **Expanding-window Sharpe objective** replaces rolling-window Brier score

**Target performance profile (Archetype 3 — Adaptive Full-Spectrum):**
- Return: 9–11% (capturing 80–95% of SPY)
- Vol: 8–12%
- Sharpe: 0.90–1.10
- Max DD: -10% to -15%
- Fully invested in calm markets, progressively defensive during confirmed stress

---

## Phase 0: Checkpoint

```bash
git add -A
git commit -m "checkpoint: pre-agreement-scaling (Sharpe 1.08, MaxDD -5.25%, Return 6.30%)"
git tag v1.1-orthogonal-baseline
git push origin main --tags
```

Verify: `git log --oneline -1` shows the checkpoint.

---

## Phase 1: Agreement-Scaled Response

**File:** `src/detectors/fuzzy_aggregator.py`

### The Core Idea

Instead of combining detector signals into a single P(stress) via max or weighted average, count how many detectors are in agreement that stress is present, and scale the portfolio response by that agreement level.

Each detector passes its signal through its calibrated sigmoid membership function, producing a transformed value in [0, 1]. A detector "agrees on stress" if its transformed signal exceeds its sigmoid crossover point `c` (the 50% activation level). The agreement count determines the **response fraction** — what proportion of the full de-risking action the system takes.

### Agreement Response Curve

```
agreement_count = number of detectors whose sigmoid_output > 0.5
                  (i.e., raw signal exceeded the learned crossover c)

Response mapping (parameterized as a linear ramp):
  0 detectors agree → response_fraction = 0.00  (fully invested, no action)
  1 detector agrees  → response_fraction = 0.15  (trim 15% of the way toward full de-risk)
  2 detectors agree  → response_fraction = 0.50  (half de-risk)
  3 detectors agree  → response_fraction = 0.80  (aggressive de-risk)
  4 detectors agree  → response_fraction = 1.00  (full de-risk)
```

**CRITICAL:** The response_fraction replaces P(stress) as the signal fed into the basket manager. Everywhere in the codebase that currently reads `p_stress` and uses it to scale basket weights should now receive `response_fraction` instead. The basket scaling math itself does NOT change — only the input signal driving it.

### Intensity Modulation

Within each agreement tier, modulate by the **mean sigmoid output of the agreeing detectors**. This provides granularity within tiers:

```python
def compute_response(self, signals: dict) -> float:
    """
    Compute agreement-scaled response fraction.
    
    Parameters
    ----------
    signals : dict
        Mapping of detector_name → raw signal value (each in [0, 1]).
    
    Returns
    -------
    float
        Response fraction in [0, 1] to be used as p_stress replacement.
    """
    # Step 1: Pass each signal through its calibrated sigmoid
    transformed = {}
    for name, raw_signal in signals.items():
        a = self.sigmoid_params[name]['a']  # steepness
        c = self.sigmoid_params[name]['c']  # crossover
        transformed[name] = 1.0 / (1.0 + np.exp(-a * (raw_signal - c)))
    
    # Step 2: Count agreements (sigmoid output > 0.5 means signal exceeded crossover)
    agreeing = {name: val for name, val in transformed.items() if val > 0.5}
    n_agree = len(agreeing)
    
    # Step 3: Base response from agreement count
    # Linear ramp: 0→0.0, 1→0.15, 2→0.50, 3→0.80, 4→1.00
    ramp = {0: 0.00, 1: 0.15, 2: 0.50, 3: 0.80, 4: 1.00}
    base_response = ramp[n_agree]
    
    # Step 4: Intensity modulation within tier
    # If in tier 1-3, modulate between this tier and next based on mean agreeing intensity
    if n_agree == 0 or n_agree == 4:
        return base_response
    
    next_response = ramp[n_agree + 1]
    mean_intensity = np.mean(list(agreeing.values()))
    # mean_intensity is in (0.5, 1.0) since all agreeing detectors have sigmoid > 0.5
    # Map (0.5, 1.0) → (0.0, 1.0) for interpolation
    interp_factor = (mean_intensity - 0.5) / 0.5
    interp_factor = np.clip(interp_factor, 0.0, 1.0)
    
    return base_response + interp_factor * (next_response - base_response)
```

### What to Remove

- Remove all detector **weight** optimization. No `raw_w` parameters, no softmax normalization, no weight bounds. Weights are gone — agreement counting replaces them.
- Remove the old aggregation function (weighted sum or max-of-top-2).
- **Keep** the sigmoid parameter optimization (a, c per detector). These are still needed — they calibrate when each detector "activates."

### Ramp Parameters

The ramp values `{0: 0.00, 1: 0.15, 2: 0.50, 3: 0.80, 4: 1.00}` are the starting point. In Phase 2, the differential evolution optimizer will search over these as well (they become optimizable parameters). For now, hard-code them as the defaults.

Specifically, parameterize the ramp as 2 free values:
- `r1` = response at 1 detector (default 0.15), bounds [0.05, 0.30]
- `r2` = response at 2 detectors (default 0.50), bounds [0.30, 0.70]
- Response at 3 detectors = `r2 + (1.0 - r2) * 0.6` (derived, not free — keeps monotonicity)
- Response at 0 = 0.0 (fixed)
- Response at 4 = 1.0 (fixed)

This keeps the total optimizable ramp parameters to just 2, not 4.

---

## Phase 2: Differential Evolution with Expanding Window

**File:** `src/detectors/fuzzy_aggregator.py` (the `optimize()` or `fit()` method)

### Replace L-BFGS-B with Differential Evolution

```python
from scipy.optimize import differential_evolution
```

The parameter vector being optimized is now:

```
[a_cusum, c_cusum, a_correlation, c_correlation, a_breadth, c_breadth, a_skewness, c_skewness, r1, r2]
```

That's **10 parameters total**: 4 detectors × 2 sigmoid params + 2 ramp params.

Bounds:
```python
bounds = [
    (1, 50), (0.05, 0.95),   # cusum: a, c
    (1, 50), (0.05, 0.95),   # correlation: a, c
    (1, 50), (0.05, 0.95),   # breadth: a, c
    (1, 50), (0.05, 0.95),   # skewness: a, c
    (0.05, 0.30),            # r1: response at 1 detector
    (0.30, 0.70),            # r2: response at 2 detectors
]
```

Differential evolution settings:
```python
result = differential_evolution(
    objective_function,
    bounds=bounds,
    seed=42,
    maxiter=200,         # enough for 10-D problem
    popsize=20,          # 20 * 10 = 200 candidates per generation
    tol=1e-6,
    mutation=(0.5, 1.5), # dithered mutation for better exploration
    recombination=0.8,
    workers=-1,           # USE ALL CPU CORES — this is the multiprocessing
    updating='deferred',  # required when workers > 1
    polish=True,          # L-BFGS-B polish on the best solution at the end
)
```

**CRITICAL:** `workers=-1` tells scipy to use `multiprocessing` with all available cores. Each candidate evaluation (one call to `objective_function`) runs on a separate core. This is where the speedup comes from. `updating='deferred'` is required for parallel evaluation — it uses generational updates instead of immediate.

**Do NOT remove the `seed=42`** — reproducibility matters.

### No Early Stopping

Remove any existing logic that stops optimization early if a "good enough" result is found. No Sharpe threshold gates, no "if sharpe > X: break" conditions anywhere. Let differential evolution exhaust its full budget (`maxiter=200`) every time. We want the global optimum, not the first acceptable local.

---

## Phase 3: Expanding-Window Sharpe Objective

This is the biggest conceptual change. The optimizer no longer minimizes Brier score on a fixed rolling window. It maximizes risk-adjusted return across all available history.

### 3A: Expanding Window

At each quarterly recalibration point `t`:

**Old (rolling):** Training data = `[t - 504 days, t]`  
**New (expanding):** Training data = `[inception, t]`

The first recalibration uses ~504 days (whatever the initial window is). The second uses ~567 days. By 2020, the training set includes 2009–2020. By 2025, it includes everything.

**Where to change this:** In `src/backtest/walk_forward.py`, find where the training window start date is computed. Currently it's something like:

```python
train_start = current_date - pd.Timedelta(days=504 * calendar_days_per_trading_day)
```

Change to:

```python
train_start = data_start_date  # Always start from the beginning of available data
```

Keep the OOS window at 63 days (quarterly). Only the training window changes.

### 3B: Sharpe Objective Function

The objective function that differential evolution calls must:

1. Unpack the 10-parameter candidate vector into sigmoid params + ramp params.
2. Run a **mini walk-forward simulation** inside the training set using those parameters.
3. Compute the Sharpe ratio of the simulated strategy returns.
4. Return **negative Sharpe** (because differential_evolution minimizes).

**The mini walk-forward inside training is essential.** Do NOT just compute strategy returns on the full training set with the candidate parameters as if they were known throughout — that's fitting to the training set directly. Instead:

```python
def objective_function(params, training_data, sector_returns_train, spy_returns_train):
    """
    Evaluate a candidate parameter set via mini walk-forward inside training data.
    
    Parameters
    ----------
    params : array-like
        10-element vector: [a_cusum, c_cusum, ..., a_skew, c_skew, r1, r2]
    training_data : various
        All available training data up to current recalibration point.
    
    Returns
    -------
    float
        Negative Sharpe ratio (to be minimized).
    """
    # Unpack parameters
    sigmoid_params = {
        'cusum':      {'a': params[0], 'c': params[1]},
        'correlation': {'a': params[2], 'c': params[3]},
        'breadth':    {'a': params[4], 'c': params[5]},
        'skewness':   {'a': params[6], 'c': params[7]},
    }
    ramp_params = {
        0: 0.00,
        1: params[8],                                    # r1
        2: params[9],                                    # r2
        3: params[9] + (1.0 - params[9]) * 0.6,         # r3 derived
        4: 1.00,
    }
    
    # --- Mini walk-forward inside training data ---
    # Split training data into sub-windows.
    # Use 252-day sub-training, 63-day sub-OOS, sliding forward by 63 days.
    # For each sub-window:
    #   1. Fit detectors on sub-training data (GARCH, OU, detector buffers)
    #   2. Run sub-OOS with the candidate sigmoid_params and ramp_params
    #   3. Collect daily strategy returns
    #
    # After all sub-windows: compute Sharpe on the concatenated OOS returns.
    
    n_days = len(spy_returns_train)
    sub_train_len = 252
    sub_oos_len = 63
    
    all_oos_returns = []
    
    start = 0
    while start + sub_train_len + sub_oos_len <= n_days:
        sub_train_end = start + sub_train_len
        sub_oos_end = sub_train_end + sub_oos_len
        
        # Fit detectors on sub-training window
        # (This reuses the same detector fit logic from the main walk-forward,
        #  but with the candidate parameters for sigmoid/ramp)
        
        # Simulate sub-OOS with candidate parameters
        # Collect daily returns
        
        # ... (implementation depends on existing code structure —
        #      extract the daily simulation logic into a callable function
        #      that accepts sigmoid_params and ramp_params as arguments)
        
        all_oos_returns.extend(sub_oos_returns)
        start += sub_oos_len  # slide forward
    
    if len(all_oos_returns) < 63:
        return 0.0  # not enough data, return neutral
    
    # Compute Sharpe
    returns = np.array(all_oos_returns)
    mean_r = returns.mean()
    std_r = returns.std(ddof=1)
    
    if std_r < 1e-10:
        return 0.0
    
    sharpe = (mean_r / std_r) * np.sqrt(252)
    
    # Regularization: penalize chronically defensive behavior
    # Target: average ~85% invested (15% cash on average)
    # If you can track mean portfolio weight in the mini walk-forward, add:
    # penalty = 0.1 * abs(mean_portfolio_weight - 0.85)
    # sharpe -= penalty
    
    return -sharpe  # minimize negative Sharpe = maximize Sharpe
```

### 3C: Extracting the Simulation Logic

**This is the hardest implementation step.** The existing walk-forward engine runs the full backtest end-to-end. You need to extract the **daily simulation** (detector signals → agreement → response_fraction → basket scaling → daily return) into a reusable function that can be called from the objective function with arbitrary sigmoid/ramp parameters.

Create a helper function, roughly:

```python
def simulate_oos_window(
    spy_returns_oos,          # 1-D array, daily SPY returns for this OOS window
    sector_returns_oos,       # DataFrame, daily sector returns for this OOS window
    detectors,                # dict of fitted detector objects (cusum, correlation, breadth, skewness)
    sigmoid_params,           # dict of {name: {a, c}} for each detector
    ramp_params,              # dict of {0: 0.0, 1: r1, 2: r2, 3: r3, 4: 1.0}
    basket_assignments,       # which sectors are in A, B, C
    base_weights,             # normalized inverse-vol weights
    entry_threshold,          # existing entry threshold for basket A/B activation
    rf_daily,                 # daily risk-free rate
    cost_bps=10,              # transaction cost in bps
) -> np.ndarray:
    """
    Simulate one OOS window and return array of daily strategy returns.
    
    This function contains the core daily loop:
    1. Get detector signals
    2. Pass through sigmoids
    3. Count agreement → compute response_fraction
    4. Apply response_fraction to basket scaling
    5. Compute portfolio weights (with implicit cash)
    6. Compute daily return including transaction costs
    """
    daily_returns = []
    prev_weights = base_weights.copy()
    
    for t in range(len(spy_returns_oos)):
        r_t = spy_returns_oos[t]
        sector_r_t = sector_returns_oos.iloc[t].values
        
        # 1. Detector signals
        cusum_sig = detectors['cusum'].signal(r_t)
        corr_sig = detectors['correlation'].signal(sector_r_t)
        breadth_sig = detectors['breadth'].signal(sector_r_t)
        skew_sig = detectors['skewness'].signal(r_t)
        
        signals = {
            'cusum': cusum_sig,
            'correlation': corr_sig,
            'breadth': breadth_sig,
            'skewness': skew_sig,
        }
        
        # 2. Compute response_fraction (replaces p_stress)
        response_fraction = compute_agreement_response(signals, sigmoid_params, ramp_params)
        
        # 3-6. Basket scaling, weight computation, return computation
        # (reuse existing basket_manager logic with response_fraction in place of p_stress)
        # ... 
        
    return np.array(daily_returns)
```

**Implementation guidance:** Look at the existing daily loop in `walk_forward.py`. The logic from "get detector signals" through "compute daily return" is what needs to be extracted. You're not rewriting it — you're wrapping it in a callable function. The existing basket scaling, weight normalization, implicit cash mechanism, and transaction cost logic all stay identical. The only input that changes is the stress signal: instead of `p_stress` from the fuzzy aggregator, it receives `response_fraction` from the agreement function.

### 3D: Regularization

Add a regularization term to prevent chronically defensive solutions (which is exactly what happened in v1). Inside the objective function, after computing Sharpe:

```python
# Track mean portfolio weight (sum of all sector weights, before cash)
# across all sub-OOS windows
mean_weight = np.mean(all_oos_weights)  # should be ~0.85 for healthy behavior

# Penalize deviation from 85% invested
# Coefficient 0.15 means: being at 60% invested (mean_weight=0.60) costs
# 0.15 * |0.60 - 0.85| = 0.0375 Sharpe points
penalty = 0.15 * abs(mean_weight - 0.85)

regularized_sharpe = sharpe - penalty
return -regularized_sharpe
```

**Why 0.85 and not 1.0?** A well-functioning regime-adaptive system should be ~85% invested on average — fully invested most of the time but occasionally in significant cash during genuine stress. Penalizing toward 1.0 would discourage any de-risking. Penalizing toward 0.85 allows the optimizer to find solutions that de-risk when appropriate while staying mostly invested.

**Why 0.15 coefficient?** It's calibrated so that the penalty only matters when the system is very far from the target. At 70% invested: penalty = 0.0225 (small). At 50% invested: penalty = 0.0525 (meaningful). At 30% invested: penalty = 0.0825 (severe). This prevents the "permanently in cash" degenerate solutions without constraining the optimizer during genuine crises.

---

## Phase 4: Multiprocessing for Parameter Sweeps

### 4A: Differential Evolution (already handled)

The `workers=-1` parameter in `differential_evolution()` (Phase 2) already parallelizes the optimization. Each candidate parameter evaluation runs on a separate core. No additional code needed.

### 4B: GARCH/OU Fitting

During each recalibration, GARCH and OU parameters are fitted independently per sector. Parallelize across sectors:

**File:** `src/backtest/walk_forward.py` (or wherever GARCH/OU fitting happens)

```python
from concurrent.futures import ProcessPoolExecutor

def fit_single_sector(sector_name, sector_returns):
    """Fit GARCH + OU for one sector. Returns (sector_name, garch_params, ou_params)."""
    garch_params = fit_garch(sector_returns)
    ou_params = fit_ou(sector_returns)
    return sector_name, garch_params, ou_params

# Parallel fitting across all 11 sectors
with ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(fit_single_sector, name, returns): name
        for name, returns in sector_data.items()
    }
    for future in futures:
        sector_name, garch_params, ou_params = future.result()
        # Store results
```

This cuts GARCH/OU fitting time by ~10x (11 sectors across however many cores are available).

### 4C: Sensitivity Sweeps (Optional, for Robustness)

If you later run parameter sensitivity analysis (different cost assumptions, window sizes, thresholds), parallelize the full backtest across configurations:

```python
from concurrent.futures import ProcessPoolExecutor

def run_backtest_with_config(config):
    """Run full backtest with a specific configuration. Returns metrics dict."""
    # Set up walk-forward with config parameters
    # Run backtest
    # Return {sharpe, max_dd, calmar, return, vol}
    pass

configs = [
    {'cost_bps': 5}, {'cost_bps': 10}, {'cost_bps': 20}, {'cost_bps': 30},
    # etc.
]

with ProcessPoolExecutor() as executor:
    results = list(executor.map(run_backtest_with_config, configs))
```

This is not required for the core changes but include the infrastructure so it's available.

---

## Phase 5: Update Walk-Forward Engine

**File:** `src/backtest/walk_forward.py`

### 5A: Expanding Window

Change the training window from rolling to expanding. Find the line that computes the training start date and change it:

```python
# OLD (rolling):
# train_start_idx = oos_start_idx - train_window_size

# NEW (expanding):
train_start_idx = 0  # Always start from the beginning
```

Keep `oos_start_idx` and `oos_end_idx` exactly as they are. Only the training start changes.

### 5B: Integration with New Aggregator

In the recalibration section of each walk-forward window, replace the old optimizer call with the new differential evolution call:

```python
# OLD:
# aggregator.optimize(signals_train, labels_train, method='L-BFGS-B')

# NEW:
aggregator.optimize_sharpe(
    spy_returns_train=spy_returns_train,
    sector_returns_train=sector_returns_train,
    detectors=detectors,
    basket_assignments=basket_assignments,
    base_weights=base_weights,
    entry_threshold=entry_threshold,
    rf_daily=rf_daily,
    cost_bps=10,
)
```

The `optimize_sharpe` method runs differential evolution internally and stores the best sigmoid_params and ramp_params.

### 5C: Daily Loop Update

In the daily OOS loop, replace the old `p_stress = aggregator.aggregate(signals)` with:

```python
response_fraction = aggregator.compute_response(signals)
```

Then feed `response_fraction` into the basket manager wherever `p_stress` was used.

### 5D: Track Portfolio Weights for Diagnostics

Add tracking of daily total portfolio weight (before cash) to the results:

```python
# Inside the daily loop, after computing weights:
total_sector_weight = sum(weights.values())  # or weights.sum() depending on data structure
daily_weight_sums.append(total_sector_weight)
```

Store in results: `results['daily_weight_sums'] = daily_weight_sums`

This lets us verify the system isn't chronically defensive.

---

## Phase 6: Update Diagnostics

**File:** `run_diagnostics.py`

### 6A: Replace Old Checks

Remove or update:
- "All detector weights >= 0.03" → **Remove entirely** (no weights in agreement model).
- "P(stress) > 0.5 on >= 8% of days" → Replace with investment level check.

Add new checks:

```python
# 1. Investment level check
weight_sums = results['daily_weight_sums']
mean_invested = np.mean(weight_sums)
print(f"Mean portfolio weight: {mean_invested:.1%} (target: 80-95%)")
assert 0.70 <= mean_invested <= 0.98, f"Investment level out of range: {mean_invested:.1%}"

# 2. Sigmoid sanity check (no degenerate crossovers)
for name, params in aggregator.sigmoid_params.items():
    c = params['c']
    print(f"  {name}: a={params['a']:.2f}, c={c:.3f}")
    assert 0.06 <= c <= 0.94, f"{name} sigmoid crossover is degenerate: {c}"

# 3. Agreement distribution
signals_history = results['agreement_counts']  # track this in the daily loop
for n in range(5):
    frac = (np.array(signals_history) == n).mean()
    print(f"  {n} detectors agree: {frac:.1%} of days")

# 4. Response fraction distribution
rf_series = results['response_fractions']  # track this too
print(f"Response fraction > 0.3: {(np.array(rf_series) > 0.3).mean():.1%} of days")
print(f"Response fraction > 0.5: {(np.array(rf_series) > 0.5).mean():.1%} of days")
print(f"Response fraction > 0.8: {(np.array(rf_series) > 0.8).mean():.1%} of days")

# 5. Core performance checks (keep these)
assert sharpe >= 0.85, f"Sharpe below target: {sharpe:.3f}"
assert max_dd >= -0.20, f"Max DD below target: {max_dd:.1%}"  # note: max_dd is negative
```

### 6B: Track Agreement Counts and Response Fractions

In the walk-forward daily loop, add:

```python
# After computing response_fraction:
agreement_counts.append(n_agree)        # the count from compute_response
response_fractions.append(response_fraction)
```

Store both in the results dict.

---

## Phase 7: Update Tests

Add/update tests:

```python
def test_agreement_scaling_zero():
    """No detectors agree → response = 0 (fully invested)."""
    signals = {'cusum': 0.1, 'correlation': 0.1, 'breadth': 0.1, 'skewness': 0.1}
    # With crossovers at ~0.5, all signals are below → 0 agreement
    sigmoid_params = {name: {'a': 10, 'c': 0.5} for name in signals}
    ramp = {0: 0.0, 1: 0.15, 2: 0.50, 3: 0.80, 4: 1.00}
    rf = compute_agreement_response(signals, sigmoid_params, ramp)
    assert rf == 0.0, f"Expected 0.0, got {rf}"

def test_agreement_scaling_full():
    """All 4 detectors agree → response = 1.0 (full de-risk)."""
    signals = {'cusum': 0.9, 'correlation': 0.9, 'breadth': 0.9, 'skewness': 0.9}
    sigmoid_params = {name: {'a': 10, 'c': 0.5} for name in signals}
    ramp = {0: 0.0, 1: 0.15, 2: 0.50, 3: 0.80, 4: 1.00}
    rf = compute_agreement_response(signals, sigmoid_params, ramp)
    assert rf == 1.0, f"Expected 1.0, got {rf}"

def test_agreement_scaling_partial():
    """2 detectors agree → response ≈ 0.50."""
    signals = {'cusum': 0.9, 'correlation': 0.9, 'breadth': 0.1, 'skewness': 0.1}
    sigmoid_params = {name: {'a': 10, 'c': 0.5} for name in signals}
    ramp = {0: 0.0, 1: 0.15, 2: 0.50, 3: 0.80, 4: 1.00}
    rf = compute_agreement_response(signals, sigmoid_params, ramp)
    assert 0.45 <= rf <= 0.80, f"Expected ~0.50-0.80, got {rf}"

def test_agreement_monotonic():
    """More agreement → higher response (monotonicity)."""
    sigmoid_params = {name: {'a': 10, 'c': 0.5}
                      for name in ['cusum', 'correlation', 'breadth', 'skewness']}
    ramp = {0: 0.0, 1: 0.15, 2: 0.50, 3: 0.80, 4: 1.00}
    
    responses = []
    for n_high in range(5):
        sigs = {}
        for i, name in enumerate(['cusum', 'correlation', 'breadth', 'skewness']):
            sigs[name] = 0.9 if i < n_high else 0.1
        responses.append(compute_agreement_response(sigs, sigmoid_params, ramp))
    
    for i in range(len(responses) - 1):
        assert responses[i] <= responses[i + 1], \
            f"Monotonicity violated: {responses}"

def test_differential_evolution_finds_global():
    """DE finds a better optimum than a single L-BFGS-B run on a known test function."""
    from scipy.optimize import differential_evolution, minimize
    
    # Rastrigin function (many local minima, one global at origin)
    def rastrigin(x):
        return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)
    
    bounds = [(-5.12, 5.12)] * 4
    
    de_result = differential_evolution(rastrigin, bounds, seed=42, maxiter=100)
    lbfgsb_result = minimize(rastrigin, x0=[2.0]*4, method='L-BFGS-B',
                              bounds=bounds)
    
    assert de_result.fun < lbfgsb_result.fun + 1.0, \
        f"DE ({de_result.fun:.4f}) should beat L-BFGS-B ({lbfgsb_result.fun:.4f})"

def test_expanding_window():
    """Training start is always the beginning of data, not a rolling offset."""
    # Verify in walk_forward.py that train_start_idx == 0 for all windows
    # This is more of an integration test — check the actual indices
    pass
```

---

## Phase 8: Update Visualization

**File:** `src/visualization/plots.py`

Update detector-related plots to reflect the new system:

1. **Detector signal plot**: Show all 4 detectors (CUSUM, correlation, breadth, skewness) with their sigmoid-transformed values. Label them by name.
2. **Agreement heatmap**: Replace the old detector heatmap with one showing agreement count over time (0–4 scale, color-coded).
3. **Response fraction plot**: Plot `response_fraction` over time alongside SPY price. This replaces the old P(stress) plot.
4. **Investment level plot**: Plot daily total portfolio weight (1.0 = fully invested, 0.0 = all cash) over time.

Keep all existing figure filenames if possible — just update the content. If new figures are needed, add them with descriptive names.

---

## Phase 9: Run and Report

```bash
python3 run_pipeline.py
python3 run_diagnostics.py
```

**Paste the FULL output.** I need to see:

1. Performance metrics table (Sharpe, Max DD, Calmar, Return, Vol, Turnover) for strategy, SPY, and equal-weight
2. Agreement distribution (% of days at each count 0–4)
3. Response fraction distribution (% > 0.3, > 0.5, > 0.8)
4. Mean portfolio weight (investment level)
5. Sigmoid parameters for all 4 detectors (a and c values)
6. Ramp parameters (r1, r2)
7. All validation check results (pass/fail)
8. Any warnings or errors

**Do NOT push to GitHub.** I will review the results and decide.

---

## Build Order (Sequential)

| Step | What | Verify |
|------|------|--------|
| 0 | Git checkpoint + tag | `git log --oneline -1` shows checkpoint |
| 1 | Implement agreement-scaled response in aggregator | Unit tests pass |
| 2 | Replace L-BFGS-B with differential evolution | Optimizer runs without errors |
| 3 | Implement expanding-window Sharpe objective with mini walk-forward | Objective function callable |
| 4 | Add multiprocessing to GARCH/OU fitting | Fitting completes faster |
| 5 | Update walk-forward engine (expanding window, new aggregator integration) | Pipeline runs end-to-end |
| 6 | Update diagnostics (agreement tracking, investment level) | Diagnostics print all new metrics |
| 7 | Update tests | `pytest` passes |
| 8 | Update visualization | Figures generate with new labels |
| 9 | Run pipeline + diagnostics | Paste full output |

---

## Do NOT Modify

- `src/detectors/cusum.py` — anchor detector, untouched
- `src/detectors/correlation.py` — new detector, working correctly
- `src/detectors/breadth.py` — new detector, working correctly
- `src/detectors/skewness.py` — new detector, working correctly
- `src/characterization/classifier.py` — tercile classification is correct
- `src/characterization/garch.py` — conditional vol estimation is correct
- `src/characterization/ou.py` — half-life estimation is correct
- `src/portfolio/basket_manager.py` — basket scaling + implicit cash is correct (but it now receives `response_fraction` instead of `p_stress` — the interface changes, the internal math doesn't)
- Old detector files (`ewma.py`, `markov_switching.py`, `structural_break.py`) — keep for revert, do not import

---

## Expected Outcomes

**If everything works correctly:**
- Sharpe: 0.90–1.10
- Max DD: -10% to -15%
- Return: 9–11% (80–95% of SPY)
- Vol: 8–12%
- Mean portfolio weight: 80–95% (mostly invested, not chronically defensive)
- Agreement count = 0 on ~50–60% of days (calm, fully invested)
- Agreement count >= 2 on ~10–15% of days (confirmed stress, real de-risking)
- No sigmoid at the boundary (all crossovers between 0.10 and 0.90)

**If Sharpe < 0.80:** Check if the regularization penalty is too strong — reduce coefficient from 0.15 to 0.08. Or check if the mini walk-forward inside the objective is using too-short sub-training windows (252 days may not be enough for expanding data — try 504).

**If Return < 7%:** The system is still too defensive. Check mean portfolio weight — if below 0.75, the ramp bounds are too permissive. Tighten r1 upper bound from 0.30 to 0.20, and r2 upper bound from 0.70 to 0.55.

**If Max DD > -20%:** The agreement scaling isn't defensive enough. Check agreement distribution — if the system never reaches 3–4 detectors during COVID, the sigmoid crossovers are too high (detectors not activating). Lower the `c` upper bound from 0.95 to 0.80.

**If optimization takes > 30 minutes per window:** Reduce `popsize` from 20 to 15, or `maxiter` from 200 to 100. The mini walk-forward is the bottleneck — each objective call runs a full sub-backtest. With `workers=-1` on 8 cores and 200 candidates per generation × 200 generations, that's ~5000 objective calls. If each takes 0.5s, that's ~42 min. Reducing to popsize=15, maxiter=100 cuts it to ~10 min.