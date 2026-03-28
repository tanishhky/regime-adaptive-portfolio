import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

def plot_v2_dashboard(results: dict, save_path: str = "v2_dashboard.png"):
    """
    Plots the V2 specific metrics: equity curve, cash drag, hedge performance, and regime gating.
    """
    strat = results['strategy_returns']
    bench = results['benchmark_returns']
    cash_hist = results.get('cash_weight_history', [])
    gate_hist = results.get('regime_probs_history', [])
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    
    # 1. Equity Curves
    ax = axes[0]
    ax.plot(strat.cumsum(), label="V2 Strategy", color='blue')
    ax.plot(bench.cumsum(), label="SPY", color='gray', alpha=0.7)
    ax.set_title("Cumulative Log Returns")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Drawdowns
    ax = axes[1]
    s_cum = strat.cumsum()
    s_dd = s_cum - s_cum.cummax()
    b_cum = bench.cumsum()
    b_dd = b_cum - b_cum.cummax()
    
    ax.plot(s_dd, label="V2 Drawdown", color='red')
    ax.plot(b_dd, label="SPY Drawdown", color='gray', alpha=0.5)
    ax.set_title("Drawdown Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Regime Probs
    ax = axes[2]
    if gate_hist:
        df_probs = pd.DataFrame(gate_hist, columns=['Bull', 'Transition', 'Crisis'])
        ax.stackplot(range(len(df_probs)), df_probs['Bull'], df_probs['Transition'], df_probs['Crisis'],
                     labels=['Bull', 'Transition', 'Crisis'], colors=['#2ecc71', '#f1c40f', '#e74c3c'], alpha=0.6)
        ax.set_title("Neural Regime Gating Network Probabilities")
        ax.legend(loc='upper left')
        ax.set_xlim(0, len(df_probs))
        
    # 4. Cash Yield
    ax = axes[3]
    if cash_hist:
        ax.plot(pd.Series(cash_hist).cumsum() * 100, color='green', label="Cumulative Yield from Cash")
        ax.set_title("Cash Yield Contribution (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
