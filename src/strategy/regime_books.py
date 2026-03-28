from dataclasses import dataclass, field
from enum import Enum

class Regime(Enum):
    BULL = 0
    TRANSITION = 1
    CRISIS = 2

@dataclass
class StrategyBook:
    """
    Complete strategy specification for a single regime.
    All parameters are used by the StrategyEngine to make allocation decisions.
    """
    regime: Regime

    # ── Equity allocation ──────────────────────────────────────
    target_equity_pct: float          # target total equity weight (e.g., 0.90 for Bull)
    max_single_position: float        # max weight in any one sector ETF
    min_sectors_held: int             # minimum number of sectors with nonzero weight
    momentum_lookback: int            # days for momentum signal (e.g., 63 for Bull, 21 for Crisis)
    prefer_cyclical: bool             # tilt toward cyclical sectors (True in Bull)
    prefer_defensive: bool            # tilt toward defensive sectors (True in Crisis)

    # ── Cash management ────────────────────────────────────────
    target_cash_pct: float            # target cash allocation (e.g., 0.05 for Bull, 0.30 for Crisis)
    max_cash_pct: float               # hard ceiling on cash (e.g., 0.15 for Bull, 0.50 for Crisis)
    deploy_cash_when: str             # condition to deploy cash: "momentum_positive" / "vix_declining" / "always"

    # ── Put option overlay ─────────────────────────────────────
    target_put_budget_pct: float      # target % of portfolio allocated to puts (e.g., 0.03 for Bull)
    put_strike_otm_pct: float         # how far OTM to buy puts (e.g., 0.05 = 5% below current SPY)
    put_tenor_days: int               # target days-to-expiry when buying (e.g., 126 = 6 months)
    put_action: str                   # "accumulate" / "hold" / "monetise"

    # ── Divergence detection ───────────────────────────────────
    divergence_lookback: int          # rolling window for stock-vs-sector comparison
    divergence_threshold: float       # z-score threshold to flag divergence (e.g., -1.5)
    liquidate_divergent: bool         # whether to liquidate divergent positions

    # ── Rebalance aggressiveness ───────────────────────────────
    max_daily_turnover: float         # maximum turnover per day (dampens whipsawing)
    rebalance_speed: float            # fraction of distance to target covered per day (0.1 = slow, 1.0 = instant)


# ── Pre-defined books (these are starting points; the neural policy refines them) ──

BULL_BOOK = StrategyBook(
    regime=Regime.BULL,
    target_equity_pct=0.92,
    max_single_position=0.20,
    min_sectors_held=7,
    momentum_lookback=63,
    prefer_cyclical=True,
    prefer_defensive=False,
    target_cash_pct=0.03,
    max_cash_pct=0.10,
    deploy_cash_when="always",
    target_put_budget_pct=0.03,     # Buy cheap insurance when VIX is low
    put_strike_otm_pct=0.07,        # 7% OTM — deep, cheap protection
    put_tenor_days=126,             # 6-month puts
    put_action="accumulate",        # Steadily buy puts during calm markets
    divergence_lookback=42,
    divergence_threshold=-1.5,
    liquidate_divergent=True,       # Boot underperformers, redeploy capital
    max_daily_turnover=0.15,
    rebalance_speed=0.3,
)

TRANSITION_BOOK = StrategyBook(
    regime=Regime.TRANSITION,
    target_equity_pct=0.70,
    max_single_position=0.15,
    min_sectors_held=8,
    momentum_lookback=42,
    prefer_cyclical=False,
    prefer_defensive=True,
    target_cash_pct=0.15,
    max_cash_pct=0.30,
    deploy_cash_when="vix_declining",
    target_put_budget_pct=0.05,     # Increase hedge budget
    put_strike_otm_pct=0.05,        # Closer to ATM — more expensive but more protective
    put_tenor_days=90,
    put_action="hold",              # Hold existing puts, don't buy more aggressively
    divergence_lookback=21,
    divergence_threshold=-1.0,      # More sensitive to divergence
    liquidate_divergent=True,
    max_daily_turnover=0.20,
    rebalance_speed=0.5,
)

CRISIS_BOOK = StrategyBook(
    regime=Regime.CRISIS,
    target_equity_pct=0.45,
    max_single_position=0.12,
    min_sectors_held=5,
    momentum_lookback=21,
    prefer_cyclical=False,
    prefer_defensive=True,
    target_cash_pct=0.30,
    max_cash_pct=0.50,
    deploy_cash_when="momentum_positive",  # only deploy cash when bottom seems in
    target_put_budget_pct=0.02,     # Puts are now expensive (VIX is high)
    put_strike_otm_pct=0.03,
    put_tenor_days=63,
    put_action="monetise",          # SELL existing puts for massive profit, reinvest in cheap equities
    divergence_lookback=10,
    divergence_threshold=-0.5,      # Extremely sensitive — cut losses fast
    liquidate_divergent=True,
    max_daily_turnover=0.30,        # Allow faster repositioning in crisis
    rebalance_speed=0.8,
)

STRATEGY_BOOKS = {
    Regime.BULL: BULL_BOOK,
    Regime.TRANSITION: TRANSITION_BOOK,
    Regime.CRISIS: CRISIS_BOOK,
}
