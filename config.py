"""
Configuration module — single source of truth for all parameters.

All thresholds and boundaries used in the system are estimated from data
at runtime. This file contains only structural parameters (dates, tickers,
fixed transaction costs) that define the experimental setup.
"""

# ── Date range ────────────────────────────────────────────────────────────────
DATA_START_DATE: str = "2007-01-01"   # Pre-GFC for regime diversity
DATA_END_DATE: str = "2025-12-31"

# ── Walk-forward parameters (trading days) ────────────────────────────────────
WALK_FORWARD_MIN_TRAIN: int = 504    # 2 years minimum training window
WALK_FORWARD_STEP: int = 63          # Rebalance quarterly

# ── Transaction costs ─────────────────────────────────────────────────────────
TRANSACTION_COST_BPS: int = 10       # 10 basis points per trade

# ── Risk-free rate (fallback; overridden by Fama-French RF when available) ────
RISK_FREE_RATE_ANNUAL: float = 0.05

# ── Universe ──────────────────────────────────────────────────────────────────
SECTOR_ETFS: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Communication Services",
}

BENCHMARK: str = "SPY"
VIX_TICKER: str = "^VIX"
