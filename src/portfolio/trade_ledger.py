from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd

class TradeType(Enum):
    EQUITY_BUY = "equity_buy"
    EQUITY_SELL = "equity_sell"
    EQUITY_REBALANCE = "equity_rebalance"
    PUT_BUY = "put_buy"
    PUT_SELL = "put_sell"
    PUT_EXPIRE = "put_expire"
    CASH_DEPLOY = "cash_deploy"
    CASH_RAISE = "cash_raise"
    DIVERGENCE_LIQUIDATION = "divergence_liquidation"

@dataclass
class TradeRecord:
    trade_id: str
    date: pd.Timestamp
    trade_type: TradeType
    ticker: str                       # e.g., "XLK" or "SPY_PUT_126d"
    direction: str                    # "BUY" or "SELL"
    quantity: float                   # weight delta (for equities) or contracts (for puts)
    price: float                      # price at execution
    cost: float                       # transaction cost
    regime: str                       # regime at time of trade
    p_stress: float                   # stress level at time of trade
    reason: str                       # human-readable reason
    pnl: float = 0.0                  # realized P&L (filled on close)

class TradeLedger:
    """
    Complete audit trail of every trade the system makes.

    Maintains:
    - Full trade history with timestamps, types, and P&L
    - Per-ticker position tracking (average cost, unrealized P&L)
    - Regime attribution (which regime triggered each trade)
    - Summary statistics (win rate, avg win/loss, best/worst trades)
    """

    def __init__(self):
        self.trades: list[TradeRecord] = []
        self._next_id: int = 0
        self.positions: dict[str, dict] = {}  # ticker -> {weight, avg_cost, entry_date}

    def record_trade(self, date: pd.Timestamp, trade_type: TradeType,
                     ticker: str, direction: str, quantity: float,
                     price: float, cost: float, regime: str,
                     p_stress: float, reason: str, pnl: float = 0.0) -> str:
        """Record a single trade. Returns trade_id."""
        trade_id = f"T{self._next_id:06d}"
        self._next_id += 1
        record = TradeRecord(
            trade_id=trade_id, date=date, trade_type=trade_type,
            ticker=ticker, direction=direction, quantity=quantity,
            price=price, cost=cost, regime=regime, p_stress=p_stress,
            reason=reason, pnl=pnl,
        )
        self.trades.append(record)
        return trade_id

    def record_weight_changes(self, date: pd.Timestamp,
                              old_weights: dict[str, float],
                              new_weights: dict[str, float],
                              prices: dict[str, float],
                              regime: str, p_stress: float,
                              cost: float, reason: str = "rebalance"):
        """
        Compare old and new weights, record individual trades for each change.
        """
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        for ticker in sorted(all_tickers):
            old_w = old_weights.get(ticker, 0.0)
            new_w = new_weights.get(ticker, 0.0)
            delta = new_w - old_w
            if abs(delta) < 1e-6:
                continue
            direction = "BUY" if delta > 0 else "SELL"
            trade_type = TradeType.EQUITY_BUY if delta > 0 else TradeType.EQUITY_SELL
            self.record_trade(
                date=date, trade_type=trade_type, ticker=ticker,
                direction=direction, quantity=abs(delta),
                price=prices.get(ticker, 0.0), cost=cost * abs(delta),
                regime=regime, p_stress=p_stress, reason=reason,
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Export full trade history as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                'trade_id': t.trade_id, 'date': t.date, 'type': t.trade_type.value,
                'ticker': t.ticker, 'direction': t.direction, 'quantity': t.quantity,
                'price': t.price, 'cost': t.cost, 'regime': t.regime,
                'p_stress': round(t.p_stress, 4), 'reason': t.reason, 'pnl': t.pnl,
            }
            for t in self.trades
        ])

    def summary(self) -> dict:
        """Summary statistics for the trade ledger."""
        df = self.to_dataframe()
        if df.empty:
            return {}
        return {
            'total_trades': len(df),
            'equity_trades': len(df[df['type'].str.startswith('equity')]),
            'put_trades': len(df[df['type'].str.startswith('put')]),
            'divergence_liquidations': len(df[df['type'] == 'divergence_liquidation']),
            'total_costs': df['cost'].sum(),
            'realized_pnl': df['pnl'].sum(),
            'trades_per_regime': df.groupby('regime').size().to_dict(),
        }
