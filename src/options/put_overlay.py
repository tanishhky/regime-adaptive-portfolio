from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
from src.options.black_scholes import bs_put_price, bs_put_delta

@dataclass
class PutPosition:
    """A single synthetic put option position."""
    position_id: str                 # unique ID
    open_date: pd.Timestamp
    strike: float                    # K
    expiry_date: pd.Timestamp        # when the put expires
    notional_contracts: float        # number of "contracts" (1 contract = 100 shares notional)
    cost_basis: float                # total cost paid to open this position
    current_value: float = 0.0       # mark-to-market value today
    is_open: bool = True
    close_date: Optional[pd.Timestamp] = None
    close_value: float = 0.0
    realized_pnl: float = 0.0
    regime_at_open: str = ""         # which regime triggered this purchase

@dataclass
class PutOverlayState:
    """Complete state of the put overlay at any point in time."""
    positions: list[PutPosition] = field(default_factory=list)
    total_invested: float = 0.0      # cumulative capital deployed to puts
    total_recovered: float = 0.0     # cumulative capital recovered from put sales
    _next_id: int = 0

class PutOverlayManager:
    """
    Manages synthetic SPY put option positions alongside the equity portfolio.

    Lifecycle:
    1. BULL regime: accumulate cheap 6-month OTM puts (VIX is low -> puts are cheap)
    2. TRANSITION: hold, let positions mature, maybe roll near-expiry ones
    3. CRISIS: VIX spikes -> existing puts gain massive value -> monetise (close positions)
       Reinvest proceeds into cheap equities or hold as cash

    Capital allocation:
    - The strategy book specifies target_put_budget_pct of total portfolio value
    - New purchases are sized so total put exposure stays near this target
    - Puts that expire worthless are just the cost of insurance (sunk cost)
    - Puts monetised during crisis generate alpha (this is the payoff)
    """

    def __init__(self):
        self.state = PutOverlayState()

    def daily_update(self, date: pd.Timestamp, spy_price: float,
                     vix: float, rf_annual: float, portfolio_value: float,
                     book: 'StrategyBook', regime_name: str) -> dict: # type: ignore
        """
        Process one trading day. Returns dict of actions taken and P&L.

        Steps:
        1. Mark-to-market all open positions using current BS price
        2. Close expired positions (T <= 0) - exercise if ITM, expire if OTM
        3. Execute strategy book actions:
           - "accumulate": buy new puts if below target budget
           - "hold": do nothing, let positions run
           - "monetise": close all positions with unrealised profit > 50%
        """
        sigma = vix / 100.0
        actions = {
            'mtm_pnl': 0.0, 'positions_opened': 0, 'positions_closed': 0,
            'capital_deployed': 0.0, 'capital_recovered': 0.0, 'total_put_value': 0.0
        }

        # 1. Mark-to-market
        for pos in self.state.positions:
            if not pos.is_open:
                continue
            T_years = (pos.expiry_date - date).days / 365.0
            if T_years <= 0:
                # Expired - exercise if ITM
                intrinsic = max(pos.strike - spy_price, 0.0) * pos.notional_contracts * 100
                pos.current_value = intrinsic
                pos.is_open = False
                pos.close_date = date
                pos.close_value = intrinsic
                pos.realized_pnl = intrinsic - pos.cost_basis
                self.state.total_recovered += intrinsic
                actions['positions_closed'] += 1
                actions['capital_recovered'] += intrinsic
            else:
                new_val = bs_put_price(spy_price, pos.strike, T_years, sigma, rf_annual)
                new_val *= pos.notional_contracts * 100  # scale by contract size
                actions['mtm_pnl'] += (new_val - pos.current_value)
                pos.current_value = new_val

        # 2. Compute total put exposure
        total_put_value = sum(p.current_value for p in self.state.positions if p.is_open)
        actions['total_put_value'] = total_put_value

        # 3. Execute strategy
        if book.put_action == "accumulate":
            target_put_value = portfolio_value * book.target_put_budget_pct
            deficit = target_put_value - total_put_value
            if deficit > portfolio_value * 0.005:  # only buy if meaningfully below target
                strike = spy_price * (1 - book.put_strike_otm_pct)
                T_years = book.put_tenor_days / 252.0
                put_price_per_share = bs_put_price(spy_price, strike, T_years, sigma, rf_annual)
                if put_price_per_share > 0.01:  # sanity check
                    n_contracts = deficit / (put_price_per_share * 100)
                    n_contracts = max(0.1, n_contracts)  # minimum position
                    cost = put_price_per_share * n_contracts * 100
                    expiry = date + pd.Timedelta(days=book.put_tenor_days)
                    pos = PutPosition(
                        position_id=f"PUT_{self.state._next_id:04d}",
                        open_date=date, strike=strike, expiry_date=expiry,
                        notional_contracts=n_contracts, cost_basis=cost,
                        current_value=cost, regime_at_open=regime_name,
                    )
                    self.state.positions.append(pos)
                    self.state.total_invested += cost
                    self.state._next_id += 1
                    actions['positions_opened'] += 1
                    actions['capital_deployed'] += cost

        elif book.put_action == "monetise":
            for pos in self.state.positions:
                if not pos.is_open:
                    continue
                profit_pct = (pos.current_value - pos.cost_basis) / max(pos.cost_basis, 1e-6)
                if profit_pct > 0.50:  # 50%+ profit -> take it
                    pos.is_open = False
                    pos.close_date = date
                    pos.close_value = pos.current_value
                    pos.realized_pnl = pos.current_value - pos.cost_basis
                    self.state.total_recovered += pos.current_value
                    actions['positions_closed'] += 1
                    actions['capital_recovered'] += pos.current_value

        return actions

    def get_total_value(self) -> float:
        """Total current value of all open put positions."""
        return sum(p.current_value for p in self.state.positions if p.is_open)

    def get_total_cost_basis(self) -> float:
        """Total cost basis of all open positions."""
        return sum(p.cost_basis for p in self.state.positions if p.is_open)

    def get_realized_pnl(self) -> float:
        """Total realized P&L from closed positions."""
        return sum(p.realized_pnl for p in self.state.positions if not p.is_open)
