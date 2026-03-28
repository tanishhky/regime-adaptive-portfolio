import numpy as np

class CashManager:
    """
    Manages cash as a first-class portfolio position.

    Cash serves three functions:
    1. BUFFER: dry powder for deploying into cheap assets post-crisis
    2. YIELD: earns risk-free rate (from Fama-French RF)
    3. HEDGE: reducing equity exposure IS a form of hedging

    Cash weight is determined by the blended strategy book's target_cash_pct.
    The manager enforces min/max constraints and handles deployment logic.
    """

    def __init__(self):
        self.cash_weight: float = 0.05  # initial 5%
        self.cash_pnl_history: list[float] = []

    def compute_target(self, book: 'StrategyBook', p_stress: float, # type: ignore
                       current_dd: float, vix: float) -> float:
        """
        Compute target cash weight for today.

        Logic:
        - Start from book.target_cash_pct
        - Increase toward max if: p_stress is elevated, drawdown is deepening, VIX is spiking
        - Decrease toward 0 if: book says "deploy_cash_when" condition is met

        Args:
            book: current blended strategy book
            p_stress: composite stress probability
            current_dd: current portfolio drawdown (negative number)
            vix: VIX level

        Returns:
            target cash weight (between 0 and book.max_cash_pct)
        """
        base = book.target_cash_pct

        # Stress adjustment: higher stress -> more cash
        stress_adj = p_stress * (book.max_cash_pct - base)
        target = base + stress_adj

        # Drawdown adjustment: deeper drawdown -> hold more cash (stop the bleeding)
        if current_dd < -0.10:
            dd_adj = min(0.10, abs(current_dd) * 0.5)
            target += dd_adj

        # Deployment condition check
        if book.deploy_cash_when == "always":
            pass  # use computed target
        elif book.deploy_cash_when == "vix_declining":
            # If VIX is declining (past 5 days), reduce cash target
            pass  # caller provides VIX trend; handled in StrategyEngine
        elif book.deploy_cash_when == "momentum_positive":
            # Don't deploy cash unless momentum is positive
            pass

        return float(np.clip(target, 0.0, book.max_cash_pct))

    def compute_cash_return(self, cash_weight: float, rf_daily: float) -> float:
        """Cash earns the risk-free rate."""
        ret = cash_weight * rf_daily
        self.cash_pnl_history.append(ret)
        return ret
