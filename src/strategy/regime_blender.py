import numpy as np
from src.strategy.regime_books import Regime, StrategyBook, STRATEGY_BOOKS

class RegimeBlender:
    """
    Blends strategy parameters from multiple regime books using soft weights.

    Given gate_weights = [p_bull, p_transition, p_crisis] (sum to 1),
    each numerical parameter is interpolated:
        param_blended = p_bull * param_bull + p_trans * param_trans + p_crisis * param_crisis

    This produces smooth transitions — no hard regime switching.
    """

    def __init__(self, books: dict[Regime, StrategyBook] = None):
        if books is None:
            books = STRATEGY_BOOKS
        self.books = books
        # Ensure ordering [BULL, TRANSITION, CRISIS]
        self._ordered_books = [self.books[Regime.BULL], 
                             self.books[Regime.TRANSITION], 
                             self.books[Regime.CRISIS]]

    def blend(self, regime_probs: np.ndarray) -> StrategyBook:
        """
        Produce a blended StrategyBook from soft regime probabilities.

        Args:
            regime_probs: array [p_bull, p_transition, p_crisis] summing to ~1

        Returns:
            StrategyBook with interpolated numerical parameters
        """
        regime_probs = np.array(regime_probs)
        regime_probs = regime_probs / (np.sum(regime_probs) + 1e-8)
        
        # For numerical params, interpolate. For boolean/string, use argmax regime.
        dominant_idx = int(np.argmax(regime_probs))
        dominant = Regime(dominant_idx)
        dom_book = self._ordered_books[dominant_idx]

        # Helper for interpolating numeric fields
        def blend_numeric(field_name):
            return sum(prob * getattr(book, field_name) 
                      for prob, book in zip(regime_probs, self._ordered_books))

        blended = StrategyBook(
            regime=dominant,
            
            # Interpolated numericals
            target_equity_pct=blend_numeric('target_equity_pct'),
            max_single_position=blend_numeric('max_single_position'),
            min_sectors_held=int(round(blend_numeric('min_sectors_held'))),
            momentum_lookback=int(round(blend_numeric('momentum_lookback'))),
            target_cash_pct=blend_numeric('target_cash_pct'),
            max_cash_pct=blend_numeric('max_cash_pct'),
            target_put_budget_pct=blend_numeric('target_put_budget_pct'),
            put_strike_otm_pct=blend_numeric('put_strike_otm_pct'),
            put_tenor_days=int(round(blend_numeric('put_tenor_days'))),
            divergence_lookback=int(round(blend_numeric('divergence_lookback'))),
            divergence_threshold=blend_numeric('divergence_threshold'),
            max_daily_turnover=blend_numeric('max_daily_turnover'),
            rebalance_speed=blend_numeric('rebalance_speed'),
            
            # Discrete / categorical from dominant book
            prefer_cyclical=dom_book.prefer_cyclical,
            prefer_defensive=dom_book.prefer_defensive,
            deploy_cash_when=dom_book.deploy_cash_when,
            put_action=dom_book.put_action,
            liquidate_divergent=dom_book.liquidate_divergent
        )
        
        return blended
