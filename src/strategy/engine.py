import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.strategy.regime_books import Regime
from src.strategy.regime_blender import RegimeBlender
from src.detectors.divergence_scanner import DivergenceScanner
from src.portfolio.cash_manager import CashManager
from src.options.put_overlay import PutOverlayManager
from src.portfolio.trade_ledger import TradeLedger, TradeType
from src.neural.hindsight_oracle import HindsightOracle

class StrategyEngine:
    """
    Central orchestrator that combines all components into trading decisions.

    At each trading day, the engine:
    1. Gets regime probabilities from the neural gate
    2. Blends strategy books using regime probs
    3. Runs divergence scanner
    4. Computes target equity weights (neural policy + regime book constraints)
    5. Computes target cash weight
    6. Updates put overlay (buy/hold/monetise based on book)
    7. Normalises all weights to sum to 1: equity + cash + puts = 1.0
    8. Applies rebalance speed (gradual transitions, not instant jumps)
    9. Records all trades in the ledger

    The engine REPLACES BasketManager.compute_weights() in the walk-forward loop.
    """

    def __init__(self, tickers: list[str]):
        self.tickers = sorted(tickers)
        self.n_assets = len(tickers)
        self.blender = RegimeBlender()
        self.divergence = DivergenceScanner(tickers)
        self.cash_mgr = CashManager()
        self.put_mgr = PutOverlayManager()
        self.ledger = TradeLedger()
        self.oracle = HindsightOracle()

        # Neural components (initialized in calibrate)
        self.policy = None
        self.imitation_trainer = None
        self.state_builder = None
        self.scaler = StandardScaler()

        # Running state
        self._portfolio_value = 1.0
        self._equity_curve = [1.0]
        self._peak_equity = 1.0
        self._current_dd = 0.0
        self._current_weights = {t: 0.0 for t in tickers}
        self._current_cash = 0.05
        self._is_trained = False
        self._window_count = 0
        self._daily_returns_history = []
        self._is_first_eval = True

    def calibrate(self, train_ret: pd.DataFrame, assignments: dict,
                  vol_dict: dict, signal_matrix: np.ndarray,
                  spy_ret: pd.Series, vix_prices: pd.Series,
                  rf_daily: float, log_ret_full: pd.DataFrame,
                  train_end_idx: int):
        """
        Called at each walk-forward rebalance point.
        """
        self._window_count += 1
        self._is_first_eval = True # reset for tracking eval seq
        
        # 1. Generate oracle training pairs
        day_indices = list(range(
            self.oracle.lookback + 63,
            train_end_idx,
            5  # every 5 days to save computation
        ))
        
        sector_cols = [t for t in self.tickers if t in train_ret.columns]
        
        # Manually building pairs because we need to build full states
        oracle_pairs = []
        for d_idx in day_indices:
            start = d_idx - self.oracle.lookback
            window_returns = train_ret[sector_cols].iloc[start:d_idx]
            if len(window_returns) < self.oracle.lookback - 5:
                continue
                
            # Dummy params for point-in-time state construction during training
            dummy_weights = {t: 1.0 / self.n_assets for t in self.tickers}
            dummy_p_stress = 0.0
            dummy_det = {'cusum': 0.0, 'ewma': 0.0, 'markov': 0.0, 'structural': 0.0}
            
            try:
                state = self.state_builder.build(
                    day_idx=start, log_ret=log_ret_full, 
                    current_weights=dummy_weights,
                    assignments=assignments, vol_dict=vol_dict,
                    p_stress=dummy_p_stress, detector_signals=dummy_det,
                    spy_ret=spy_ret, vix_prices=vix_prices, 
                    portfolio_returns=self._daily_returns_history,
                    days_since_rebalance=0
                )
                target = self.oracle.compute_optimal(window_returns)
                oracle_pairs.append((state, target))
            except Exception:
                continue

        # 2. Imitation learning (warm-start the policy from oracle)
        if len(oracle_pairs) > 50 and self._window_count >= 2 and self.imitation_trainer is not None:
            self.imitation_trainer.train(oracle_pairs)
            self._is_trained = True

        # 3. Fit scaler on training states
        if oracle_pairs:
            states = np.array([s for s, _ in oracle_pairs])
            self.scaler.fit(states)

        # 4. Reset LSTM hidden
        if self.policy is not None:
            self.policy.reset_hidden()


    def compute_allocation(self, day_idx: int, date: pd.Timestamp,
                           log_ret: pd.DataFrame, prices: pd.DataFrame,
                           regime_probs: np.ndarray, p_stress: float,
                           detector_signals: dict, assignments: dict,
                           vol_dict: dict, spy_price: float, vix: float,
                           rf_daily: float,
                           spy_ret: pd.Series, vix_prices: pd.Series, 
                           days_since_rebalance: int) -> dict[str, float]:
        """
        THE MAIN DECISION FUNCTION. Called once per trading day.
        """
        # 1. Blend strategy books
        book = self.blender.blend(regime_probs)
        dominant_regime = Regime(int(np.argmax(regime_probs))).name

        # 2. Divergence scan
        div_results = self.divergence.scan(
            log_ret, day_idx, lookback=book.divergence_lookback,
            threshold=book.divergence_threshold
        )

        # 3. Neural policy weights (or fallback)
        if self._is_trained and self.policy is not None:
            # We must pass the correct arguments expected by state_builder
            state = self.state_builder.build(
                day_idx=day_idx, log_ret=log_ret,
                current_weights=self._current_weights, 
                assignments=assignments,
                vol_dict=vol_dict, p_stress=p_stress, 
                detector_signals=detector_signals,
                spy_ret=spy_ret, vix_prices=vix_prices,
                portfolio_returns=self._daily_returns_history,
                days_since_rebalance=days_since_rebalance
            )
            state_scaled = self.scaler.transform(state.reshape(1, -1)).flatten()
            state_scaled = np.clip(state_scaled, -5, 5)
            # Add batch and seq_len dimensions
            state_tensor = torch.FloatTensor(state_scaled).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                out = self.policy(state_tensor)
                raw_weights = out[0] # The weights are the first element of tuple
            target_equity_w = raw_weights.squeeze().numpy()
        else:
            # Fallback: inverse-vol with momentum tilt
            target_equity_w = self._fallback_weights(assignments, vol_dict, p_stress, book)

        # 4. Apply regime book constraints
        target_equity_w = self._apply_constraints(target_equity_w, book)

        # 5. Penalise divergent sectors
        if book.liquidate_divergent:
            for i, ticker in enumerate(self.tickers):
                if ticker in div_results and div_results[ticker]['diverging_negative']:
                    freed_capital = target_equity_w[i] * 0.7  # reduce by 70%
                    target_equity_w[i] *= 0.3
                    
                    if day_idx > 0:
                        t_price = float(prices[ticker].iloc[day_idx]) if ticker in prices.columns else 0.0
                        self.ledger.record_trade(
                            date=date, trade_type=TradeType.DIVERGENCE_LIQUIDATION,
                            ticker=ticker, direction="SELL",
                            quantity=freed_capital,
                            price=t_price,
                            cost=freed_capital * 10 / 10000,  # 10 bps
                            regime=dominant_regime, p_stress=p_stress,
                            reason=f"Divergence z={div_results[ticker]['z_score']:.2f}",
                        )
                    
                    # Redistribute freed capital: equally to non-divergent assets
                    non_div = [j for j in range(len(self.tickers))
                              if not div_results.get(self.tickers[j], {}).get('diverging_negative', False)]
                    if non_div:
                        per_asset = (freed_capital * 0.5) / len(non_div) # distribute half back to equities
                        for j in non_div:
                            target_equity_w[j] += per_asset

        # Re-normalise equity weights
        eq_sum = target_equity_w.sum()
        if eq_sum > 0:
            target_equity_w /= eq_sum

        # 6. Cash target
        cash_target = self.cash_mgr.compute_target(book, p_stress, self._current_dd, vix)

        # 7. Put overlay
        put_actions = self.put_mgr.daily_update(
            date, spy_price, vix, rf_daily * 252,
            self._portfolio_value, book, dominant_regime
        )
        put_value_pct = self.put_mgr.get_total_value() / max(self._portfolio_value, 1e-6)

        # 8. Normalise: equity gets the remainder after cash and puts
        equity_budget = max(0.0, 1.0 - cash_target - put_value_pct)
        equity_budget = min(equity_budget, book.target_equity_pct)

        final_weights = {t: float(target_equity_w[i] * equity_budget)
                         for i, t in enumerate(self.tickers)}

        # 9. Apply rebalance speed (gradual transition)
        # Skip smoothing on first eval of the window, jump straight to target?
        # The prompt implies applying it always, dampening whipsaw.
        for t in self.tickers:
            old = self._current_weights.get(t, 0.0)
            new = final_weights[t]
            final_weights[t] = old + book.rebalance_speed * (new - old)

        # Clamp total to respect max_daily_turnover
        turnover = sum(abs(final_weights[t] - self._current_weights.get(t, 0.0))
                       for t in self.tickers)
        if turnover > book.max_daily_turnover:
            scale = book.max_daily_turnover / turnover
            for t in self.tickers:
                old = self._current_weights.get(t, 0.0)
                delta = final_weights[t] - old
                final_weights[t] = old + delta * scale

        # Force Normalization Check
        # final sum might slightly differ due to turnover dampening, 
        # but reallocate shortfall/surplus to cash to be safe
        final_eq_sum = sum(final_weights.values())
        self._current_cash = max(0.0, 1.0 - final_eq_sum - put_value_pct)

        # 10. Record trades
        if day_idx > 0:
            ticker_prices = {t: float(prices[t].iloc[day_idx])
                            for t in self.tickers if t in prices.columns}
            self.ledger.record_weight_changes(
                date, self._current_weights, final_weights,
                ticker_prices, dominant_regime, p_stress,
                10 / 10000,  # cost per unit turnover
                reason=f"regime={dominant_regime}, p_stress={p_stress:.3f}"
            )

        self._current_weights = dict(final_weights)

        self._is_first_eval = False
        return final_weights

    def record_daily_return(self, portfolio_return: float, rf_daily: float):
        """Called after each day to update running state."""
        # Cash return
        cash_ret = self._current_cash * rf_daily
        total_ret = portfolio_return + cash_ret

        self._daily_returns_history.append(total_ret)
        self._portfolio_value *= (1 + total_ret)
        self._peak_equity = max(self._peak_equity, self._portfolio_value)
        self._current_dd = (self._portfolio_value - self._peak_equity) / self._peak_equity

    def _fallback_weights(self, assignments, vol_dict, p_stress, book) -> np.ndarray:
        """Inverse-vol weights with regime-aware tilts when neural policy isn't trained."""
        weights = np.zeros(self.n_assets)
        # DEFENSIVE vs CYCLICAL
        DEFENSIVE = {'XLP', 'XLU', 'XLV'}
        CYCLICAL = {'XLY', 'XLF', 'XLI', 'XLK', 'XLE'}
        
        for i, t in enumerate(self.tickers):
            vol = vol_dict.get(t, 0.2)
            inv_vol = 1.0 / max(vol, 1e-6)
            
            # Regime tilt
            if book.prefer_defensive and t in DEFENSIVE:
                inv_vol *= 1.3
            elif book.prefer_cyclical and t in CYCLICAL:
                inv_vol *= 1.3
                
            # Stress scaling
            # Support both dict with string values and object values
            b = assignments.get(t)
            basket_val = getattr(b, 'basket', b) if b else "C"
                
            if basket_val == "B":
                inv_vol *= (1.0 - p_stress)
                
            weights[i] = inv_vol

        total = weights.sum()
        if total > 0:
            weights /= total
        return weights

    def _apply_constraints(self, weights: np.ndarray, book: 'StrategyBook') -> np.ndarray:
        """Enforce strategy book constraints on weights."""
        # Cap individual positions
        weights = np.minimum(weights, book.max_single_position)
        # Ensure minimum diversification
        nonzero = np.sum(weights > 0.01)
        if nonzero < book.min_sectors_held:
            # Force small allocations to zero-weight sectors
            zero_idx = np.where(weights <= 0.01)[0]
            needed = book.min_sectors_held - nonzero
            for idx in zero_idx[:int(needed)]:
                weights[idx] = 0.02  # small allocation
        # Re-normalise
        total = weights.sum()
        if total > 0:
            weights /= total
        return weights
