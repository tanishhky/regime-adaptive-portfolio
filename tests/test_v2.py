import unittest
import pandas as pd
import numpy as np
from src.options.black_scholes import bs_put_price
from src.options.put_overlay import PutOverlayManager
from src.portfolio.cash_manager import CashManager
from src.detectors.divergence_scanner import DivergenceScanner
from src.strategy.regime_books import BULL_BOOK, CRISIS_BOOK

class TestV2Components(unittest.TestCase):
    
    def test_black_scholes(self):
        # Near ATM put
        price = bs_put_price(S=100.0, K=95.0, T=0.5, sigma=0.20, r=0.05)
        self.assertGreater(price, 0.0)
        self.assertLess(price, 10.0)
        
        # ITM put expiring immediately
        price_exp = bs_put_price(S=90.0, K=100.0, T=0.0, sigma=0.20, r=0.05)
        self.assertAlmostEqual(price_exp, 10.0)

    def test_cash_manager(self):
        mgr = CashManager()
        # Bull book -> target cash should be close to base
        target = mgr.compute_target(BULL_BOOK, p_stress=0.1, current_dd=0.0, vix=15.0)
        self.assertAlmostEqual(target, BULL_BOOK.target_cash_pct + 0.1 * (BULL_BOOK.max_cash_pct - BULL_BOOK.target_cash_pct))

        # Crisis deep drawdown
        target_stress = mgr.compute_target(CRISIS_BOOK, p_stress=0.9, current_dd=-0.20, vix=40.0)
        self.assertGreater(target_stress, CRISIS_BOOK.target_cash_pct)

    def test_divergence_scanner(self):
        tickers = ["XLK", "XLF"]
        scanner = DivergenceScanner(tickers)
        
        # Synthetic data: SPY is up, XLK is up (normal), XLF is down (diverging negative)
        dates = pd.date_range("2020-01-01", periods=100)
        df = pd.DataFrame({
            "SPY": np.random.normal(0.001, 0.01, 100),
            "XLK": np.random.normal(0.001, 0.01, 100),
            "XLF": np.random.normal(-0.002, 0.015, 100), # persistent underperformance
        }, index=dates)
        
        # Force a specific shape to ensure z-score triggers
        df["XLF"].iloc[-42:] = -0.01
        df["SPY"].iloc[-42:] = 0.01
        
        res = scanner.scan(df, day_idx=99, lookback=42, threshold=-1.0)
        self.assertIn("XLF", res)
        self.assertIn("XLK", res)
        
        self.assertTrue(res["XLF"]["diverging_negative"] or res["XLF"]["z_score"] < 0)

if __name__ == "__main__":
    unittest.main()
