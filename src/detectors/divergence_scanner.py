import numpy as np
import pandas as pd

class DivergenceScanner:
    """
    Detects sectors that diverge from the broader market direction.

    A sector is "diverging negatively" when:
    1. SPY trailing return is positive (market is up)
    2. The sector's trailing return is significantly below SPY
    3. The underperformance z-score exceeds the threshold

    A sector is "diverging positively" (opportunity) when:
    1. SPY trailing return is negative (market is down)
    2. The sector is outperforming SPY (relative strength)
    3. This suggests the sector is resilient — good candidate for overweight

    Uses rolling z-score of (sector_return - spy_return) vs its own history.
    """

    def __init__(self, tickers: list[str]):
        self.tickers = sorted(tickers)

    def scan(self, log_ret: pd.DataFrame, day_idx: int,
             lookback: int = 42, threshold: float = -1.5,
             spy_ticker: str = "SPY") -> dict[str, dict]:
        """
        Scan all sectors for divergence at time t (strictly point-in-time).

        Args:
            log_ret: DataFrame of daily log returns (all tickers + SPY)
            day_idx: current day index (only uses data up to and including this index)
            lookback: rolling window in trading days
            threshold: z-score threshold for negative divergence flag

        Returns:
            dict mapping ticker -> {
                'relative_return': float,     # sector - SPY over lookback
                'z_score': float,             # how unusual this is
                'diverging_negative': bool,   # sector lagging while market up
                'diverging_positive': bool,   # sector leading while market down
                'spy_return': float,          # SPY return over lookback
                'sector_return': float,       # sector return over lookback
            }
        """
        if day_idx < lookback * 2:
            return {t: {'relative_return': 0.0, 'z_score': 0.0,
                        'diverging_negative': False, 'diverging_positive': False,
                        'spy_return': 0.0, 'sector_return': 0.0}
                    for t in self.tickers}

        # Strictly point-in-time: only use data up to day_idx
        data = log_ret.iloc[:day_idx + 1]
        spy_ret = data[spy_ticker].tail(lookback).sum()

        results = {}
        for ticker in self.tickers:
            if ticker not in data.columns:
                continue

            sector_ret = data[ticker].tail(lookback).sum()
            relative = sector_ret - spy_ret

            # Historical relative returns for z-score computation
            # Use a LONGER history to build the distribution
            history_len = min(len(data) - lookback, 504)
            
            # Vectorized rolling relative return calculation instead of loop for speed
            if history_len > 10:
                start_hist = max(0, len(data) - lookback - history_len)
                end_hist = len(data) - lookback
                
                # We need sliding window sums
                sec_hist = data[ticker].iloc[start_hist:end_hist+lookback-1].rolling(lookback).sum().dropna()
                spy_hist = data[spy_ticker].iloc[start_hist:end_hist+lookback-1].rolling(lookback).sum().dropna()
                
                rel_history = (sec_hist - spy_hist).values
                
                mu = np.mean(rel_history)
                sigma = np.std(rel_history)
                z = (relative - mu) / max(sigma, 1e-8)
            else:
                z = 0.0

            results[ticker] = {
                'relative_return': relative,
                'z_score': z,
                'diverging_negative': bool((spy_ret > 0) and (z < threshold)),
                'diverging_positive': bool((spy_ret < 0) and (z > abs(threshold))),
                'spy_return': spy_ret,
                'sector_return': sector_ret,
            }

        return results
