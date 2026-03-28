import numpy as np
from scipy.stats import norm

def bs_put_price(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """
    Black-Scholes European put option price.

    Args:
        S: current underlying price (SPY Close)
        K: strike price
        T: time to expiry in YEARS (e.g., 0.5 for 6 months)
        sigma: implied volatility (annualised, decimal — use VIX/100)
        r: risk-free rate (annualised, decimal — from Fama-French RF * 252)

    Returns:
        put_price: theoretical put price per share

    Point-in-time: all inputs are observable at time t. VIX is published
    in real time. RF is the prevailing risk-free rate. S and K are known.
    T decreases by 1/252 each trading day.
    """
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)  # intrinsic value at expiry

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(put, 0.0)


def bs_put_delta(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Put delta for position sizing and hedging."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def bs_put_vega(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Put vega — sensitivity to VIX changes. Key for P&L attribution."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)
